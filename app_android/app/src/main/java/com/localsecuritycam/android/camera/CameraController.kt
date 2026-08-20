package com.localsecuritycam.android.camera

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.os.Handler
import android.os.HandlerThread
import android.util.Range
import android.view.Surface
import com.localsecuritycam.android.diagnostics.CleanupCollector
import com.localsecuritycam.android.diagnostics.CleanupReport
import com.localsecuritycam.android.diagnostics.StreamErrorFormatter
import com.localsecuritycam.android.diagnostics.StreamErrorKind
import com.localsecuritycam.android.diagnostics.StreamErrorLogger
import com.localsecuritycam.android.diagnostics.StreamFailure
import com.localsecuritycam.android.diagnostics.StreamFailureException
import com.localsecuritycam.android.settings.CameraLens
import com.localsecuritycam.android.settings.StreamSettings
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

class CameraController(
    context: Context,
    private val previewDiagnosticMode: PreviewDiagnosticMode = PreviewDiagnosticMode.NORMAL,
    private val onFrame: () -> Unit,
    private val errorCallback: (StreamFailure) -> Unit,
    private val previewErrorCallback: (StreamFailure) -> Unit = {},
    private val previewRecoveredCallback: () -> Unit = {},
) {
    private val appContext = context.applicationContext
    private val manager = appContext.getSystemService(CameraManager::class.java)
    private var thread: HandlerThread? = null
    private var handler: Handler? = null
    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    @Volatile
    private var renderer: VideoFrameRenderer? = null
    private val stopping = AtomicBoolean(false)
    private var settings: StreamSettings? = null
    private var cameraCharacteristics: CameraCharacteristics? = null
    private var cameraLensFacing: CameraLens? = null
    private var onOpenedCallback: (() -> Unit)? = null
    private var captureReadyCallback: (() -> Unit)? = null
    private var captureGeneration: Long? = null
    private var captureStartedGeneration: Long? = null
    @Volatile
    private var previewSurface: PreviewSurfaceAttachment? = null
    @Volatile
    private var displayRotationDegrees: Int = 0
    private val generation = AtomicLong(0L)

    fun open(streamSettings: StreamSettings, onOpened: () -> Unit) {
        StreamErrorLogger.info("Camera opening")
        if (appContext.checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            throw StreamFailureException(
                StreamFailure(StreamErrorKind.PERMISSION, "camera permission is not granted", retryable = false),
            )
        }
        if (thread != null) {
            throw StreamFailureException(StreamFailure(StreamErrorKind.CAMERA, "camera already started"))
        }
        val token = generation.incrementAndGet()
        settings = streamSettings
        onOpenedCallback = onOpened
        captureReadyCallback = null
        captureGeneration = null
        captureStartedGeneration = null
        stopping.set(false)
        StreamErrorLogger.info("Camera opening generation=$token")
        val worker = HandlerThread("camera-capture")
        thread = worker
        try {
            worker.start()
            handler = Handler(worker.looper)
        } catch (error: Exception) {
            val cleanup = CleanupCollector()
            cleanup.runUnit("camera thread") { worker.quitSafely() }
            if (worker.isAlive && Thread.currentThread() !== worker) {
                cleanup.runUnit("camera thread join") { worker.join(1_000) }
            }
            thread = null
            handler = null
            throw StreamFailureException(
                StreamErrorFormatter.withCleanup(
                    StreamErrorFormatter.fromThrowable(StreamErrorKind.THREAD, error),
                    cleanup.report(),
                ),
            )
        }
        val cameraHandler = handler ?: throw StreamFailureException(
            StreamFailure(StreamErrorKind.THREAD, "camera handler unavailable"),
        )
        if (!cameraHandler.post {
            try {
                val capabilities = CameraCapabilitiesProvider(appContext).query(streamSettings.lens)
                val cameraId = capabilities.cameraId
                val characteristics = manager.getCameraCharacteristics(cameraId)
                val capabilityErrors = CameraCapabilitiesProvider.validationErrors(capabilities, streamSettings)
                if (capabilityErrors.isNotEmpty()) {
                    throw StreamFailureException(
                        StreamErrorFormatter.fromMessage(
                            StreamErrorKind.CONFIGURATION,
                            capabilityErrors.joinToString("; "),
                            retryable = false,
                        ),
                    )
                }
                cameraCharacteristics = characteristics
                cameraLensFacing = resolveLensFacing(characteristics) ?: streamSettings.lens
                StreamErrorLogger.info(
                    "Camera2 facts: camera_sensor_orientation=" +
                        "${characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0} " +
                        "lens_facing=${cameraLensFacing?.name?.lowercase()}",
                )
                manager.openCamera(cameraId, createStateCallback(token), cameraHandler)
            } catch (error: StreamFailureException) {
                reportError(error.failure)
            } catch (error: SecurityException) {
                reportError(StreamErrorFormatter.fromThrowable(StreamErrorKind.PERMISSION, error, retryable = false))
            } catch (error: Exception) {
                reportError(StreamErrorFormatter.fromThrowable(StreamErrorKind.CAMERA, error))
            }
        }) {
            throw StreamFailureException(StreamFailure(StreamErrorKind.THREAD, "camera open callback rejected"))
        }
    }

    fun outputTransform(deviceOrientation: DeviceOrientation): VideoTransform {
        val streamSettings = settings ?: throw StreamFailureException(
            StreamFailure(StreamErrorKind.CAPTURE_SESSION, "camera settings unavailable"),
        )
        val characteristics = cameraCharacteristics ?: throw StreamFailureException(
            StreamFailure(StreamErrorKind.CAPTURE_SESSION, "camera characteristics unavailable"),
        )
        val actualLensFacing = cameraLensFacing ?: resolveLensFacing(characteristics) ?: streamSettings.lens
        return computeVideoOutputTransform(
            sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0,
            deviceOrientation = deviceOrientation,
            lensFacing = actualLensFacing,
            sourceWidth = streamSettings.resolution.width,
            sourceHeight = streamSettings.resolution.height,
            outputAspectRatio = streamSettings.aspectRatio,
        )
    }

    fun startCapture(
        encoderInput: Surface?,
        preview: PreviewSurfaceAttachment?,
        initialTransform: VideoTransform,
        onReady: () -> Unit,
    ) {
        if (thread == null) {
            throw StreamFailureException(StreamFailure(StreamErrorKind.CAPTURE_SESSION, "camera has not been opened"))
        }
        if (captureReadyCallback != null || captureStartedGeneration != null) {
            throw StreamFailureException(StreamFailure(StreamErrorKind.CAPTURE_SESSION, "camera capture already started"))
        }
        val token = generation.get()
        if (stopping.get()) {
            throw StreamFailureException(StreamFailure(StreamErrorKind.CAPTURE_SESSION, "camera is stopping"))
        }
        captureReadyCallback = onReady
        captureGeneration = token
        captureStartedGeneration = token
        previewSurface = preview
        val cameraHandler = handler ?: throw StreamFailureException(
            StreamFailure(StreamErrorKind.THREAD, "camera handler unavailable"),
        )
        if (!cameraHandler.post {
            try {
                val streamSettings = settings ?: error("camera settings unavailable")
                val characteristics = cameraCharacteristics ?: error("camera characteristics unavailable")
                val camera = device ?: error("camera device is not open")
                val transform = initialTransform
                displayRotationDegrees = transform.orientation.displayRotationDegrees
                StreamErrorLogger.info(
                    "CAMERA_ORIENTATION sensor=${transform.orientation.sensorOrientationDegrees} " +
                        "physical=${transform.orientation.physicalOrientationDegrees ?: "fallback"} " +
                        "display=${transform.orientation.displayRotationDegrees} " +
                        "target_surface_rotation=${transform.orientation.targetSurfaceRotationDegrees} " +
                        "lens_facing=${transform.orientation.lensFacing.name.lowercase()} " +
                        "requested_rotation=${transform.rotationDegrees} " +
                        "pixel_clockwise_rotation=${transform.pixelClockwiseRotationDegrees} " +
                        "output_pixel_clockwise_rotation=${transform.outputPixelClockwiseRotationDegrees} " +
                        "output_aspect_ratio=${transform.orientation.outputAspectRatio?.label ?: "auto"} " +
                        "camera_gl_texture_rotation=${transform.glTextureRotationDegrees} " +
                        "mirror_preview=${transform.mirrorPreview} mirror_stream=${transform.mirrorStream} " +
                        "camera_buffer=${transform.sourceWidth}x${transform.sourceHeight} " +
                        "encoder_output=${transform.targetWidth}x${transform.targetHeight}",
                )
                val videoRenderer = VideoFrameRenderer(
                    streamSettings.resolution.width,
                    streamSettings.resolution.height,
                    initialTransform = transform,
                    diagnosticMode = previewDiagnosticMode,
                    onFrameRendered = { _ -> onFrame() },
                    onError = ::reportError,
                    onPreviewError = previewErrorCallback,
                    onPreviewRecovered = previewRecoveredCallback,
                )
                renderer = videoRenderer
                val cameraSurface = try {
                    videoRenderer.start(encoderInput, previewSurface)
                } catch (error: StreamFailureException) {
                    reportError(error.failure)
                    return@post
                } catch (error: Exception) {
                    reportError(StreamErrorFormatter.fromThrowable(StreamErrorKind.SURFACE, error))
                    return@post
                }
                // SurfaceView callbacks run on the Activity thread and may
                // race renderer creation on the camera thread. Re-apply the
                // latest generation after initialization so an attach that
                // arrived during VideoFrameRenderer.start() is not lost.
                videoRenderer.setPreviewSurface(previewSurface)
                try {
                    configureCaptureSession(camera, cameraSurface, characteristics, token)
                } catch (error: Exception) {
                    reportError(StreamErrorFormatter.fromThrowable(StreamErrorKind.CAPTURE_SESSION, error))
                }
            } catch (error: Exception) {
                reportError(StreamErrorFormatter.fromThrowable(StreamErrorKind.CAPTURE_SESSION, error))
            }
        }) {
            throw StreamFailureException(StreamFailure(StreamErrorKind.THREAD, "capture callback rejected"))
        }
    }

    fun setPreviewSurface(surface: PreviewSurfaceAttachment?) {
        val previous = previewSurface
        previewSurface = surface
        if (previous != surface) {
            StreamErrorLogger.info(
                if (surface == null) {
                    "Preview surface detached"
                } else {
                    "Preview surface attached size=${surface.width}x${surface.height}"
                },
            )
        }
        try {
            renderer?.setPreviewSurface(surface)
        } catch (error: Exception) {
            if (!stopping.get()) {
                reportPreviewError(StreamErrorFormatter.fromThrowable(StreamErrorKind.SURFACE, error))
            } else {
                StreamErrorLogger.cleanup(StreamErrorFormatter.cleanupFailure("preview Surface", error))
            }
        }
    }

    fun setEncoderInput(surface: Surface?) {
        val activeRenderer = renderer ?: throw StreamFailureException(
            StreamFailure(StreamErrorKind.SURFACE, "camera renderer is unavailable"),
        )
        activeRenderer.setEncoderSurface(surface)
    }

    fun applyVideoTransform(next: VideoTransform) {
        displayRotationDegrees = next.orientation.displayRotationDegrees
        renderer?.setTransform(next)
        StreamErrorLogger.info(
            "ORIENTATION_CHANGED source=${next.orientation.source.name.lowercase()} " +
                "physical=${next.orientation.physicalOrientationDegrees ?: "fallback"} " +
                "display=${next.orientation.displayRotationDegrees} " +
                "requested_rotation=${next.rotationDegrees} " +
                "pixel_clockwise_rotation=${next.pixelClockwiseRotationDegrees} " +
                "output_pixel_clockwise_rotation=${next.outputPixelClockwiseRotationDegrees} " +
                "output_aspect_ratio=${next.orientation.outputAspectRatio?.label ?: "auto"} " +
                "camera_gl_texture_rotation=${next.glTextureRotationDegrees} " +
                "camera_buffer=${next.sourceWidth}x${next.sourceHeight} " +
                "encoder_output=${next.targetWidth}x${next.targetHeight}",
        )
    }

    fun stop(): CleanupReport {
        val hadResources = thread != null || handler != null || device != null || session != null || renderer != null
        if (hadResources) StreamErrorLogger.info("Camera closing")
        stopping.set(true)
        val invalidatedGeneration = generation.incrementAndGet()
        StreamErrorLogger.info("Camera generation invalidated generation=$invalidatedGeneration")
        onOpenedCallback = null
        captureReadyCallback = null
        captureGeneration = null
        captureStartedGeneration = null
        val currentHandler = handler
        val cleanupStarted = AtomicBoolean(false)
        val stopped = CountDownLatch(1)
        val cleanupReport = CleanupCollector()
        fun cleanupResources() {
            if (!cleanupStarted.compareAndSet(false, true)) return
            val currentSession = session.also { session = null }
            cleanupReport.runUnit("camera capture session") { currentSession?.close() }
            val currentDevice = device.also { device = null }
            cleanupReport.runUnit("camera device") { currentDevice?.close() }
            val currentRenderer = renderer.also { renderer = null }
            cleanupReport.run("camera renderer") { currentRenderer?.stop() ?: CleanupReport() }
        }
        val cleanup = Runnable {
            try {
                cleanupResources()
            } finally {
                stopped.countDown()
            }
        }
        val cleanupCompleted = when {
            currentHandler == null -> {
                cleanup.run()
                true
            }
            Thread.currentThread() === thread -> {
                cleanup.run()
                true
            }
            currentHandler.post(cleanup) -> {
                val completed = stopped.await(2, java.util.concurrent.TimeUnit.SECONDS)
                if (!completed) {
                    cleanupReport.add(
                        StreamErrorFormatter.cleanupFailure(
                            "camera thread",
                            IllegalStateException("camera cleanup timed out"),
                        ),
                    )
                }
                completed
            }
            else -> {
                cleanupReport.add(
                    StreamErrorFormatter.cleanupFailure(
                        "camera thread",
                        IllegalStateException("camera cleanup callback rejected"),
                    ),
                )
                false
            }
        }
        if (!cleanupCompleted && currentHandler != null) {
            currentHandler.removeCallbacks(cleanup)
            cleanup.run()
        }
        val currentThread = thread
        cleanupReport.runUnit("camera thread") { currentThread?.quitSafely() }
        if (currentThread != null && Thread.currentThread() !== currentThread) {
            cleanupReport.runUnit("camera thread join") { currentThread.join(1_000) }
            if (currentThread.isAlive) {
                cleanupReport.add(
                    StreamErrorFormatter.cleanupFailure(
                        "camera thread join",
                        IllegalStateException("camera thread did not stop"),
                    ),
                )
            }
        }
        handler = null
        thread = null
        cameraCharacteristics = null
        cameraLensFacing = null
        previewSurface = null
        settings = null
        val report = cleanupReport.report()
        if (hadResources && report.isSuccessful) StreamErrorLogger.info("Camera closed")
        return report
    }

    private fun configureCaptureSession(
        camera: CameraDevice,
        surface: Surface,
        characteristics: CameraCharacteristics,
        token: Long,
    ) {
        val cameraHandler = handler ?: throw StreamFailureException(
            StreamFailure(StreamErrorKind.THREAD, "camera handler unavailable"),
        )
        StreamErrorLogger.info("Capture session starting")
        camera.createCaptureSession(
            listOf(surface),
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(captureSession: CameraCaptureSession) {
                    if (!isCurrent(token) || camera !== device) {
                        StreamErrorLogger.info("Stale camera callback ignored generation=$token")
                        try {
                            captureSession.close()
                        } catch (error: Exception) {
                            StreamErrorLogger.cleanup(StreamErrorFormatter.cleanupFailure("camera capture session", error))
                        }
                        return
                    }
                    session = captureSession
                    try {
                        val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_RECORD).apply {
                            addTarget(surface)
                            set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)
                            val fps = selectFpsRange(characteristics, settings?.fps ?: 20)
                            if (fps != null) set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fps)
                            val afModes: IntArray =
                                characteristics.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES)
                                    ?: intArrayOf()
                            if (afModes.contains(CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO)) {
                                set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO)
                            }
                        }.build()
                        captureSession.setRepeatingRequest(request, null, cameraHandler)
                        StreamErrorLogger.info(
                            "camera_session_configured generation=$token targets=1 camera_surface=1",
                        )
                        StreamErrorLogger.info("Capture session active generation=$token")
                        if (captureGeneration == token) {
                            val callback = captureReadyCallback
                            captureReadyCallback = null
                            captureGeneration = null
                            callback?.invoke()
                        }
                    } catch (error: Exception) {
                        if (!stopping.get()) {
                            reportError(StreamErrorFormatter.fromThrowable(StreamErrorKind.CAPTURE_SESSION, error))
                        }
                    }
                }

                override fun onConfigureFailed(captureSession: CameraCaptureSession) {
                    try {
                        captureSession.close()
                    } catch (error: Exception) {
                        StreamErrorLogger.cleanup(StreamErrorFormatter.cleanupFailure("camera capture session", error))
                    }
                    if (isCurrent(token)) {
                        captureReadyCallback = null
                        captureGeneration = null
                        captureStartedGeneration = null
                        reportError(
                            StreamErrorFormatter.fromMessage(
                                StreamErrorKind.CAPTURE_SESSION,
                                "camera capture session configuration failed",
                            ),
                        )
                    }
                }
            },
            cameraHandler,
        )
    }

    private fun createStateCallback(token: Long): CameraDevice.StateCallback = object : CameraDevice.StateCallback() {
        override fun onOpened(camera: CameraDevice) {
            if (!isCurrent(token)) {
                StreamErrorLogger.info("Stale camera callback ignored generation=$token")
                try {
                    camera.close()
                } catch (error: Exception) {
                    StreamErrorLogger.cleanup(StreamErrorFormatter.cleanupFailure("camera device", error))
                }
                return
            }
            device = camera
            StreamErrorLogger.info("Camera opened generation=$token")
            try {
                val callback = onOpenedCallback
                onOpenedCallback = null
                callback?.invoke()
            } catch (error: Exception) {
                reportError(StreamErrorFormatter.fromThrowable(StreamErrorKind.THREAD, error))
            }
        }

        override fun onDisconnected(camera: CameraDevice) {
            try {
                camera.close()
            } catch (error: Exception) {
                StreamErrorLogger.cleanup(StreamErrorFormatter.cleanupFailure("camera device", error))
            }
            if (!isCurrent(token)) {
                StreamErrorLogger.info("Stale camera callback ignored generation=$token")
                return
            }
            if (device === camera) device = null
            if (!stopping.get()) {
                this@CameraController.reportError(
                    StreamErrorFormatter.fromMessage(StreamErrorKind.CAMERA, "camera disconnected"),
                )
            }
        }

        override fun onError(camera: CameraDevice, error: Int) {
            try {
                camera.close()
            } catch (closeError: Exception) {
                StreamErrorLogger.cleanup(StreamErrorFormatter.cleanupFailure("camera device", closeError))
            }
            if (!isCurrent(token)) {
                StreamErrorLogger.info("Stale camera callback ignored generation=$token")
                return
            }
            if (device === camera) device = null
            if (!stopping.get()) {
                this@CameraController.reportError(
                    StreamErrorFormatter.fromMessage(StreamErrorKind.CAMERA, "camera error code $error"),
                )
            }
        }
    }

    private fun isCurrent(token: Long): Boolean = !stopping.get() && token == generation.get()

    private fun resolveLensFacing(characteristics: CameraCharacteristics): CameraLens? = when (
        characteristics.get(CameraCharacteristics.LENS_FACING)
    ) {
        CameraCharacteristics.LENS_FACING_BACK -> CameraLens.BACK
        CameraCharacteristics.LENS_FACING_FRONT -> CameraLens.FRONT
        else -> null
    }

    private fun reportError(failure: StreamFailure) {
        try {
            errorCallback(failure)
        } catch (error: Exception) {
            StreamErrorLogger.observer(error)
        }
    }

    private fun reportPreviewError(failure: StreamFailure) {
        try {
            previewErrorCallback(failure)
        } catch (error: Exception) {
            StreamErrorLogger.observer(error)
        }
    }

    private fun selectFpsRange(characteristics: CameraCharacteristics, target: Int): Range<Int>? {
        val ranges: Array<Range<Int>> =
            characteristics.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)
                ?: emptyArray()
        return ranges.filter { target in it.lower..it.upper }
            .minWithOrNull(compareBy<Range<Int>> { it.upper - it.lower }.thenBy { it.upper })
            ?: ranges.minByOrNull { kotlin.math.abs(it.upper - target) }
    }
}
