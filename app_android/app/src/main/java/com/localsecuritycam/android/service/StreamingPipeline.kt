package com.localsecuritycam.android.service

import android.content.Context
import android.view.Surface
import com.localsecuritycam.android.camera.CameraController
import com.localsecuritycam.android.camera.CameraOrientationState
import com.localsecuritycam.android.camera.DeviceOrientation
import com.localsecuritycam.android.camera.PreviewDiagnosticMode
import com.localsecuritycam.android.camera.PreviewSurfaceAttachment
import com.localsecuritycam.android.camera.VideoTransform
import com.localsecuritycam.android.diagnostics.CleanupCollector
import com.localsecuritycam.android.diagnostics.CleanupReport
import com.localsecuritycam.android.diagnostics.StreamErrorFormatter
import com.localsecuritycam.android.diagnostics.StreamErrorKind
import com.localsecuritycam.android.diagnostics.StreamErrorLogger
import com.localsecuritycam.android.diagnostics.StreamFailure
import com.localsecuritycam.android.diagnostics.StreamFailureException
import com.localsecuritycam.android.diagnostics.StreamMetrics
import com.localsecuritycam.android.diagnostics.StreamSubsystem
import com.localsecuritycam.android.diagnostics.SubsystemState
import com.localsecuritycam.android.encoder.H264Encoder
import com.localsecuritycam.android.settings.AppSettings
import com.localsecuritycam.android.settings.Resolution
import com.localsecuritycam.android.settings.StreamSettings
import com.localsecuritycam.android.streaming.EncodedAccessUnit
import com.localsecuritycam.android.streaming.H264ParameterSets
import com.localsecuritycam.android.streaming.RtspCredentials
import com.localsecuritycam.android.streaming.RtspServer
import com.localsecuritycam.android.streaming.StreamBroadcaster
import java.net.InetAddress
import java.util.concurrent.Executors

internal interface EncoderInput

internal class AndroidEncoderInput(val surface: Surface) : EncoderInput

internal data class StreamPipelineRequest(
    val settings: AppSettings,
    val preview: PreviewSurfaceAttachment?,
    val displayRotationDegrees: Int = 0,
    val initialOrientation: DeviceOrientation = DeviceOrientation(displayRotationDegrees = displayRotationDegrees),
    val initialReconnectCount: Long,
    val initialSessionRestartCount: Long,
    val previewDiagnosticMode: PreviewDiagnosticMode = PreviewDiagnosticMode.NORMAL,
)

internal data class StreamPipelineCallbacks(
    val onPreviewReady: () -> Unit,
    val onStreamReady: () -> Unit,
    val onError: (PipelineStage, StreamFailure) -> Unit,
    val onPreviewDiagnostic: (StreamFailure) -> Unit = {},
    val onPreviewRecovered: () -> Unit = {},
    val onSubsystemStateChanged: (StreamSubsystem, SubsystemState) -> Unit = { _, _ -> },
    val onOrientationChanged: (CameraOrientationState) -> Unit = {},
)

internal enum class PipelineStage {
    PREVIEW,
    STREAM,
}

internal interface StreamPipeline {
    val metrics: StreamMetrics

    fun startPreview()

    fun startStreaming(bindAddress: InetAddress)

    fun stopStreaming(): CleanupReport

    fun stop(): CleanupReport

    fun setPreviewSurface(surface: PreviewSurfaceAttachment?)

    fun setOrientation(orientation: DeviceOrientation) = Unit

    /** Activity display state is diagnostics/fallback only; physical input wins. */
    fun setDisplayRotation(rotationDegrees: Int) = setOrientation(
        DeviceOrientation(displayRotationDegrees = ((rotationDegrees % 360) + 360) % 360),
    )
}

internal fun interface StreamPipelineFactory {
    fun create(request: StreamPipelineRequest, callbacks: StreamPipelineCallbacks): StreamPipeline
}

internal interface CameraPort {
    fun open(settings: StreamSettings, onOpened: () -> Unit)

    fun outputTransform(orientation: DeviceOrientation): VideoTransform

    fun startCapture(
        encoderInput: EncoderInput?,
        preview: PreviewSurfaceAttachment?,
        initialTransform: VideoTransform,
        onReady: () -> Unit,
    )

    fun setPreviewSurface(surface: PreviewSurfaceAttachment?)

    /** Detaches/attaches only the EGL target for a MediaCodec input Surface. */
    fun setEncoderInput(encoderInput: EncoderInput?) = Unit

    fun applyVideoTransform(transform: VideoTransform) = Unit

    fun stop(): CleanupReport
}

internal interface EncoderPort {
    fun start(): EncoderInput

    fun requestSyncFrame() = Unit

    fun stop(): CleanupReport
}

internal interface RtspPort {
    fun start(bindAddress: InetAddress)

    fun stop(): CleanupReport
}

internal interface StreamingResourceFactory {
    fun createCamera(
        onFrame: () -> Unit,
        onError: (StreamFailure) -> Unit,
        onPreviewError: (StreamFailure) -> Unit,
        onPreviewRecovered: () -> Unit,
        previewDiagnosticMode: PreviewDiagnosticMode = PreviewDiagnosticMode.NORMAL,
    ): CameraPort

    fun createEncoder(
        settings: StreamSettings,
        onAccessUnit: (EncodedAccessUnit) -> Unit,
        onParameterSets: (H264ParameterSets) -> Unit,
        onOutputFormat: (Resolution) -> Unit,
        onError: (StreamFailure) -> Unit,
    ): EncoderPort

    fun createServer(
        settings: StreamSettings,
        metrics: StreamMetrics,
        broadcaster: StreamBroadcaster,
        credentialsProvider: () -> RtspCredentials?,
        onError: (StreamFailure) -> Unit,
    ): RtspPort
}

internal class AndroidStreamingResourceFactory(
    private val context: Context,
) : StreamingResourceFactory {
    override fun createCamera(
        onFrame: () -> Unit,
        onError: (StreamFailure) -> Unit,
        onPreviewError: (StreamFailure) -> Unit,
        onPreviewRecovered: () -> Unit,
        previewDiagnosticMode: PreviewDiagnosticMode,
    ): CameraPort {
        val controller = CameraController(
            context = context,
            previewDiagnosticMode = previewDiagnosticMode,
            onFrame = onFrame,
            errorCallback = onError,
            previewErrorCallback = onPreviewError,
            previewRecoveredCallback = onPreviewRecovered,
        )
        return object : CameraPort {
            override fun open(settings: StreamSettings, onOpened: () -> Unit) = controller.open(settings, onOpened)

            override fun outputTransform(orientation: DeviceOrientation): VideoTransform =
                controller.outputTransform(orientation)

            override fun startCapture(
                encoderInput: EncoderInput?,
                preview: PreviewSurfaceAttachment?,
                initialTransform: VideoTransform,
                onReady: () -> Unit,
            ) {
                val input = encoderInput as? AndroidEncoderInput
                if (encoderInput != null && input == null) {
                    throw StreamFailureException(
                        StreamFailure(StreamErrorKind.SURFACE, "unsupported encoder input surface"),
                    )
                }
                controller.startCapture(input?.surface, preview, initialTransform, onReady)
            }

            override fun setPreviewSurface(surface: PreviewSurfaceAttachment?) = controller.setPreviewSurface(surface)

            override fun setEncoderInput(encoderInput: EncoderInput?) {
                val input = encoderInput as? AndroidEncoderInput
                if (encoderInput != null && input == null) {
                    throw StreamFailureException(
                        StreamFailure(StreamErrorKind.SURFACE, "unsupported encoder input surface"),
                    )
                }
                controller.setEncoderInput(input?.surface)
            }

            override fun applyVideoTransform(transform: VideoTransform) = controller.applyVideoTransform(transform)

            override fun stop(): CleanupReport = controller.stop()
        }
    }

    override fun createEncoder(
        settings: StreamSettings,
        onAccessUnit: (EncodedAccessUnit) -> Unit,
        onParameterSets: (H264ParameterSets) -> Unit,
        onOutputFormat: (Resolution) -> Unit,
        onError: (StreamFailure) -> Unit,
    ): EncoderPort {
        val encoder = H264Encoder(
            settings = settings,
            onAccessUnit = onAccessUnit,
            onParameterSets = onParameterSets,
            onOutputFormat = { format -> onOutputFormat(Resolution(format.width, format.height)) },
            onError = onError,
        )
        return object : EncoderPort {
            override fun start(): EncoderInput = AndroidEncoderInput(encoder.start())

            override fun requestSyncFrame() = encoder.requestSyncFrame()

            override fun stop(): CleanupReport = encoder.stop()
        }
    }

    override fun createServer(
        settings: StreamSettings,
        metrics: StreamMetrics,
        broadcaster: StreamBroadcaster,
        credentialsProvider: () -> RtspCredentials?,
        onError: (StreamFailure) -> Unit,
    ): RtspPort {
        val server = RtspServer(settings, metrics, broadcaster, credentialsProvider, onError)
        return object : RtspPort {
            override fun start(bindAddress: InetAddress) = server.start(bindAddress)

            override fun stop(): CleanupReport = server.stop()
        }
    }
}

/**
 * Owns one Camera2 lifecycle and swaps only MediaCodec/RTSP generations when
 * the oriented output changes aspect class. Orientation requests are coalesced
 * on a serial executor; stale encoder callbacks carry a generation token.
 */
internal class StreamingPipeline(
    private val request: StreamPipelineRequest,
    private val callbacks: StreamPipelineCallbacks,
    private val resources: StreamingResourceFactory,
    private val dispatch: ((() -> Unit) -> Boolean),
    private val onMetricsChanged: () -> Unit,
) : StreamPipeline {
    private val lock = Any()
    override val metrics = StreamMetrics(
        request.initialReconnectCount,
        request.initialSessionRestartCount,
    )
    private var camera: CameraPort? = null
    private var output: OutputResources? = null
    private val orientationExecutor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "stream-orientation-reconfigure").apply { isDaemon = true }
    }
    private var pendingOrientation: DeviceOrientation? = null
    private var orientationTaskScheduled = false
    private var currentOrientation = request.initialOrientation
    private var currentTransform: VideoTransform? = null
    private var previewSurface = request.preview
    private var nextOutputGeneration = 0L
    private var previewReady = false
    private var streamRequestedBindAddress: InetAddress? = null
    @Volatile
    private var started = false
    @Volatile
    private var terminated = false

    override fun startPreview() {
        synchronized(lock) {
            check(!started) { "streaming pipeline already started" }
            check(!terminated) { "streaming pipeline already stopped" }
            started = true
            metrics.start()
            StreamErrorLogger.info("PREVIEW_START requested")
            notifySubsystemState(StreamSubsystem.CAMERA, SubsystemState.STARTING)
            try {
                val cameraResource = resources.createCamera(
                    onFrame = {
                        if (!terminated) {
                            metrics.recordCameraFrame()
                            notifyMetricsChanged()
                        }
                    },
                    onError = ::reportError,
                    onPreviewError = ::reportPreviewDiagnostic,
                    onPreviewRecovered = ::reportPreviewRecovered,
                    previewDiagnosticMode = request.previewDiagnosticMode,
                )
                camera = cameraResource
                cameraResource.open(request.settings.stream) {
                    dispatchOrFail(::startPreviewCapture, "camera-open")
                }
            } catch (error: StreamFailureException) {
                failPreview(error.failure)
            } catch (error: Exception) {
                failPreview(StreamErrorFormatter.fromThrowable(StreamErrorKind.CAMERA, error))
            }
        }
    }

    private fun startPreviewCapture() {
        synchronized(lock) {
            if (!isActive()) return
            val cameraResource = camera ?: run {
                failPreview(StreamFailure(StreamErrorKind.CAPTURE_SESSION, "camera resource is unavailable"))
                return
            }
            val initialTransform = try {
                cameraResource.outputTransform(currentOrientation)
            } catch (error: StreamFailureException) {
                failPreview(error.failure)
                return
            } catch (error: Exception) {
                failPreview(StreamErrorFormatter.fromThrowable(StreamErrorKind.CAPTURE_SESSION, error))
                return
            }
            currentTransform = initialTransform
            notifyOrientationChanged(initialTransform.orientation)
            try {
                cameraResource.startCapture(null, previewSurface, initialTransform) {
                    dispatchOrFail(
                        action = ::onPreviewCaptureReady,
                        context = "preview-capture-ready",
                    )
                }
            } catch (error: StreamFailureException) {
                failPreview(error.failure)
            } catch (error: Exception) {
                failPreview(StreamErrorFormatter.fromThrowable(StreamErrorKind.CAPTURE_SESSION, error))
            }
        }
    }

    private fun onPreviewCaptureReady() {
        synchronized(lock) {
            if (!isActive() || previewReady) return
            previewReady = true
            notifySubsystemState(StreamSubsystem.CAMERA, SubsystemState.RUNNING)
            try {
                callbacks.onPreviewReady()
            } catch (error: Exception) {
                StreamErrorLogger.observer(error)
            }
            streamRequestedBindAddress?.let { bindAddress ->
                startOutputIfReadyLocked(bindAddress)
            }
        }
    }

    override fun startStreaming(bindAddress: InetAddress) {
        synchronized(lock) {
            if (!isActive()) return
            try {
                request.settings.stream.requireValid(
                    if (request.settings.stream.authEnabled) request.settings.password else null,
                )
                validateRtspStart(bindAddress)
                streamRequestedBindAddress = bindAddress
                if (previewReady) startOutputIfReadyLocked(bindAddress)
            } catch (error: StreamFailureException) {
                failStream(error.failure)
            } catch (error: Exception) {
                failStream(StreamErrorFormatter.fromThrowable(StreamErrorKind.CONFIGURATION, error))
            }
        }
    }

    private fun validateRtspStart(bindAddress: InetAddress) {
        val credentials = if (request.settings.stream.authEnabled) {
            RtspCredentials(
                enabled = true,
                username = request.settings.stream.username,
                password = request.settings.password.orEmpty(),
            )
        } else {
            null
        }
        RtspServer.validateStartConfiguration(
            settings = request.settings.stream,
            bindAddress = bindAddress,
            credentials = credentials,
        )?.let { throw StreamFailureException(it) }
    }

    private fun startOutputIfReadyLocked(bindAddress: InetAddress) {
        if (!isActive() || !previewReady || output != null) return
        val cameraResource = camera ?: run {
            failPreview(StreamFailure(StreamErrorKind.CAPTURE_SESSION, "camera resource is unavailable"))
            return
        }
        val transform = currentTransform ?: try {
            cameraResource.outputTransform(currentOrientation).also {
                currentTransform = it
                notifyOrientationChanged(it.orientation)
            }
        } catch (error: StreamFailureException) {
            failPreview(error.failure)
            return
        } catch (error: Exception) {
            failPreview(StreamErrorFormatter.fromThrowable(StreamErrorKind.CAPTURE_SESSION, error))
            return
        }
        try {
            val activeOutput = startOutputLocked(transform, bindAddress)
            val encoderInput = activeOutput.encoderInput
            cameraResource.setEncoderInput(encoderInput)
            StreamErrorLogger.info("STREAM_ENCODER_ATTACHED generation=${activeOutput.generation}")
        } catch (error: StreamFailureException) {
            failStream(error.failure)
        } catch (error: Exception) {
            failStream(StreamErrorFormatter.fromThrowable(StreamErrorKind.MEDIACODEC, error))
        }
    }

    /** Creates a new encoder only. RTSP begins after MediaCodec confirms its actual dimensions. */
    private fun startOutputLocked(transform: VideoTransform, bindAddress: InetAddress): OutputResources {
        check(isActive()) { "streaming pipeline is not active" }
        val generation = ++nextOutputGeneration
        val outputSettings = request.settings.stream.copy(resolution = transform.outputResolution)
        val broadcaster = StreamBroadcaster(metrics)
        notifySubsystemState(StreamSubsystem.ENCODER, SubsystemState.STARTING)
        val encoder = resources.createEncoder(
            settings = outputSettings,
            onAccessUnit = { unit ->
                if (isCurrentOutputGeneration(generation)) {
                    metrics.recordEncodedFrame()
                    broadcaster.publish(unit)
                    notifyMetricsChanged()
                } else {
                    StreamErrorLogger.info("Stale encoder access unit ignored generation=$generation")
                }
            },
            onParameterSets = { sets ->
                if (isCurrentOutputGeneration(generation)) {
                    broadcaster.setParameterSets(sets)
                } else {
                    StreamErrorLogger.info("Stale encoder SPS/PPS ignored generation=$generation")
                }
            },
            onOutputFormat = { actual ->
                dispatchOrFail(
                    action = { onEncoderOutputFormat(generation, actual) },
                    context = "encoder-output-format",
                )
            },
            onError = { failure ->
                if (isCurrentOutputGeneration(generation)) reportStreamError(failure)
                else StreamErrorLogger.info("Stale encoder callback ignored generation=$generation")
            },
        )
        val activeOutput = OutputResources(
            generation = generation,
            settings = outputSettings,
            transform = transform,
            bindAddress = bindAddress,
            broadcaster = broadcaster,
            encoder = encoder,
        )
        output = activeOutput
        val encoderInput = encoder.start()
        activeOutput.encoderInput = encoderInput
        notifySubsystemState(StreamSubsystem.ENCODER, SubsystemState.RUNNING)
        StreamErrorLogger.info("STREAM_ENCODER_START generation=$generation")
        return activeOutput
    }

    private fun onEncoderOutputFormat(generation: Long, actualResolution: Resolution) {
        synchronized(lock) {
            val activeOutput = output
            if (!isActive() || activeOutput?.generation != generation) {
                StreamErrorLogger.info("Stale MediaCodec format callback ignored generation=$generation")
                return
            }
            if (actualResolution != activeOutput.settings.resolution) {
                failStream(
                    StreamFailure(
                        StreamErrorKind.MEDIACODEC,
                        "MediaCodec output geometry mismatch requested=${activeOutput.settings.resolution} " +
                            "actual=$actualResolution",
                        retryable = false,
                    ),
                )
                return
            }
            if (activeOutput.formatValidated) return
            activeOutput.formatValidated = true
            StreamErrorLogger.info(
                "encoder_output_ready generation=$generation " +
                    "resolution=${actualResolution.width}x${actualResolution.height}",
            )
            try {
                startServerLocked(activeOutput)
                activeOutput.encoder.requestSyncFrame()
                maybeNotifyReadyLocked(activeOutput)
            } catch (error: StreamFailureException) {
                failStream(error.failure)
            } catch (error: Exception) {
                failStream(StreamErrorFormatter.fromRtspThrowable(error))
            }
        }
    }

    private fun startServerLocked(activeOutput: OutputResources) {
        if (activeOutput.server != null) return
        notifySubsystemState(StreamSubsystem.RTSP_SERVER, SubsystemState.STARTING)
        val server = resources.createServer(
            settings = activeOutput.settings,
            metrics = metrics,
            broadcaster = activeOutput.broadcaster,
            credentialsProvider = {
                if (request.settings.stream.authEnabled) {
                    RtspCredentials(
                        true,
                        request.settings.stream.username,
                        request.settings.password.orEmpty(),
                    )
                } else {
                    null
                }
            },
            onError = { failure ->
                if (isCurrentOutputGeneration(activeOutput.generation)) reportStreamError(failure)
                else StreamErrorLogger.info("Stale RTSP callback ignored generation=${activeOutput.generation}")
            },
        )
        activeOutput.server = server
        server.start(activeOutput.bindAddress)
        notifySubsystemState(StreamSubsystem.RTSP_SERVER, SubsystemState.RUNNING)
        StreamErrorLogger.info(
            "RTSP output active generation=${activeOutput.generation} sdp=${activeOutput.settings.resolution}",
        )
    }

    private fun maybeNotifyReadyLocked(activeOutput: OutputResources) {
        if (!previewReady || !activeOutput.formatValidated || activeOutput.server == null || activeOutput.readyNotified) return
        activeOutput.readyNotified = true
        try {
            callbacks.onStreamReady()
        } catch (error: Exception) {
            StreamErrorLogger.observer(error)
        }
    }

    private fun reconfigureOutput(orientation: DeviceOrientation) {
        synchronized(lock) {
            currentOrientation = orientation
            if (!isActive()) return
            if (!previewReady && currentTransform == null) {
                // The physical listener may win the race against Camera2.open.
                // Keep its latest state; startPreviewCapture will use it as
                // soon as camera characteristics are available.
                StreamErrorLogger.info("ORIENTATION_CHANGED queued_until_camera_open=true")
                return
            }
            val cameraResource = camera ?: return
            val nextTransform = try {
                cameraResource.outputTransform(orientation)
            } catch (error: StreamFailureException) {
                failPreview(error.failure)
                return
            } catch (error: Exception) {
                failPreview(StreamErrorFormatter.fromThrowable(StreamErrorKind.CAPTURE_SESSION, error))
                return
            }
            val activeOutput = output
            if (activeOutput == null) {
                try {
                    cameraResource.applyVideoTransform(nextTransform)
                    currentTransform = nextTransform
                    notifyOrientationChanged(nextTransform.orientation)
                    StreamErrorLogger.info(
                        "ORIENTATION_CHANGED preview_only=true rotation=${nextTransform.rotationDegrees}",
                    )
                } catch (error: Exception) {
                    failPreview(StreamErrorFormatter.fromThrowable(StreamErrorKind.SURFACE, error))
                }
                return
            }
            if (activeOutput.settings.resolution == nextTransform.outputResolution) {
                try {
                    cameraResource.applyVideoTransform(nextTransform)
                    currentTransform = nextTransform
                    notifyOrientationChanged(nextTransform.orientation)
                    StreamErrorLogger.info(
                        "ORIENTATION_CHANGED generation=${activeOutput.generation} transform_only=true " +
                            "rotation=${nextTransform.rotationDegrees} output=${nextTransform.outputResolution}",
                    )
                } catch (error: Exception) {
                    failStream(StreamErrorFormatter.fromThrowable(StreamErrorKind.SURFACE, error))
                }
                return
            }

            StreamErrorLogger.info(
                "ENCODER_RECONFIGURE old=${activeOutput.settings.resolution} new=${nextTransform.outputResolution} " +
                    "rotation=${nextTransform.rotationDegrees} generation=${activeOutput.generation + 1}",
            )
            try {
                // Keep Camera2 and its SurfaceTexture alive; only the encoder EGL
                // target, RTSP sessions and MediaCodec generation are replaced.
                output = null
                stopOutput(activeOutput)
                val newOutput = startOutputLocked(nextTransform, activeOutput.bindAddress)
                cameraResource.applyVideoTransform(nextTransform)
                cameraResource.setEncoderInput(newOutput.encoderInput)
                currentTransform = nextTransform
                notifyOrientationChanged(nextTransform.orientation)
                maybeNotifyReadyLocked(newOutput)
                StreamErrorLogger.info(
                    "RTSP_RECONNECT_REQUIRED generation=${newOutput.generation} sdp=${newOutput.settings.resolution}",
                )
            } catch (error: StreamFailureException) {
                failStream(error.failure)
            } catch (error: Exception) {
                failStream(StreamErrorFormatter.fromThrowable(StreamErrorKind.MEDIACODEC, error))
            }
        }
    }

    private fun drainOrientationRequests() {
        while (true) {
            val next = synchronized(lock) {
                if (!isActive()) {
                    pendingOrientation = null
                    orientationTaskScheduled = false
                    return
                }
                val candidate = pendingOrientation
                pendingOrientation = null
                if (candidate == null) {
                    orientationTaskScheduled = false
                    return
                }
                candidate
            }
            reconfigureOutput(next)
        }
    }

    override fun setOrientation(orientation: DeviceOrientation) {
        var schedule = false
        synchronized(lock) {
            if (!isActive()) return
            pendingOrientation = orientation
            if (!orientationTaskScheduled) {
                orientationTaskScheduled = true
                schedule = true
            }
        }
        if (schedule) {
            try {
                orientationExecutor.execute(::drainOrientationRequests)
            } catch (error: Exception) {
                synchronized(lock) {
                    orientationTaskScheduled = false
                    pendingOrientation = null
                }
                if (isActive()) failPreview(StreamErrorFormatter.fromThrowable(StreamErrorKind.THREAD, error))
            }
        }
    }

    override fun setDisplayRotation(rotationDegrees: Int) {
        setOrientation(
            currentOrientation.copy(displayRotationDegrees = normalizeRotation(rotationDegrees)),
        )
    }

    override fun setPreviewSurface(surface: PreviewSurfaceAttachment?) {
        synchronized(lock) {
            if (!isActive()) return
            previewSurface = surface
            try {
                camera?.setPreviewSurface(surface)
            } catch (error: Exception) {
                StreamErrorLogger.info(
                    "Preview Surface update rejected; camera pipeline continues: " +
                        (error.message ?: error.javaClass.simpleName),
                )
                reportPreviewDiagnostic(StreamErrorFormatter.fromThrowable(StreamErrorKind.SURFACE, error))
            }
        }
    }

    private fun reportError(failure: StreamFailure) {
        if (failure.kind == StreamErrorKind.SURFACE && output != null) {
            reportStreamError(failure)
        } else {
            dispatchOrFail(action = { failPreview(failure) }, context = "preview-resource-error")
        }
    }

    private fun reportStreamError(failure: StreamFailure) {
        dispatchOrFail(action = { failStream(failure) }, context = "stream-resource-error")
    }

    private fun reportPreviewDiagnostic(failure: StreamFailure) {
        StreamErrorLogger.info("preview_failed_nonfatal kind=${failure.kind.name.lowercase()}")
        if (!dispatch {
                callbacks.onPreviewDiagnostic(failure)
            }
        ) {
            StreamErrorLogger.error(
                StreamFailure(StreamErrorKind.THREAD, "preview diagnostic callback dispatch rejected"),
            )
        }
    }

    private fun reportPreviewRecovered() {
        if (!dispatch {
                callbacks.onPreviewRecovered()
            }
        ) {
            StreamErrorLogger.error(
                StreamFailure(StreamErrorKind.THREAD, "preview recovery callback dispatch rejected"),
            )
        }
    }

    private fun dispatchOrFail(action: () -> Unit, context: String) {
        if (!dispatch(action)) {
            failPreview(
                StreamFailure(
                    kind = StreamErrorKind.THREAD,
                    detail = "$context callback dispatch rejected",
                ),
            )
        }
    }

    private fun failPreview(failure: StreamFailure) {
        val finalFailure: StreamFailure?
        synchronized(lock) {
            if (!started || terminated) return
            terminated = true
            finalFailure = StreamErrorFormatter.withCleanup(failure, closeResourcesLocked())
        }
        finalFailure?.let { error ->
            StreamErrorLogger.error(error)
            try {
                callbacks.onError(PipelineStage.PREVIEW, error)
            } catch (callbackError: Exception) {
                StreamErrorLogger.observer(callbackError)
            }
        }
    }

    private fun failStream(failure: StreamFailure) {
        val finalFailure: StreamFailure
        synchronized(lock) {
            if (!started || terminated) return
            val cleanup = CleanupCollector()
            val activeOutput = output.also { output = null }
            streamRequestedBindAddress = null
            detachEncoderAndStopOutput(activeOutput, cleanup)
            notifySubsystemState(StreamSubsystem.ENCODER, SubsystemState.IDLE)
            notifySubsystemState(StreamSubsystem.RTSP_SERVER, SubsystemState.IDLE)
            finalFailure = StreamErrorFormatter.withCleanup(failure, cleanup.report())
        }
        StreamErrorLogger.error(finalFailure)
        try {
            callbacks.onError(PipelineStage.STREAM, finalFailure)
        } catch (callbackError: Exception) {
            StreamErrorLogger.observer(callbackError)
        }
    }

    override fun stopStreaming(): CleanupReport {
        synchronized(lock) {
            if (!started || terminated) return CleanupReport()
            streamRequestedBindAddress = null
            val activeOutput = output.also { output = null }
            val cleanup = CleanupCollector()
            detachEncoderAndStopOutput(activeOutput, cleanup)
            notifySubsystemState(StreamSubsystem.ENCODER, SubsystemState.IDLE)
            notifySubsystemState(StreamSubsystem.RTSP_SERVER, SubsystemState.IDLE)
            if (activeOutput != null) {
                StreamErrorLogger.info("STREAM_ENCODER_STOP generation=${activeOutput.generation}")
            }
            return cleanup.report()
        }
    }

    override fun stop(): CleanupReport {
        synchronized(lock) {
            if (terminated) return CleanupReport()
            terminated = true
            return closeResourcesLocked()
        }
    }

    private fun closeResourcesLocked(): CleanupReport {
        val activeOutput = output.also { output = null }
        val activeCamera = camera.also { camera = null }
        pendingOrientation = null
        orientationTaskScheduled = false
        currentTransform = null
        previewReady = false
        streamRequestedBindAddress = null
        val cleanup = CleanupCollector()
        detachEncoderAndStopOutput(activeOutput, cleanup)
        cleanup.run("camera") { activeCamera?.stop() ?: CleanupReport() }
        metrics.stop()
        orientationExecutor.shutdownNow()
        return cleanup.report()
    }

    private fun stopOutput(candidate: OutputResources?): CleanupReport {
        val cleanup = CleanupCollector()
        detachEncoderAndStopOutput(candidate, cleanup)
        return cleanup.report()
    }

    private fun detachEncoderAndStopOutput(candidate: OutputResources?, cleanup: CleanupCollector) {
        cleanup.run("encoder input") {
            if (candidate != null) camera?.setEncoderInput(null)
            CleanupReport()
        }
        cleanup.run("RTSP server") { candidate?.server?.stop() ?: CleanupReport() }
        cleanup.run("encoder") {
            candidate?.encoder?.stop() ?: CleanupReport()
        }
    }

    private fun isCurrentOutputGeneration(generation: Long): Boolean = synchronized(lock) {
        isActive() && output?.generation == generation
    }

    private fun isActive(): Boolean = started && !terminated

    private fun notifyMetricsChanged() {
        try {
            onMetricsChanged()
        } catch (error: Exception) {
            StreamErrorLogger.observer(error)
        }
    }

    private fun notifySubsystemState(subsystem: StreamSubsystem, next: SubsystemState) {
        try {
            callbacks.onSubsystemStateChanged(subsystem, next)
        } catch (error: Exception) {
            StreamErrorLogger.observer(error)
        }
    }

    private fun notifyOrientationChanged(state: CameraOrientationState) {
        try {
            callbacks.onOrientationChanged(state)
        } catch (error: Exception) {
            StreamErrorLogger.observer(error)
        }
    }

    private data class OutputResources(
        val generation: Long,
        val settings: StreamSettings,
        val transform: VideoTransform,
        val bindAddress: InetAddress,
        val broadcaster: StreamBroadcaster,
        val encoder: EncoderPort,
        var encoderInput: EncoderInput? = null,
        var server: RtspPort? = null,
        var formatValidated: Boolean = false,
        var readyNotified: Boolean = false,
    )

    private fun VideoTransform.matches(orientation: DeviceOrientation): Boolean =
        this.orientation.physicalOrientationDegrees == orientation.physicalOrientationDegrees &&
            this.orientation.displayRotationDegrees == normalizeRotation(orientation.displayRotationDegrees)

    private companion object {
        fun normalizeRotation(value: Int): Int = ((value % 360) + 360) % 360
    }
}
