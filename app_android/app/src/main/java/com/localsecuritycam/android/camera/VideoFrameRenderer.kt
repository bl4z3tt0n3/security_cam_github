package com.localsecuritycam.android.camera

import android.graphics.SurfaceTexture
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.EGLExt
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.Matrix
import android.os.Handler
import android.os.HandlerThread
import android.view.Surface
import com.localsecuritycam.android.diagnostics.CleanupCollector
import com.localsecuritycam.android.diagnostics.CleanupReport
import com.localsecuritycam.android.diagnostics.StreamErrorFormatter
import com.localsecuritycam.android.diagnostics.StreamErrorKind
import com.localsecuritycam.android.diagnostics.StreamErrorLogger
import com.localsecuritycam.android.diagnostics.StreamFailure
import com.localsecuritycam.android.diagnostics.StreamFailureException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * GPU bridge from the camera SurfaceTexture to the MediaCodec input surface and
 * optional preview surface. It keeps rotation out of the encoded bitstream's
 * consumer and avoids CPU frame copies.
 */
class VideoFrameRenderer(
    private val width: Int,
    private val height: Int,
    initialTransform: VideoTransform,
    private val diagnosticMode: PreviewDiagnosticMode = PreviewDiagnosticMode.NORMAL,
    private val onFrameRendered: (Long) -> Unit,
    private val onError: (StreamFailure) -> Unit = {},
    private val onPreviewError: (StreamFailure) -> Unit = {},
    private val onPreviewRecovered: () -> Unit = {},
) {
    private var thread: HandlerThread? = null
    private var handler: Handler? = null
    private var display: EGLDisplay = EGL14.EGL_NO_DISPLAY
    private var context: EGLContext = EGL14.EGL_NO_CONTEXT
    private var config: EGLConfig? = null
    private var encoderSurface: EGLSurface = EGL14.EGL_NO_SURFACE
    private var previewSurface: EGLSurface = EGL14.EGL_NO_SURFACE
    private var pbufferSurface: EGLSurface = EGL14.EGL_NO_SURFACE
    private var encoderNativeSurface: Surface? = null
    private var requestedEncoderSurface: Surface? = null
    private val previewBinding = PreviewSurfaceBinding<PreviewSurfaceAttachment>()
    private var textureId = 0
    private var cameraTexture: SurfaceTexture? = null
    private var cameraSurface: Surface? = null
    private var program = 0
    private var positionHandle = 0
    private var textureHandle = 0
    private var matrixHandle = 0
    private var rotationHandle = 0
    private var geometryHandle = 0
    private var samplerHandle = -1
    private var vertexBuffer: FloatBuffer? = null
    private var texCoordBuffer: FloatBuffer? = null
    @Volatile
    private var started = false
    @Volatile
    private var stopping = false
    @Volatile
    private var transform = initialTransform
    private var framePending = false
    private var lastEncoderGeometry: LoggedGeometry? = null
    private var lastPreviewGeometry: LoggedGeometry? = null
    private var firstFrameReceivedLogged = false
    private var firstPreviewFrameRenderedLogged = false
    private var previewDiagnosticActive = false
    private var lastPreviewDiagnosticTarget: String? = null
    // Render-thread-only scratch buffers. Reusing them removes several small
    // allocations from every camera frame and therefore reduces GC pressure.
    private val textureMatrixScratch = FloatArray(16)
    private val rotationMatrixScratch = FloatArray(16)
    private val geometryMatrixScratch = FloatArray(16)

    private val diagnosticsEnabled: Boolean
        get() = diagnosticMode != PreviewDiagnosticMode.NORMAL

    fun start(encoderInput: Surface?, preview: PreviewSurfaceAttachment?): Surface {
        check(!started && !stopping) { "renderer already started" }
        stopping = false
        previewBinding.begin()
        previewBinding.request(preview)
        requestedEncoderSurface = encoderInput
        val ready = CountDownLatch(1)
        var failure: Throwable? = null
        val renderThread = HandlerThread("camera-gl-renderer")
        thread = renderThread
        try {
            renderThread.start()
            handler = Handler(renderThread.looper)
        } catch (error: Exception) {
            val cleanup = stop()
            throw StreamFailureException(
                StreamErrorFormatter.withCleanup(
                    StreamErrorFormatter.fromThrowable(StreamErrorKind.THREAD, error),
                    cleanup,
                ),
            )
        }
        if (!handler!!.post {
            try {
                initialize()
            } catch (error: Throwable) {
                failure = error
            } finally {
                ready.countDown()
            }
        }) {
            val cleanup = stop()
            throw StreamFailureException(
                StreamErrorFormatter.withCleanup(
                    StreamFailure(StreamErrorKind.THREAD, "renderer initialization callback rejected"),
                    cleanup,
                ),
            )
        }
        if (!ready.await(5, TimeUnit.SECONDS)) {
            val cleanup = stop()
            throw StreamFailureException(
                StreamErrorFormatter.withCleanup(
                    StreamFailure(StreamErrorKind.SURFACE, "timed out starting video renderer"),
                    cleanup,
                ),
            )
        }
        failure?.let {
            val cleanup = stop()
            throw StreamFailureException(
                StreamErrorFormatter.withCleanup(
                    StreamErrorFormatter.fromThrowable(StreamErrorKind.SURFACE, it),
                    cleanup,
                ),
            )
        }
        started = true
        return cameraSurface ?: error("renderer did not create a camera surface")
    }

    fun setPreviewSurface(surface: PreviewSurfaceAttachment?) {
        val update = previewBinding.request(surface)
        if (update.kind != PreviewSurfaceUpdateKind.APPLY || stopping) return
        val currentHandler = handler
        if (currentHandler == null) {
            if (!stopping) {
                reportPreviewDiagnostic(StreamFailure(StreamErrorKind.THREAD, "renderer handler unavailable"))
            }
            return
        }
        if (!currentHandler.post {
            if (stopping) return@post
            try {
                applyPreviewSurface(update)
            } catch (error: Exception) {
                StreamErrorLogger.info(
                    "Preview EGL attach failed; camera pipeline continues: " +
                        (error.message ?: error.javaClass.simpleName),
                )
                reportPreviewDiagnostic(
                    StreamErrorFormatter.fromThrowable(StreamErrorKind.SURFACE, error),
                )
                recoverPreviewTarget()
            }
        }) {
            if (!stopping) {
                reportPreviewDiagnostic(StreamFailure(StreamErrorKind.THREAD, "preview Surface callback rejected"))
            }
        }
    }

    /** Update geometry without touching Camera2, MediaCodec, or RTSP state. */
    fun setTransform(next: VideoTransform) {
        transform = next
        StreamErrorLogger.info(
            "Renderer transform updated camera2_rotation=${next.rotationDegrees} " +
                "pixel_clockwise_rotation=${next.pixelClockwiseRotationDegrees} " +
                "camera_gl_texture_rotation=${next.glTextureRotationDegrees} " +
                "mirror_preview=${next.mirrorPreview} mirror_stream=${next.mirrorStream} " +
                "logical_video_size=${next.logicalWidth}x${next.logicalHeight}",
        )
    }

    /**
     * Replaces only the EGL target for MediaCodec. The Camera2 SurfaceTexture,
     * capture session and preview EGL surface remain alive during an aspect
     * change, avoiding a camera restart while the encoder is recreated.
     */
    fun setEncoderSurface(surface: Surface?) {
        if (stopping) return
        val currentHandler = handler
            ?: throw StreamFailureException(StreamFailure(StreamErrorKind.THREAD, "renderer handler unavailable"))
        val completed = CountDownLatch(1)
        var failure: Throwable? = null
        if (!currentHandler.post {
            try {
                applyEncoderSurface(surface)
            } catch (error: Throwable) {
                failure = error
            } finally {
                completed.countDown()
            }
        }) {
            throw StreamFailureException(StreamFailure(StreamErrorKind.THREAD, "encoder Surface callback rejected"))
        }
        if (!completed.await(2, TimeUnit.SECONDS)) {
            throw StreamFailureException(StreamFailure(StreamErrorKind.SURFACE, "timed out replacing encoder Surface"))
        }
        failure?.let { error ->
            throw StreamFailureException(StreamErrorFormatter.fromThrowable(StreamErrorKind.SURFACE, error))
        }
    }

    fun stop(): CleanupReport {
        stopping = true
        previewBinding.stop()
        val cleanup = CleanupCollector()
        val currentHandler = handler
        val currentThread = thread
        val stopped = CountDownLatch(1)
        val releaseStarted = AtomicBoolean(false)
        fun releaseOnce() {
            if (releaseStarted.compareAndSet(false, true)) cleanup.add(releaseGl())
        }
        if (currentHandler != null && Thread.currentThread() === currentThread) {
            releaseOnce()
            stopped.countDown()
        } else if (currentHandler != null) {
            if (!currentHandler.post {
                releaseOnce()
                stopped.countDown()
            }) {
                cleanup.add(StreamErrorFormatter.cleanupFailure("renderer thread", IllegalStateException("release callback rejected")))
                releaseOnce()
            } else if (!stopped.await(2, TimeUnit.SECONDS)) {
                cleanup.add(StreamErrorFormatter.cleanupFailure("renderer thread", IllegalStateException("release timed out")))
                currentHandler.removeCallbacksAndMessages(null)
                releaseOnce()
            }
        } else {
            releaseOnce()
        }
        cleanup.runUnit("renderer thread") { thread?.quitSafely() }
        if (currentThread != null && Thread.currentThread() !== currentThread) {
            cleanup.runUnit("renderer thread join") { currentThread.join(1_000) }
            if (currentThread.isAlive) {
                cleanup.add(
                    StreamErrorFormatter.cleanupFailure(
                        "renderer thread join",
                        IllegalStateException("renderer thread did not stop"),
                    ),
                )
            }
        }
        handler = null
        thread = null
        started = false
        return cleanup.report()
    }

    private fun reportError(failure: StreamFailure) {
        try {
            onError(failure)
        } catch (error: Exception) {
            StreamErrorLogger.observer(error)
        }
    }

    private fun initialize() {
        val renderHandler = handler ?: error("renderer handler unavailable")
        display = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        check(display != EGL14.EGL_NO_DISPLAY) { "EGL display unavailable" }
        val version = IntArray(2)
        check(EGL14.eglInitialize(display, version, 0, version, 1)) { "EGL initialization failed" }
        val attributes = intArrayOf(
            EGL14.EGL_RED_SIZE, 8,
            EGL14.EGL_GREEN_SIZE, 8,
            EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_ALPHA_SIZE, 8,
            EGL14.EGL_SURFACE_TYPE, EGL14.EGL_WINDOW_BIT or EGL14.EGL_PBUFFER_BIT,
            EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
            EGL_RECORDABLE_ANDROID, 1,
            EGL14.EGL_NONE,
        )
        val configs = arrayOfNulls<EGLConfig>(1)
        val count = IntArray(1)
        check(EGL14.eglChooseConfig(display, attributes, 0, configs, 0, 1, count, 0) && count[0] > 0) {
            "EGL configuration unavailable"
        }
        config = configs[0]
        context = EGL14.eglCreateContext(
            display,
            config,
            EGL14.EGL_NO_CONTEXT,
            intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE),
            0,
        )
        check(context != EGL14.EGL_NO_CONTEXT) { "EGL context unavailable" }
        logEglError("eglCreateContext")
        pbufferSurface = EGL14.eglCreatePbufferSurface(
            display,
            config,
            intArrayOf(EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE),
            0,
        )
        check(pbufferSurface != EGL14.EGL_NO_SURFACE) { "EGL pbuffer unavailable" }
        logEglError("eglCreatePbufferSurface")
        check(EGL14.eglMakeCurrent(display, pbufferSurface, pbufferSurface, context)) {
            "EGL context could not be made current"
        }
        logEglError("eglMakeCurrent(pbuffer)")
        StreamErrorLogger.info("Renderer EGL initialized pbuffer=1x1")
        if (diagnosticsEnabled) {
            StreamErrorLogger.info(
                "GLES_CAPS renderer=${GLES20.glGetString(GLES20.GL_RENDERER)} " +
                    "version=${GLES20.glGetString(GLES20.GL_VERSION)} " +
                    "glsl=${GLES20.glGetString(GLES20.GL_SHADING_LANGUAGE_VERSION)} " +
                    "external_oes=${GLES20.glGetString(GLES20.GL_EXTENSIONS)?.contains("GL_OES_EGL_image_external")}",
            )
        }
        program = createProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        positionHandle = requireLocation("aPosition", GLES20.glGetAttribLocation(program, "aPosition"))
        textureHandle = requireLocation("aTexCoord", GLES20.glGetAttribLocation(program, "aTexCoord"))
        matrixHandle = requireLocation("uTextureMatrix", GLES20.glGetUniformLocation(program, "uTextureMatrix"))
        rotationHandle = requireLocation("uRotationMatrix", GLES20.glGetUniformLocation(program, "uRotationMatrix"))
        geometryHandle = requireLocation("uGeometryMatrix", GLES20.glGetUniformLocation(program, "uGeometryMatrix"))
        samplerHandle = requireLocation("sTexture", GLES20.glGetUniformLocation(program, "sTexture"))
        checkGlErrors("program handles")
        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        checkGlErrors("glGenTextures")
        textureId = textures[0]
        check(textureId != 0) { "camera OES texture unavailable" }
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        checkGlErrors("camera OES texture setup")
        vertexBuffer = floatBuffer(floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f))
        // SurfaceTexture.getTransformMatrix() supplies the producer crop and
        // vertical-axis conversion. Keep OES coordinates canonical so that
        // this conversion is applied exactly once in the vertex shader.
        texCoordBuffer = floatBuffer(floatArrayOf(0f, 0f, 1f, 0f, 0f, 1f, 1f, 1f))
        cameraTexture = SurfaceTexture(textureId).also { texture ->
            StreamErrorLogger.info("SurfaceTexture camera_buffer=${width}x${height}")
            texture.setDefaultBufferSize(width, height)
            texture.setOnFrameAvailableListener({
                if (!firstFrameReceivedLogged) {
                    firstFrameReceivedLogged = true
                    StreamErrorLogger.info("PREVIEW_FIRST_FRAME_RECEIVED")
                }
                if (!stopping && !framePending) {
                    framePending = true
                    val renderHandler = handler
                    if (renderHandler == null || !renderHandler.post {
                            try {
                                renderFrame()
                            } catch (error: Exception) {
                                reportError(StreamErrorFormatter.fromThrowable(StreamErrorKind.SURFACE, error))
                            }
                        }
                    ) {
                        if (!stopping) reportError(StreamFailure(StreamErrorKind.THREAD, "render frame callback rejected"))
                    }
                }
            }, renderHandler)
        }
        cameraSurface = Surface(cameraTexture)
        requestedEncoderSurface?.let(::applyEncoderSurface)
        requestedEncoderSurface = null
        try {
            applyPreviewSurface(previewBinding.markReady())
        } catch (error: Exception) {
            StreamErrorLogger.info(
                "Initial preview EGL target unavailable; continuing with fallback: " +
                    (error.message ?: error.javaClass.simpleName),
            )
            reportPreviewDiagnostic(
                StreamErrorFormatter.fromThrowable(StreamErrorKind.SURFACE, error),
            )
            recoverPreviewTarget()
        }
    }

    private fun renderFrame() {
        if (stopping || !framePending || cameraTexture == null) return
        framePending = false
        val texture = cameraTexture ?: return
        try {
            makeCurrentForTextureUpdate()
            texture.updateTexImage()
        } catch (error: Exception) {
            if (!stopping) {
                reportError(StreamErrorFormatter.fromThrowable(StreamErrorKind.SURFACE, error))
            }
            return
        }
        val textureMatrix = textureMatrixScratch
        texture.getTransformMatrix(textureMatrix)
        val surfaceTexturePixelClockwiseRotation = surfaceTexturePixelClockwiseRotationDegrees(textureMatrix)
        val currentTransform = transform
        if (encoderSurface != EGL14.EGL_NO_SURFACE) {
            val encoderRendered = drawTarget(
                target = encoderSurface,
                textureMatrix = textureMatrix,
                surfaceTexturePixelClockwiseRotation = surfaceTexturePixelClockwiseRotation,
                sourceTransform = currentTransform,
                isEncoder = true,
                timestampNs = texture.timestamp,
            )
            if (!encoderRendered && !stopping) {
                destroyEncoderSurface()
                reportError(
                    StreamFailure(
                        StreamErrorKind.SURFACE,
                        "encoder EGL target lost during eglSwapBuffers",
                    ),
                )
            }
        }
        if (previewSurface != EGL14.EGL_NO_SURFACE) {
            val previewRendered = drawTarget(
                target = previewSurface,
                textureMatrix = textureMatrix,
                surfaceTexturePixelClockwiseRotation = surfaceTexturePixelClockwiseRotation,
                sourceTransform = currentTransform,
                isEncoder = false,
                timestampNs = texture.timestamp,
            )
            if (previewRendered && !firstPreviewFrameRenderedLogged) {
                firstPreviewFrameRenderedLogged = true
                StreamErrorLogger.info("preview_first_frame_presented")
                StreamErrorLogger.info("PREVIEW_FIRST_FRAME_RENDERED")
            }
        }
        onFrameRendered(texture.timestamp)
    }

    private fun textureRotationMatrix(
        transform: VideoTransform,
        mirror: Boolean,
        surfaceTexturePixelClockwiseRotation: Int,
    ): FloatArray {
        val matrix = rotationMatrixScratch
        Matrix.setIdentityM(matrix, 0)
        val glTextureRotation = textureCoordinateRotationDegrees(
            cameraRelativeRotationDegrees = transform.rotationDegrees,
            lensFacing = transform.orientation.lensFacing,
            surfaceTexturePixelClockwiseRotationDegrees = surfaceTexturePixelClockwiseRotation,
            outputPixelClockwiseRotationDegrees = transform.outputPixelClockwiseRotationDegrees,
        )
        if (glTextureRotation != 0 || mirror) {
            Matrix.translateM(matrix, 0, 0.5f, 0.5f, 0f)
            Matrix.rotateM(
                matrix,
                0,
                glTextureRotation.toFloat(),
                0f,
                0f,
                1f,
            )
            if (mirror) Matrix.scaleM(matrix, 0, -1f, 1f, 1f)
            Matrix.translateM(matrix, 0, -0.5f, -0.5f, 0f)
        }
        return matrix
    }

    private fun makeCurrentForTextureUpdate() {
        val target = when {
            encoderSurface != EGL14.EGL_NO_SURFACE -> encoderSurface
            previewSurface != EGL14.EGL_NO_SURFACE -> previewSurface
            else -> pbufferSurface
        }
        check(target != EGL14.EGL_NO_SURFACE) { "no EGL surface available for camera texture" }
        check(EGL14.eglMakeCurrent(display, target, target, context)) {
            "EGL context could not be made current for camera texture"
        }
        logEglError("eglMakeCurrent(texture_update)")
    }

    private fun drawTarget(
        target: EGLSurface,
        textureMatrix: FloatArray,
        surfaceTexturePixelClockwiseRotation: Int,
        sourceTransform: VideoTransform,
        isEncoder: Boolean,
        timestampNs: Long,
    ): Boolean {
        return try {
            val renderMode = if (isEncoder) PreviewDiagnosticMode.NORMAL else diagnosticMode
            val useGeometry = isEncoder || renderMode == PreviewDiagnosticMode.NORMAL ||
                renderMode == PreviewDiagnosticMode.FULL
            val glTextureRotation = if (renderMode == PreviewDiagnosticMode.OES_IDENTITY) {
                0
            } else {
                textureCoordinateRotationDegrees(
                    cameraRelativeRotationDegrees = sourceTransform.rotationDegrees,
                    lensFacing = sourceTransform.orientation.lensFacing,
                    surfaceTexturePixelClockwiseRotationDegrees = surfaceTexturePixelClockwiseRotation,
                    outputPixelClockwiseRotationDegrees = sourceTransform.outputPixelClockwiseRotationDegrees,
                )
            }
            val rotationMatrix = when (renderMode) {
                PreviewDiagnosticMode.OES_IDENTITY -> identityMatrix()
                PreviewDiagnosticMode.OES_ROTATION -> textureRotationMatrix(
                    sourceTransform,
                    mirror = false,
                    surfaceTexturePixelClockwiseRotation = surfaceTexturePixelClockwiseRotation,
                )
                else -> textureRotationMatrix(
                    sourceTransform,
                    if (isEncoder) sourceTransform.mirrorStream else sourceTransform.mirrorPreview,
                    surfaceTexturePixelClockwiseRotation,
                )
            }
            drawTo(
                target = target,
                textureMatrix = textureMatrix,
                surfaceTexturePixelClockwiseRotation = surfaceTexturePixelClockwiseRotation,
                glTextureRotation = glTextureRotation,
                rotationMatrix = rotationMatrix,
                sourceTransform = sourceTransform,
                isEncoder = isEncoder,
                timestampNs = timestampNs,
                useGeometry = useGeometry,
                renderMode = renderMode,
            )
            true
        } catch (error: Exception) {
            val eglError = EGL14.eglGetError()
            val targetName = if (isEncoder) "encoder" else "preview"
            StreamErrorLogger.info(
                "EGL_SWAP_BUFFERS_FAILED target=$targetName error=0x${eglError.toString(16)} " +
                    "detail=${error.message ?: error.javaClass.simpleName}",
            )
            if (!isEncoder) {
                recoverPreviewTarget()
                reportPreviewDiagnostic(
                    StreamErrorFormatter.fromThrowable(StreamErrorKind.SURFACE, error),
                )
            }
            false
        }
    }

    private fun drawTo(
        target: EGLSurface,
        textureMatrix: FloatArray,
        surfaceTexturePixelClockwiseRotation: Int,
        glTextureRotation: Int,
        rotationMatrix: FloatArray,
        sourceTransform: VideoTransform,
        isEncoder: Boolean,
        timestampNs: Long,
        useGeometry: Boolean,
        renderMode: PreviewDiagnosticMode,
    ) {
        check(EGL14.eglMakeCurrent(display, target, target, context)) { "EGL surface could not be made current" }
        logEglError("eglMakeCurrent(${if (isEncoder) "encoder" else "preview"})")
        val vertices = vertexBuffer ?: return
        val texCoords = texCoordBuffer ?: return
        // Target geometry is already known by the pipeline. Querying EGL for
        // width/height on every draw adds driver calls to the hot path.
        val targetWidth = if (isEncoder) {
            sourceTransform.targetWidth
        } else {
            previewNativeSurface?.width ?: sourceTransform.targetWidth
        }
        val targetHeight = if (isEncoder) {
            sourceTransform.targetHeight
        } else {
            previewNativeSurface?.height ?: sourceTransform.targetHeight
        }
        val targetTransform = sourceTransform.forTarget(
            targetWidth.coerceAtLeast(1),
            targetHeight.coerceAtLeast(1),
        )
        logGeometry(
            isEncoder = isEncoder,
            width = targetWidth.coerceAtLeast(1),
            height = targetHeight.coerceAtLeast(1),
            transform = targetTransform,
        )
        val geometryMatrix = geometryMatrixScratch
        Matrix.setIdentityM(geometryMatrix, 0)
        if (useGeometry) {
            Matrix.scaleM(geometryMatrix, 0, targetTransform.scaleX, targetTransform.scaleY, 1f)
        }
        if (diagnosticsEnabled && !isEncoder) {
            val blendEnabled = GLES20.glIsEnabled(GLES20.GL_BLEND)
            val scissorEnabled = GLES20.glIsEnabled(GLES20.GL_SCISSOR_TEST)
            val targetDescription =
                "${renderMode.wireValue}:${targetWidth.coerceAtLeast(1)}x${targetHeight.coerceAtLeast(1)}:" +
                    "$textureId:$blendEnabled:$scissorEnabled"
            val firstTargetLog = lastPreviewDiagnosticTarget != targetDescription
            if (firstTargetLog) {
                lastPreviewDiagnosticTarget = targetDescription
                StreamErrorLogger.info(
                    "PREVIEW_GLES_TARGET mode=${renderMode.wireValue} " +
                        "surface=${targetWidth.coerceAtLeast(1)}x${targetHeight.coerceAtLeast(1)} " +
                        "viewport=${targetWidth.coerceAtLeast(1)}x${targetHeight.coerceAtLeast(1)} " +
                        "texture_id=$textureId texture_target=GL_TEXTURE_EXTERNAL_OES " +
                        "texture_unit=GL_TEXTURE0 blend=$blendEnabled scissor=$scissorEnabled",
                )
                StreamErrorLogger.info(
                    "PREVIEW_GLES_MATRICES mode=${renderMode.wireValue} " +
                        "destination=${if (isEncoder) "encoder" else "preview"} " +
                        "camera2_rotation=${sourceTransform.rotationDegrees} " +
                        "pixel_clockwise_rotation=${sourceTransform.pixelClockwiseRotationDegrees} " +
                        "output_pixel_clockwise_rotation=${sourceTransform.outputPixelClockwiseRotationDegrees} " +
                        "surface_texture_pixel_clockwise_rotation=$surfaceTexturePixelClockwiseRotation " +
                        "gl_texture_rotation=$glTextureRotation " +
                        "texture=${formatMatrix(textureMatrix)} " +
                        "rotation=${formatMatrix(rotationMatrix)} " +
                        "geometry=${formatMatrix(geometryMatrix)}",
                )
            }
            if (blendEnabled) GLES20.glDisable(GLES20.GL_BLEND)
            if (scissorEnabled) GLES20.glDisable(GLES20.GL_SCISSOR_TEST)
            checkGlErrors("preview state")
        }
        GLES20.glViewport(0, 0, targetWidth.coerceAtLeast(1), targetHeight.coerceAtLeast(1))
        checkGlErrors("glViewport")
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
        checkGlErrors("glClear")
        GLES20.glUseProgram(program)
        checkGlErrors("glUseProgram")
        GLES20.glEnableVertexAttribArray(positionHandle)
        GLES20.glVertexAttribPointer(positionHandle, 2, GLES20.GL_FLOAT, false, 0, vertices)
        checkGlErrors("position attribute")
        GLES20.glEnableVertexAttribArray(textureHandle)
        GLES20.glVertexAttribPointer(textureHandle, 2, GLES20.GL_FLOAT, false, 0, texCoords)
        checkGlErrors("texture attribute")
        GLES20.glUniformMatrix4fv(matrixHandle, 1, false, textureMatrix, 0)
        GLES20.glUniformMatrix4fv(rotationHandle, 1, false, rotationMatrix, 0)
        GLES20.glUniformMatrix4fv(geometryHandle, 1, false, geometryMatrix, 0)
        GLES20.glUniform1i(samplerHandle, 0)
        checkGlErrors("glUniform")
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
        checkGlErrors("OES texture bind")
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        checkGlErrors("glDrawArrays")
        GLES20.glDisableVertexAttribArray(positionHandle)
        GLES20.glDisableVertexAttribArray(textureHandle)
        if (isEncoder && timestampNs > 0) EGLExt.eglPresentationTimeANDROID(display, target, timestampNs)
        check(EGL14.eglSwapBuffers(display, target)) {
            val error = EGL14.eglGetError()
            "eglSwapBuffers failed error=0x${error.toString(16)}"
        }
        logEglError("eglSwapBuffers(${if (isEncoder) "encoder" else "preview"})")
        checkGlErrors("after eglSwapBuffers")
    }

    private fun identityMatrix(): FloatArray {
        Matrix.setIdentityM(rotationMatrixScratch, 0)
        return rotationMatrixScratch
    }

    private fun requireLocation(name: String, location: Int): Int {
        if (diagnosticsEnabled && location < 0) {
            error("GLES location unavailable name=$name")
        }
        return location
    }

    private fun checkGlErrors(stage: String) {
        if (!diagnosticsEnabled) return
        var error = GLES20.glGetError()
        while (error != GLES20.GL_NO_ERROR) {
            StreamErrorLogger.info("GLES_ERROR stage=$stage code=0x${error.toString(16)}")
            error = GLES20.glGetError()
        }
    }

    private fun logEglError(stage: String) {
        if (!diagnosticsEnabled) return
        val error = EGL14.eglGetError()
        if (error != EGL14.EGL_SUCCESS) {
            StreamErrorLogger.info("EGL_ERROR stage=$stage code=0x${error.toString(16)}")
        }
    }

    private fun formatMatrix(values: FloatArray): String = values.joinToString(
        prefix = "[",
        postfix = "]",
        separator = ",",
    ) { String.format(Locale.US, "%.4f", it) }

    private fun createWindowSurface(surface: Surface): EGLSurface = EGL14.eglCreateWindowSurface(
        display,
        config ?: error("EGL config missing"),
        surface,
        intArrayOf(EGL14.EGL_NONE),
        0,
    ).also { created ->
        check(created != EGL14.EGL_NO_SURFACE) {
            val error = EGL14.eglGetError()
            "EGL window surface unavailable error=0x${error.toString(16)}"
        }
        logEglError("eglCreateWindowSurface")
    }

    private fun applyEncoderSurface(surface: Surface?) {
        if (surface === encoderNativeSurface &&
            (surface == null || encoderSurface != EGL14.EGL_NO_SURFACE)
        ) return
        destroyEncoderSurface()
        if (surface != null) {
            check(surface.isValid) { "encoder Surface is invalid" }
            val created = createWindowSurface(surface)
            encoderSurface = created
            encoderNativeSurface = surface
            check(EGL14.eglMakeCurrent(display, created, created, context)) {
                "EGL context could not be made current for encoder Surface"
            }
            logEglError("eglMakeCurrent(encoder_attach)")
            StreamErrorLogger.info("Encoder EGL surface attached")
        } else {
            makeCurrentForTextureUpdate()
            StreamErrorLogger.info("Encoder EGL surface detached")
        }
    }

    private fun applyPreviewSurface(update: PreviewSurfaceUpdate<PreviewSurfaceAttachment>) {
        if (update.kind != PreviewSurfaceUpdateKind.APPLY || stopping) return
        if (!previewBinding.isCurrent(update)) {
            StreamErrorLogger.info("Stale preview EGL callback ignored")
            return
        }
        val attachment = update.surface
        val surface = attachment?.surface
        if (surface != null && !surface.isValid) {
            StreamErrorLogger.info("Preview Surface rejected because it is invalid")
            reportPreviewDiagnostic(
                StreamFailure(StreamErrorKind.SURFACE, "preview Surface is invalid"),
            )
            recoverPreviewTarget()
            return
        }
        if (
            attachment != null &&
            previewNativeSurface?.surface === attachment.surface &&
            previewNativeSurface?.width == attachment.width &&
            previewNativeSurface?.height == attachment.height
        ) return
        destroyPreviewSurface()
        if (attachment != null) {
            val created = createWindowSurface(attachment.surface)
            previewSurface = created
            previewNativeSurface = attachment
            StreamErrorLogger.info(
                "preview_surface_attached generation=${update.generation} " +
                    "size=${attachment.width}x${attachment.height}",
            )
            if (previewDiagnosticActive) {
                previewDiagnosticActive = false
                try {
                    onPreviewRecovered()
                } catch (error: Exception) {
                    StreamErrorLogger.observer(error)
                }
            }
        }
    }

    private var previewNativeSurface: PreviewSurfaceAttachment? = null

    private fun recoverPreviewTarget() {
        destroyPreviewSurface()
        StreamErrorLogger.info("Preview EGL target dropped; awaiting a new valid Surface")
        runCatching { makeCurrentForTextureUpdate() }
            .onFailure { error ->
                if (!stopping) {
                    StreamErrorLogger.info(
                        "Preview EGL fallback unavailable: ${error.message ?: error.javaClass.simpleName}",
                    )
                }
            }
    }

    private fun reportPreviewDiagnostic(failure: StreamFailure) {
        if (stopping || previewDiagnosticActive) return
        previewDiagnosticActive = true
        try {
            onPreviewError(failure)
        } catch (error: Exception) {
            StreamErrorLogger.observer(error)
        }
    }

    private fun destroyPreviewSurface() {
        val current = previewSurface
        previewSurface = EGL14.EGL_NO_SURFACE
        previewNativeSurface = null
        if (current != EGL14.EGL_NO_SURFACE) {
            firstPreviewFrameRenderedLogged = false
            lastPreviewDiagnosticTarget = null
            StreamErrorLogger.info("preview_surface_detached")
            StreamErrorLogger.info("Preview EGL surface detached")
        }
        if (current != EGL14.EGL_NO_SURFACE) EGL14.eglDestroySurface(display, current)
    }

    private fun destroyEncoderSurface() {
        val current = encoderSurface
        encoderSurface = EGL14.EGL_NO_SURFACE
        encoderNativeSurface = null
        if (current != EGL14.EGL_NO_SURFACE) {
            val fallback = if (previewSurface != EGL14.EGL_NO_SURFACE) previewSurface else pbufferSurface
            if (fallback != EGL14.EGL_NO_SURFACE) {
                EGL14.eglMakeCurrent(display, fallback, fallback, context)
            }
        }
        if (current != EGL14.EGL_NO_SURFACE) EGL14.eglDestroySurface(display, current)
    }

    private fun logGeometry(isEncoder: Boolean, width: Int, height: Int, transform: VideoTransform) {
        val geometry = LoggedGeometry(
            width = width,
            height = height,
            rotation = transform.outputRotationDegrees,
            mirror = if (isEncoder) transform.mirrorStream else transform.mirrorPreview,
            logicalWidth = transform.logicalWidth,
            logicalHeight = transform.logicalHeight,
            scaleX = transform.scaleX,
            scaleY = transform.scaleY,
            uniformScale = transform.uniformScale,
        )
        if (isEncoder) {
            if (lastEncoderGeometry == geometry) return
            lastEncoderGeometry = geometry
        } else {
            if (lastPreviewGeometry == geometry) return
            lastPreviewGeometry = geometry
        }
        val label = if (isEncoder) "encoder" else "preview"
        StreamErrorLogger.info(
            "$label viewport=${width}x${height} " +
                "camera2_rotation=${transform.rotationDegrees} " +
                "pixel_clockwise_rotation=${transform.pixelClockwiseRotationDegrees} " +
                "output_pixel_clockwise_rotation=${transform.outputPixelClockwiseRotationDegrees} " +
                "camera_gl_texture_rotation=${transform.glTextureRotationDegrees} " +
                "logical_video_size=" +
                "${transform.logicalWidth}x${transform.logicalHeight} " +
                "source_aspect=${transform.sourceAspectRatio} " +
                "destination_aspect=${transform.destinationAspectRatio} " +
                "${label}_scale=${transform.scaleX}x${transform.scaleY} " +
                "uniform_scale=${transform.uniformScale}",
        )
    }

    private fun releaseGl(): CleanupReport {
        val cleanup = CleanupCollector()
        val currentCameraSurface = cameraSurface
        cameraSurface = null
        cleanup.runUnit("camera surface") { currentCameraSurface?.release() }
        val currentCameraTexture = cameraTexture
        cameraTexture = null
        cleanup.runUnit("camera texture") { currentCameraTexture?.release() }
        if (display != EGL14.EGL_NO_DISPLAY) {
            val currentSurface = when {
                encoderSurface != EGL14.EGL_NO_SURFACE -> encoderSurface
                previewSurface != EGL14.EGL_NO_SURFACE -> previewSurface
                else -> pbufferSurface
            }
            if (context != EGL14.EGL_NO_CONTEXT && currentSurface != EGL14.EGL_NO_SURFACE) {
                cleanup.runUnit("EGL make current") {
                    check(EGL14.eglMakeCurrent(display, currentSurface, currentSurface, context)) {
                        "EGL make current failed during cleanup"
                    }
                }
                val currentProgram = program
                program = 0
                cleanup.runUnit("GL program") { if (currentProgram != 0) GLES20.glDeleteProgram(currentProgram) }
                val currentTextureId = textureId
                textureId = 0
                cleanup.runUnit("GL texture") {
                    if (currentTextureId != 0) GLES20.glDeleteTextures(1, intArrayOf(currentTextureId), 0)
                }
            }
            vertexBuffer = null
            texCoordBuffer = null
            cleanup.runUnit("EGL detach") {
                check(EGL14.eglMakeCurrent(display, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)) {
                    "EGL detach failed during cleanup"
                }
            }
            cleanup.runUnit("preview surface") { destroyPreviewSurface() }
            val currentEncoderSurface = encoderSurface
            encoderSurface = EGL14.EGL_NO_SURFACE
            encoderNativeSurface = null
            cleanup.runUnit("encoder EGL surface") {
                if (currentEncoderSurface != EGL14.EGL_NO_SURFACE) EGL14.eglDestroySurface(display, currentEncoderSurface)
            }
            val currentPbufferSurface = pbufferSurface
            pbufferSurface = EGL14.EGL_NO_SURFACE
            cleanup.runUnit("EGL pbuffer surface") {
                if (currentPbufferSurface != EGL14.EGL_NO_SURFACE) EGL14.eglDestroySurface(display, currentPbufferSurface)
            }
            val currentContext = context
            context = EGL14.EGL_NO_CONTEXT
            cleanup.runUnit("EGL context") {
                if (currentContext != EGL14.EGL_NO_CONTEXT) EGL14.eglDestroyContext(display, currentContext)
            }
            cleanup.runUnit("EGL thread") { EGL14.eglReleaseThread() }
            val currentDisplay = display
            display = EGL14.EGL_NO_DISPLAY
            cleanup.runUnit("EGL display") { EGL14.eglTerminate(currentDisplay) }
        }
        context = EGL14.EGL_NO_CONTEXT
        return cleanup.report()
    }

    private fun floatBuffer(values: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(values.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .apply {
                put(values)
                position(0)
            }

    private fun createProgram(vertexSource: String, fragmentSource: String): Int {
        val vertex = compileShader(GLES20.GL_VERTEX_SHADER, vertexSource)
        var fragment = 0
        try {
            fragment = compileShader(GLES20.GL_FRAGMENT_SHADER, fragmentSource)
            val result = GLES20.glCreateProgram()
            checkGlErrors("glCreateProgram")
            try {
                GLES20.glAttachShader(result, vertex)
                GLES20.glAttachShader(result, fragment)
                checkGlErrors("glAttachShader")
                GLES20.glLinkProgram(result)
                checkGlErrors("glLinkProgram")
                val status = IntArray(1)
                GLES20.glGetProgramiv(result, GLES20.GL_LINK_STATUS, status, 0)
                if (status[0] == 0) error(GLES20.glGetProgramInfoLog(result))
                return result
            } catch (error: Throwable) {
                if (result != 0) GLES20.glDeleteProgram(result)
                throw error
            } finally {
                GLES20.glDeleteShader(vertex)
                if (fragment != 0) GLES20.glDeleteShader(fragment)
            }
        } catch (error: Throwable) {
            // The vertex shader is deleted in the program finally block when
            // fragment compilation succeeds; this covers fragment compile failure.
            if (fragment == 0) GLES20.glDeleteShader(vertex)
            throw error
        }
    }

    private fun compileShader(type: Int, source: String): Int {
        val shader = GLES20.glCreateShader(type)
        checkGlErrors("glCreateShader")
        GLES20.glShaderSource(shader, source)
        checkGlErrors("glShaderSource")
        GLES20.glCompileShader(shader)
        checkGlErrors("glCompileShader")
        val status = IntArray(1)
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, status, 0)
        if (status[0] == 0) {
            val log = GLES20.glGetShaderInfoLog(shader)
            GLES20.glDeleteShader(shader)
            error(log)
        }
        return shader
    }

    private companion object {
        // EGL_ANDROID_recordable; required for rendering into a MediaCodec input Surface.
        const val EGL_RECORDABLE_ANDROID = 0x3142
        const val VERTEX_SHADER = """
            attribute vec4 aPosition;
            attribute vec4 aTexCoord;
            uniform mat4 uTextureMatrix;
            uniform mat4 uRotationMatrix;
            uniform mat4 uGeometryMatrix;
            varying vec2 vTexCoord;
            void main() {
                gl_Position = uGeometryMatrix * aPosition;
                // Rotate canonical OES coordinates in the un-cropped camera
                // buffer, then apply SurfaceTexture's producer crop and
                // V-axis normalization exactly once.
                vTexCoord = (uTextureMatrix * uRotationMatrix * aTexCoord).xy;
            }
        """
        const val FRAGMENT_SHADER = """
            #extension GL_OES_EGL_image_external : require
            precision mediump float;
            uniform samplerExternalOES sTexture;
            varying vec2 vTexCoord;
            void main() { gl_FragColor = texture2D(sTexture, vTexCoord); }
        """
    }

    private data class LoggedGeometry(
        val width: Int,
        val height: Int,
        val rotation: Int,
        val mirror: Boolean,
        val logicalWidth: Int,
        val logicalHeight: Int,
        val scaleX: Float,
        val scaleY: Float,
        val uniformScale: Float,
    )
}
