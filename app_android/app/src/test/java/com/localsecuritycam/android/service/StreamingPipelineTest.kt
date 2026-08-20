package com.localsecuritycam.android.service

import android.view.Surface
import android.graphics.SurfaceTexture
import com.localsecuritycam.android.camera.DeviceOrientation
import com.localsecuritycam.android.camera.PreviewSurfaceAttachment
import com.localsecuritycam.android.camera.PreviewDiagnosticMode
import com.localsecuritycam.android.camera.VideoTransform
import com.localsecuritycam.android.camera.computeVideoOutputTransform
import com.localsecuritycam.android.diagnostics.CleanupFailure
import com.localsecuritycam.android.diagnostics.CleanupReport
import com.localsecuritycam.android.diagnostics.StreamErrorFormatter
import com.localsecuritycam.android.diagnostics.StreamErrorKind
import com.localsecuritycam.android.diagnostics.StreamFailure
import com.localsecuritycam.android.diagnostics.StreamMetrics
import com.localsecuritycam.android.diagnostics.StreamSubsystem
import com.localsecuritycam.android.diagnostics.SubsystemState
import com.localsecuritycam.android.settings.AppSettings
import com.localsecuritycam.android.settings.CameraLens
import com.localsecuritycam.android.settings.Resolution
import com.localsecuritycam.android.settings.StreamSettings
import com.localsecuritycam.android.streaming.EncodedAccessUnit
import com.localsecuritycam.android.streaming.H264ParameterSets
import com.localsecuritycam.android.streaming.RtspCredentials
import com.localsecuritycam.android.streaming.StreamBroadcaster
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.net.InetAddress
import java.util.concurrent.TimeUnit

class StreamingPipelineTest {
    @Test
    fun missingBasicAuthCredentialsPreventAnyResourceStart() {
        val fixture = Fixture(settings = AppSettings())

        fixture.startStreaming()

        assertEquals(listOf("camera.open", "camera.capture"), fixture.events)
        assertEquals(StreamErrorKind.CONFIGURATION, fixture.errors.single().kind)
        assertEquals(1, fixture.previewReadyCount)
        assertEquals(0, fixture.camera.stopCount)
    }

    @Test
    fun previewStartsWithoutEncoderNetworkOrRtsp() {
        val fixture = Fixture()

        fixture.pipeline.startPreview()

        assertEquals(listOf("camera.open", "camera.capture"), fixture.events)
        assertEquals(1, fixture.previewReadyCount)
        assertEquals(0, fixture.resources.encoders.size)
        assertEquals(0, fixture.resources.servers.size)
        assertEquals(0, fixture.camera.encoderInputs.size)
    }

    @Test
    fun latePreviewSurfaceAttachDoesNotRestartCameraOrCreateStreamOutput() {
        val fixture = Fixture()
        val attachment = previewAttachment()

        fixture.pipeline.startPreview()
        fixture.pipeline.setPreviewSurface(attachment)

        assertEquals(1, fixture.camera.captureStartCount)
        assertEquals(listOf(attachment), fixture.camera.previewSurfaces)
        assertEquals(0, fixture.camera.stopCount)
        assertTrue(fixture.resources.encoders.isEmpty())
        assertTrue(fixture.resources.servers.isEmpty())
        fixture.pipeline.stop()
    }

    @Test
    fun previewSurfaceReplacementAndDetachKeepTheActiveStreamOutput() {
        val fixture = Fixture(initialOrientation = DeviceOrientation(physicalOrientationDegrees = 0))
        val first = previewAttachment(640, 480)
        val replacement = previewAttachment(480, 640)

        fixture.startStreaming()
        fixture.pipeline.setPreviewSurface(first)
        fixture.pipeline.setPreviewSurface(null)
        fixture.pipeline.setPreviewSurface(replacement)

        assertEquals(listOf(first, null, replacement), fixture.camera.previewSurfaces)
        assertEquals(1, fixture.camera.captureStartCount)
        assertEquals(0, fixture.camera.stopCount)
        assertEquals(1, fixture.resources.encoders.size)
        assertEquals(1, fixture.resources.servers.size)
        fixture.pipeline.stop()
    }

    @Test
    fun previewSurfaceAttachExceptionIsReportedNonfatally() {
        val fixture = Fixture(initialOrientation = DeviceOrientation(physicalOrientationDegrees = 0))
        fixture.startStreaming()
        fixture.camera.throwOnPreviewSurface = true

        fixture.pipeline.setPreviewSurface(previewAttachment())

        assertEquals(1, fixture.previewDiagnostics.size)
        assertTrue(fixture.errors.isEmpty())
        assertEquals(0, fixture.camera.stopCount)
        assertEquals(1, fixture.resources.encoders.size)
        assertEquals(1, fixture.resources.servers.size)
        fixture.pipeline.stop()
    }

    @Test
    fun fatalCameraFailureStopsTheCameraAndStreamOutput() {
        val fixture = Fixture(initialOrientation = DeviceOrientation(physicalOrientationDegrees = 0))
        fixture.startStreaming()

        fixture.camera.error?.invoke(StreamFailure(StreamErrorKind.CAMERA, "camera disconnected"))

        assertEquals(StreamErrorKind.CAMERA, fixture.errors.single().kind)
        assertEquals(1, fixture.camera.stopCount)
        assertEquals(1, fixture.resources.encoders.single().stopCount)
        assertEquals(1, fixture.resources.servers.single().stopCount)
    }

    @Test
    fun rejectsUnauthenticatedNonLoopbackBeforeCreatingEncoder() {
        val fixture = Fixture()

        fixture.pipeline.startPreview()
        fixture.pipeline.startStreaming(InetAddress.getByName("192.0.2.10"))

        assertEquals(listOf("camera.open", "camera.capture"), fixture.events)
        assertEquals(1, fixture.previewReadyCount)
        assertEquals(0, fixture.resources.encoders.size)
        assertEquals(0, fixture.resources.servers.size)
        assertEquals(0, fixture.camera.stopCount)
        assertTrue(
            fixture.errors.single().detail.contains(
                "Basic authentication is required for a non-loopback RTSP listener",
            ),
        )
    }

    @Test
    fun startsEncoderWithTheActualPortraitOutputAndSdp() {
        val fixture = Fixture(initialOrientation = DeviceOrientation(physicalOrientationDegrees = 0))

        fixture.startStreaming()

        assertEquals(listOf("camera.open", "camera.capture", "encoder.start", "rtsp.start"), fixture.events)
        assertEquals("720x1280", fixture.encoder.settings.resolution.toString())
        assertEquals("720x1280", fixture.server.settings.resolution.toString())
        assertEquals(1, fixture.encoder.syncFrameRequests)
        assertEquals(1, fixture.readyCount)
        assertEquals(1, fixture.previewReadyCount)
        assertTrue(fixture.errors.isEmpty())
    }

    @Test
    fun stopStreamingKeepsThePreviewAndSingleCameraSessionActive() {
        val fixture = Fixture(initialOrientation = DeviceOrientation(physicalOrientationDegrees = 0))
        fixture.startStreaming()

        fixture.pipeline.stopStreaming()

        assertEquals(1, fixture.camera.captureStartCount)
        assertEquals(0, fixture.camera.stopCount)
        assertEquals(1, fixture.resources.encoders.first().stopCount)
        assertEquals(1, fixture.resources.servers.first().stopCount)
        assertEquals(1, fixture.previewReadyCount)
        assertTrue(fixture.events.containsAll(listOf("rtsp.stop", "encoder.stop")))
    }

    @Test
    fun temporaryPreviewSurfaceErrorDoesNotTerminateCameraOrStreamOutput() {
        val fixture = Fixture(initialOrientation = DeviceOrientation(physicalOrientationDegrees = 0))
        fixture.startStreaming()

        fixture.camera.previewError?.invoke(
            StreamFailure(StreamErrorKind.SURFACE, "preview swap failed"),
        )

        assertEquals(listOf("preview swap failed"), fixture.previewDiagnostics)
        assertEquals(0, fixture.camera.stopCount)
        assertEquals(1, fixture.resources.encoders.size)
        assertEquals(1, fixture.resources.servers.size)
    }

    @Test
    fun sameOutputGeometryUpdatesOnlyTheGlTransform() {
        val fixture = Fixture(initialOrientation = DeviceOrientation(physicalOrientationDegrees = 0))
        fixture.startStreaming()

        fixture.pipeline.setOrientation(DeviceOrientation(physicalOrientationDegrees = 180))
        fixture.await { fixture.camera.appliedTransforms.size == 1 }

        assertEquals(1, fixture.camera.captureStartCount)
        assertEquals(1, fixture.resources.encoders.size)
        assertEquals(1, fixture.resources.servers.size)
        assertEquals(270, fixture.camera.appliedTransforms.single().rotationDegrees)
        assertEquals("720x1280", fixture.camera.appliedTransforms.single().outputResolution.toString())
        assertEquals(0, fixture.encoder.stopCount)
        assertEquals(0, fixture.server.stopCount)
        fixture.pipeline.stop()
    }

    @Test
    fun portraitLandscapeChangeReplacesOnlyEncoderAndRtspGeneration() {
        val fixture = Fixture(initialOrientation = DeviceOrientation(physicalOrientationDegrees = 0))
        fixture.startStreaming()

        fixture.pipeline.setOrientation(DeviceOrientation(physicalOrientationDegrees = 90))
        fixture.await {
            fixture.resources.encoders.size == 2 &&
                fixture.resources.servers.size == 2 &&
                fixture.camera.encoderInputs.size == 2
        }

        assertEquals(1, fixture.camera.captureStartCount)
        assertEquals(listOf("720x1280", "1280x720"), fixture.resources.encoders.map { it.settings.resolution.toString() })
        assertEquals(listOf("720x1280", "1280x720"), fixture.resources.servers.map { it.settings.resolution.toString() })
        assertEquals(1, fixture.resources.encoders.first().stopCount)
        assertEquals(1, fixture.resources.servers.first().stopCount)
        assertEquals(2, fixture.camera.encoderInputs.size)
        assertEquals(1, fixture.camera.encoderDetaches)
        assertTrue(fixture.errors.isEmpty())
        fixture.pipeline.stop()
    }

    @Test
    fun staleOldEncoderFormatCannotStartAnotherRtspServer() {
        val fixture = Fixture(initialOrientation = DeviceOrientation(physicalOrientationDegrees = 0))
        fixture.startStreaming()
        val oldEncoder = fixture.encoder

        fixture.pipeline.setOrientation(DeviceOrientation(physicalOrientationDegrees = 90))
        fixture.await { fixture.resources.encoders.size == 2 && fixture.resources.servers.size == 2 }
        oldEncoder.emitOutputFormat(Resolution(720, 1280))

        assertEquals(2, fixture.resources.servers.size)
        assertTrue(fixture.errors.isEmpty())
        fixture.pipeline.stop()
    }

    @Test
    fun mediaCodecFormatMismatchFailsBeforePublishingAnRtspServer() {
        val fixture = Fixture(autoFormat = false, initialOrientation = DeviceOrientation(physicalOrientationDegrees = 0))
        fixture.startStreaming()

        fixture.encoder.emitOutputFormat(Resolution(1280, 720))

        assertEquals(StreamErrorKind.MEDIACODEC, fixture.errors.single().kind)
        assertTrue(fixture.resources.servers.isEmpty())
        assertEquals(0, fixture.camera.stopCount)
        assertEquals(1, fixture.camera.captureStartCount)
        assertTrue(fixture.events.contains("camera.capture"))
    }

    @Test
    fun requestsAnIdrAfterNewOutputServerStarts() {
        val fixture = Fixture(initialOrientation = DeviceOrientation(physicalOrientationDegrees = 0))
        fixture.startStreaming()
        fixture.pipeline.setOrientation(DeviceOrientation(physicalOrientationDegrees = 90))
        fixture.await {
            fixture.resources.servers.size == 2 &&
                fixture.resources.encoders.size == 2 &&
                fixture.resources.encoders[1].syncFrameRequests == 1
        }

        assertEquals(1, fixture.resources.encoders[0].syncFrameRequests)
        assertEquals(1, fixture.resources.encoders[1].syncFrameRequests)
        fixture.pipeline.stop()
    }

    @Test
    fun cleanupStopsRtspThenEncoderThenTheSingleCameraLifecycle() {
        val fixture = Fixture(initialOrientation = DeviceOrientation(physicalOrientationDegrees = 0))
        fixture.startStreaming()

        fixture.pipeline.stop()
        fixture.pipeline.stop()

        assertEquals(
            listOf("camera.open", "camera.capture", "encoder.start", "rtsp.start", "rtsp.stop", "encoder.stop", "camera.stop"),
            fixture.events,
        )
    }

    @Test
    fun cleanupContinuesWhenOutputStopsReportFailures() {
        val fixture = Fixture(serverStopFailure = true, encoderStopFailure = true)
        fixture.startStreaming()

        fixture.pipeline.stop()

        assertTrue(fixture.events.containsAll(listOf("rtsp.stop", "encoder.stop", "camera.stop")))
    }

    @Test
    fun highDefinitionPortraitUsesTheSwappedCodecAndSdpDimensions() {
        val fixture = Fixture(
            settings = AppSettings(stream = StreamSettings(resolution = Resolution(1920, 1080), authEnabled = false)),
            initialOrientation = DeviceOrientation(physicalOrientationDegrees = 0),
        )

        fixture.startStreaming()

        assertEquals("1080x1920", fixture.encoder.settings.resolution.toString())
        assertEquals("1080x1920", fixture.server.settings.resolution.toString())
        fixture.pipeline.stop()
    }

    @Test
    fun displayFallbackCanReconfigureWhenNoPhysicalSampleExists() {
        val fixture = Fixture(initialOrientation = DeviceOrientation(displayRotationDegrees = 0))
        fixture.startStreaming()

        fixture.pipeline.setOrientation(DeviceOrientation(displayRotationDegrees = 90))
        fixture.await { fixture.resources.encoders.size == 2 }

        assertEquals(listOf("720x1280", "1280x720"), fixture.resources.encoders.map { it.settings.resolution.toString() })
        assertEquals(1, fixture.camera.captureStartCount)
        fixture.pipeline.stop()
    }

    @Test
    fun completePhysicalCycleKeepsOneCameraAcrossMultipleOutputGenerations() {
        val fixture = Fixture(initialOrientation = DeviceOrientation(physicalOrientationDegrees = 0))
        fixture.startStreaming()

        fixture.pipeline.setOrientation(DeviceOrientation(physicalOrientationDegrees = 90))
        fixture.await { fixture.resources.encoders.size == 2 }
        fixture.pipeline.setOrientation(DeviceOrientation(physicalOrientationDegrees = 180))
        fixture.await { fixture.resources.encoders.size == 3 }

        assertEquals(1, fixture.camera.captureStartCount)
        assertEquals(listOf("720x1280", "1280x720", "720x1280"), fixture.resources.encoders.map { it.settings.resolution.toString() })
        fixture.pipeline.stop()
    }

    @Test
    fun rtspAndReadyWaitForConfirmedMediaCodecOutputFormat() {
        val fixture = Fixture(autoFormat = false, initialOrientation = DeviceOrientation(physicalOrientationDegrees = 0))
        fixture.startStreaming()

        assertTrue(fixture.resources.servers.isEmpty())
        assertEquals(0, fixture.readyCount)
        fixture.encoder.emitOutputFormat(Resolution(720, 1280))

        assertEquals(1, fixture.resources.servers.size)
        assertEquals(1, fixture.readyCount)
        fixture.pipeline.stop()
    }

    @Test
    fun orientationJustAfterStartupStillKeepsTheSingleCameraCaptureSession() {
        val fixture = Fixture(autoFormat = false, initialOrientation = DeviceOrientation(physicalOrientationDegrees = 0))
        fixture.startStreaming()

        fixture.pipeline.setOrientation(DeviceOrientation(physicalOrientationDegrees = 90))
        fixture.encoder.emitOutputFormat(Resolution(720, 1280))
        fixture.await { fixture.resources.encoders.size == 2 }

        assertEquals(1, fixture.camera.captureStartCount)
        assertEquals(listOf("720x1280", "1280x720"), fixture.resources.encoders.map { it.settings.resolution.toString() })
        fixture.pipeline.stop()
    }

    @Test
    fun failedReplacementEncoderCleansTheOldOutputAndCameraWithoutRestartingCamera2() {
        val fixture = Fixture(
            failSecondEncoderStart = true,
            initialOrientation = DeviceOrientation(physicalOrientationDegrees = 0),
        )
        fixture.startStreaming()

        fixture.pipeline.setOrientation(DeviceOrientation(physicalOrientationDegrees = 90))
        fixture.await { fixture.errors.isNotEmpty() }

        assertEquals(StreamErrorKind.MEDIACODEC, fixture.errors.single().kind)
        assertEquals(1, fixture.camera.captureStartCount)
        assertEquals(0, fixture.camera.stopCount)
        assertTrue(fixture.events.containsAll(listOf("rtsp.stop", "encoder.stop")))
    }

    private class Fixture(
        val settings: AppSettings = AppSettings(stream = StreamSettings(authEnabled = false)),
        private val autoFormat: Boolean = true,
        private val encoderStopFailure: Boolean = false,
        private val serverStopFailure: Boolean = false,
        private val failSecondEncoderStart: Boolean = false,
        initialOrientation: DeviceOrientation = DeviceOrientation(physicalOrientationDegrees = 0),
    ) {
        val events = mutableListOf<String>()
        val errors = mutableListOf<StreamFailure>()
        val previewDiagnostics = mutableListOf<String>()
        var previewReadyCount = 0
        var readyCount = 0
        val camera = FakeCamera(events)
        val resources = FakeResources(
            events,
            camera,
            autoFormat,
            encoderStopFailure,
            serverStopFailure,
            failSecondEncoderStart,
        )
        val encoder: FakeEncoder
            get() = resources.encoders.first()
        val server: FakeServer
            get() = resources.servers.first()
        val pipeline = StreamingPipeline(
            request = StreamPipelineRequest(
                settings = settings,
                preview = null,
                initialOrientation = initialOrientation,
                initialReconnectCount = 0L,
                initialSessionRestartCount = 0L,
            ),
            callbacks = StreamPipelineCallbacks(
                onPreviewReady = { previewReadyCount++ },
                onStreamReady = { readyCount++ },
                onError = { _, error -> errors += error },
                onPreviewDiagnostic = { previewDiagnostics += it.detail },
            ),
            resources = resources,
            dispatch = { action -> action(); true },
            onMetricsChanged = {},
        )

        fun startStreaming() {
            pipeline.startPreview()
            pipeline.startStreaming(InetAddress.getByName("127.0.0.1"))
        }

        fun await(condition: () -> Boolean) {
            val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2)
            while (!condition() && System.nanoTime() < deadline) Thread.sleep(10)
            assertTrue("timed out waiting for orientation reconfiguration", condition())
        }
    }

    private fun previewAttachment(width: Int = 640, height: Int = 480): PreviewSurfaceAttachment =
        PreviewSurfaceAttachment(Surface(SurfaceTexture(0)), width, height)

    private class FakeResources(
        private val events: MutableList<String>,
        private val camera: FakeCamera,
        private val autoFormat: Boolean,
        private val encoderStopFailure: Boolean,
        private val serverStopFailure: Boolean,
        private val failSecondEncoderStart: Boolean,
    ) : StreamingResourceFactory {
        val encoders = mutableListOf<FakeEncoder>()
        val servers = mutableListOf<FakeServer>()

        override fun createCamera(
            onFrame: () -> Unit,
            onError: (StreamFailure) -> Unit,
            onPreviewError: (StreamFailure) -> Unit,
            onPreviewRecovered: () -> Unit,
            previewDiagnosticMode: PreviewDiagnosticMode,
        ): CameraPort {
            camera.error = onError
            camera.previewError = onPreviewError
            return camera
        }

        override fun createEncoder(
            settings: StreamSettings,
            onAccessUnit: (EncodedAccessUnit) -> Unit,
            onParameterSets: (H264ParameterSets) -> Unit,
            onOutputFormat: (Resolution) -> Unit,
            onError: (StreamFailure) -> Unit,
        ): EncoderPort = FakeEncoder(
            events = events,
            settings = settings,
            autoFormat = autoFormat,
            stopFailure = encoderStopFailure,
            startFailure = failSecondEncoderStart && encoders.isNotEmpty(),
            onOutputFormat = onOutputFormat,
        ).also(encoders::add)

        override fun createServer(
            settings: StreamSettings,
            metrics: StreamMetrics,
            broadcaster: StreamBroadcaster,
            credentialsProvider: () -> RtspCredentials?,
            onError: (StreamFailure) -> Unit,
        ): RtspPort = FakeServer(events, settings, serverStopFailure).also(servers::add)
    }

    private class FakeCamera(
        private val events: MutableList<String>,
    ) : CameraPort {
        var error: ((StreamFailure) -> Unit)? = null
        var previewError: ((StreamFailure) -> Unit)? = null
        private var settings: StreamSettings? = null
        var captureStartCount = 0
        var stopCount = 0
        var throwOnPreviewSurface = false
        val appliedTransforms = mutableListOf<VideoTransform>()
        val encoderInputs = mutableListOf<EncoderInput>()
        val previewSurfaces = mutableListOf<PreviewSurfaceAttachment?>()
        var encoderDetaches = 0

        override fun open(settings: StreamSettings, onOpened: () -> Unit) {
            events += "camera.open"
            this.settings = settings
            onOpened()
        }

        override fun outputTransform(orientation: DeviceOrientation): VideoTransform = computeVideoOutputTransform(
            sensorOrientation = 90,
            deviceOrientation = orientation,
            lensFacing = settings?.lens ?: CameraLens.BACK,
            sourceWidth = settings?.resolution?.width ?: 1280,
            sourceHeight = settings?.resolution?.height ?: 720,
        )

        override fun startCapture(
            encoderInput: EncoderInput?,
            preview: com.localsecuritycam.android.camera.PreviewSurfaceAttachment?,
            initialTransform: VideoTransform,
            onReady: () -> Unit,
        ) {
            events += "camera.capture"
            captureStartCount++
            if (encoderInput != null) encoderInputs += encoderInput
            onReady()
        }

        override fun setPreviewSurface(surface: PreviewSurfaceAttachment?) {
            previewSurfaces += surface
            if (throwOnPreviewSurface) error("preview Surface attach failed")
        }

        override fun setEncoderInput(encoderInput: EncoderInput?) {
            if (encoderInput == null) encoderDetaches++ else encoderInputs += encoderInput
        }

        override fun applyVideoTransform(transform: VideoTransform) {
            appliedTransforms += transform
        }

        override fun stop(): CleanupReport {
            events += "camera.stop"
            stopCount++
            return CleanupReport()
        }
    }

    private class FakeEncoder(
        private val events: MutableList<String>,
        val settings: StreamSettings,
        private val autoFormat: Boolean,
        private val stopFailure: Boolean,
        private val startFailure: Boolean,
        private val onOutputFormat: (Resolution) -> Unit,
    ) : EncoderPort {
        var startCount = 0
        var stopCount = 0
        var syncFrameRequests = 0

        override fun start(): EncoderInput {
            events += "encoder.start"
            startCount++
            if (startFailure) {
                throw com.localsecuritycam.android.diagnostics.StreamFailureException(
                    StreamErrorFormatter.fromMessage(StreamErrorKind.MEDIACODEC, "replacement encoder failed"),
                )
            }
            if (autoFormat) emitOutputFormat(settings.resolution)
            return object : EncoderInput {}
        }

        fun emitOutputFormat(resolution: Resolution) = onOutputFormat(resolution)

        override fun requestSyncFrame() {
            syncFrameRequests++
        }

        override fun stop(): CleanupReport {
            events += "encoder.stop"
            stopCount++
            return if (stopFailure) CleanupReport(listOf(CleanupFailure("encoder", "stop failure"))) else CleanupReport()
        }
    }

    private class FakeServer(
        private val events: MutableList<String>,
        val settings: StreamSettings,
        private val stopFailure: Boolean,
    ) : RtspPort {
        var stopCount = 0

        override fun start(bindAddress: InetAddress) {
            events += "rtsp.start"
        }

        override fun stop(): CleanupReport {
            events += "rtsp.stop"
            stopCount++
            return if (stopFailure) CleanupReport(listOf(CleanupFailure("RTSP", "stop failure"))) else CleanupReport()
        }
    }
}
