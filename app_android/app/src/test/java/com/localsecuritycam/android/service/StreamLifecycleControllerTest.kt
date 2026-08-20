package com.localsecuritycam.android.service

import com.localsecuritycam.android.camera.PreviewSurfaceAttachment
import com.localsecuritycam.android.diagnostics.CleanupReport
import com.localsecuritycam.android.diagnostics.StreamErrorKind
import com.localsecuritycam.android.diagnostics.StreamFailure
import com.localsecuritycam.android.diagnostics.StreamMetrics
import com.localsecuritycam.android.diagnostics.StreamSubsystem
import com.localsecuritycam.android.diagnostics.SubsystemState
import com.localsecuritycam.android.settings.AppSettings
import com.localsecuritycam.android.settings.StreamSettings
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test
import java.net.InetAddress

class StreamLifecycleControllerTest {
    @Test
    fun startOwnsPreviewAndStreamingIsRequestedSeparately() {
        val factory = FakeFactory()
        val previewReady = mutableListOf<Unit>()
        val streamReady = mutableListOf<Unit>()
        val errors = mutableListOf<Pair<PipelineStage, StreamFailure>>()
        val controller = controller(factory, previewReady, streamReady, errors)

        assertTrue(controller.start(request()))
        val pipeline = factory.pipelines.single()
        assertEquals(1, pipeline.previewStartCount)
        assertEquals(0, pipeline.streamStartCount)

        pipeline.previewReady()
        assertEquals(1, previewReady.size)

        assertTrue(controller.startStreaming(InetAddress.getByName("127.0.0.1")))
        assertEquals(1, pipeline.streamStartCount)
        pipeline.streamReady()

        assertEquals(1, streamReady.size)
        assertTrue(controller.activePipeline === pipeline)
        assertTrue(errors.isEmpty())
    }

    @Test
    fun stopStreamingKeepsThePreviewPipelineActive() {
        val factory = FakeFactory()
        val controller = controller(factory)
        controller.start(request())
        val pipeline = factory.pipelines.single()

        controller.startStreaming(InetAddress.getByName("127.0.0.1"))
        controller.stopStreaming()

        assertSame(pipeline, controller.activePipeline)
        assertEquals(1, pipeline.streamStopCount)
        assertEquals(0, pipeline.stopCount)
    }

    @Test
    fun stopReleasesPipelineAndIsIdempotent() {
        val factory = FakeFactory()
        val controller = controller(factory)
        controller.start(request())
        val pipeline = factory.pipelines.single()

        controller.stop()
        controller.stop()

        assertNull(controller.activePipeline)
        assertEquals(1, pipeline.stopCount)
    }

    @Test
    fun startStopStartCreatesOnlyOneActivePipelineAtATime() {
        val factory = FakeFactory()
        val controller = controller(factory)

        controller.start(request())
        val first = factory.pipelines.single()
        controller.stop()
        controller.start(request())
        val second = factory.pipelines.last()

        assertEquals(2, factory.createCount)
        assertEquals(1, first.stopCount)
        assertSame(second, controller.activePipeline)
        assertFalse(first === controller.activePipeline)
    }

    @Test
    fun doubleStartIsIgnored() {
        val factory = FakeFactory()
        val controller = controller(factory)

        assertTrue(controller.start(request()))
        assertFalse(controller.start(request()))

        assertEquals(1, factory.createCount)
        assertEquals(1, factory.pipelines.single().previewStartCount)
    }

    @Test
    fun lateCallbacksAfterStopAreIgnored() {
        val factory = FakeFactory()
        val previewReady = mutableListOf<Unit>()
        val streamReady = mutableListOf<Unit>()
        val errors = mutableListOf<Pair<PipelineStage, StreamFailure>>()
        val controller = controller(factory, previewReady, streamReady, errors)

        controller.start(request())
        val pipeline = factory.pipelines.single()
        controller.stop()
        pipeline.previewReady()
        pipeline.streamReady()
        pipeline.fail(PipelineStage.PREVIEW, "late preview failure")
        pipeline.fail(PipelineStage.STREAM, "late stream failure")

        assertTrue(previewReady.isEmpty())
        assertTrue(streamReady.isEmpty())
        assertTrue(errors.isEmpty())
        assertEquals(1, pipeline.stopCount)
    }

    @Test
    fun streamErrorDoesNotDestroyThePreviewPipeline() {
        val factory = FakeFactory()
        val errors = mutableListOf<Pair<PipelineStage, StreamFailure>>()
        val controller = controller(factory, errors = errors)

        controller.start(request())
        val pipeline = factory.pipelines.single()
        pipeline.fail(PipelineStage.STREAM, "encoder failed")

        assertSame(pipeline, controller.activePipeline)
        assertEquals(PipelineStage.STREAM, errors.single().first)
        assertEquals(0, pipeline.stopCount)
    }

    @Test
    fun temporaryPreviewDiagnosticKeepsThePipelineAndCanRecover() {
        val factory = FakeFactory()
        val diagnostics = mutableListOf<StreamFailure>()
        val streamReady = mutableListOf<Unit>()
        var recovered = 0
        val controller = controller(
            factory,
            streamReady = streamReady,
            previewDiagnostics = diagnostics,
            onPreviewRecovered = { recovered++ },
        )

        controller.start(request())
        val pipeline = factory.pipelines.single()
        pipeline.previewReady()
        assertTrue(controller.startStreaming(InetAddress.getByName("127.0.0.1")))
        pipeline.streamReady()
        pipeline.previewDiagnostic("preview swap failed")
        pipeline.previewRecovered()

        assertSame(pipeline, controller.activePipeline)
        assertEquals("preview swap failed", diagnostics.single().detail)
        assertEquals(1, recovered)
        assertEquals(1, streamReady.size)
        assertEquals(0, pipeline.stopCount)
    }

    @Test
    fun previewErrorCleansUpAndAllowsRetry() {
        val factory = FakeFactory()
        val errors = mutableListOf<Pair<PipelineStage, StreamFailure>>()
        val controller = controller(factory, errors = errors)

        controller.start(request())
        val first = factory.pipelines.single()
        first.fail(PipelineStage.PREVIEW, "first preview failed")

        assertNull(controller.activePipeline)
        assertEquals("first preview failed", errors.single().second.detail)
        assertEquals(1, first.stopCount)

        controller.start(request())
        assertEquals(2, factory.createCount)
        assertSame(factory.pipelines.last(), controller.activePipeline)
    }

    @Test
    fun synchronousPreviewStartErrorCleansUpAndAllowsRetry() {
        val factory = FakeFactory(throwOnPreviewStart = true)
        val errors = mutableListOf<Pair<PipelineStage, StreamFailure>>()
        val controller = controller(factory, errors = errors)

        controller.start(request())
        val failed = factory.pipelines.single()

        assertNull(controller.activePipeline)
        assertEquals(1, failed.stopCount)
        assertEquals(1, errors.size)
        assertEquals(PipelineStage.PREVIEW, errors.single().first)

        factory.throwOnPreviewStart = false
        controller.start(request())

        assertEquals(2, factory.createCount)
        assertSame(factory.pipelines.last(), controller.activePipeline)
    }

    @Test
    fun lateSubsystemStateCallbackAfterStopIsIgnored() {
        val factory = FakeFactory()
        val states = mutableListOf<Pair<StreamSubsystem, SubsystemState>>()
        val controller = controller(factory, subsystemStates = states)

        controller.start(request())
        val pipeline = factory.pipelines.single()
        pipeline.subsystemState(StreamSubsystem.CAMERA, SubsystemState.STARTING)
        controller.stop()
        pipeline.subsystemState(StreamSubsystem.CAMERA, SubsystemState.RUNNING)

        assertEquals(listOf(StreamSubsystem.CAMERA to SubsystemState.STARTING), states)
    }

    private fun controller(
        factory: FakeFactory,
        previewReady: MutableList<Unit> = mutableListOf(),
        streamReady: MutableList<Unit> = mutableListOf(),
        errors: MutableList<Pair<PipelineStage, StreamFailure>> = mutableListOf(),
        subsystemStates: MutableList<Pair<StreamSubsystem, SubsystemState>> = mutableListOf(),
        previewDiagnostics: MutableList<StreamFailure> = mutableListOf(),
        onPreviewRecovered: () -> Unit = {},
    ) = StreamLifecycleController(
        factory = factory,
        dispatch = { action -> action(); true },
        onPreviewReady = { previewReady += Unit },
        onStreamReady = { streamReady += Unit },
        onError = { stage, error -> errors += stage to error },
        onPreviewDiagnostic = { previewDiagnostics += it },
        onPreviewRecovered = onPreviewRecovered,
        onSubsystemStateChanged = { subsystem, state -> subsystemStates += subsystem to state },
    )

    private fun request() = StreamPipelineRequest(
        settings = AppSettings(stream = StreamSettings(authEnabled = false)),
        preview = null,
        initialReconnectCount = 0L,
        initialSessionRestartCount = 0L,
    )

    private class FakeFactory(
        var throwOnPreviewStart: Boolean = false,
    ) : StreamPipelineFactory {
        val pipelines = mutableListOf<FakePipeline>()
        var createCount = 0

        override fun create(
            request: StreamPipelineRequest,
            callbacks: StreamPipelineCallbacks,
        ): StreamPipeline {
            createCount++
            return FakePipeline(callbacks, throwOnPreviewStart).also(pipelines::add)
        }
    }

    private class FakePipeline(
        private val callbacks: StreamPipelineCallbacks,
        private val throwOnPreviewStart: Boolean,
    ) : StreamPipeline {
        override val metrics = StreamMetrics()
        var previewStartCount = 0
        var streamStartCount = 0
        var streamStopCount = 0
        var stopCount = 0

        override fun startPreview() {
            previewStartCount++
            if (throwOnPreviewStart) error("fake preview start failure")
        }

        override fun startStreaming(bindAddress: InetAddress) {
            streamStartCount++
        }

        override fun stopStreaming(): CleanupReport {
            streamStopCount++
            return CleanupReport()
        }

        override fun stop(): CleanupReport {
            stopCount++
            return CleanupReport()
        }

        override fun setPreviewSurface(surface: PreviewSurfaceAttachment?) = Unit

        fun previewReady() = callbacks.onPreviewReady()

        fun streamReady() = callbacks.onStreamReady()

        fun subsystemState(subsystem: StreamSubsystem, state: SubsystemState) =
            callbacks.onSubsystemStateChanged(subsystem, state)

        fun fail(stage: PipelineStage, message: String) = callbacks.onError(
            stage,
            StreamFailure(StreamErrorKind.CAMERA, message),
        )

        fun previewDiagnostic(message: String) = callbacks.onPreviewDiagnostic(
            StreamFailure(StreamErrorKind.SURFACE, message),
        )

        fun previewRecovered() = callbacks.onPreviewRecovered()
    }
}
