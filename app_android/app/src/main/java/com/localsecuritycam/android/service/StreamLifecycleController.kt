package com.localsecuritycam.android.service

import com.localsecuritycam.android.camera.CameraOrientationState
import com.localsecuritycam.android.diagnostics.CleanupCollector
import com.localsecuritycam.android.diagnostics.CleanupReport
import com.localsecuritycam.android.diagnostics.StreamErrorFormatter
import com.localsecuritycam.android.diagnostics.StreamErrorKind
import com.localsecuritycam.android.diagnostics.StreamErrorLogger
import com.localsecuritycam.android.diagnostics.StreamFailure
import com.localsecuritycam.android.diagnostics.StreamFailureException
import com.localsecuritycam.android.diagnostics.StreamSubsystem
import com.localsecuritycam.android.diagnostics.SubsystemState
import java.net.InetAddress
import java.util.concurrent.atomic.AtomicLong

/**
 * Owns the persistent preview pipeline. Stream output generations are started
 * and stopped without replacing this controller's active Camera2 pipeline.
 */
internal class StreamLifecycleController(
    private val factory: StreamPipelineFactory,
    private val dispatch: ((() -> Unit) -> Boolean),
    private val onPreviewReady: () -> Unit,
    private val onStreamReady: () -> Unit,
    private val onError: (PipelineStage, StreamFailure) -> Unit,
    private val onPreviewDiagnostic: (StreamFailure) -> Unit = {},
    private val onPreviewRecovered: () -> Unit = {},
    private val onSubsystemStateChanged: (StreamSubsystem, SubsystemState) -> Unit = { _, _ -> },
    private val onOrientationChanged: (CameraOrientationState) -> Unit = {},
) {
    private val lock = Any()
    private val generation = AtomicLong(0L)
    private var starting = false
    private var active: StreamPipeline? = null

    val activePipeline: StreamPipeline?
        get() = synchronized(lock) { active }

    fun start(request: StreamPipelineRequest): Boolean {
        var created: StreamPipeline? = null
        var failure: StreamFailure? = null
        synchronized(lock) {
            if (starting || active != null) return false
            starting = true
            val token = generation.incrementAndGet()
            var callbackPipeline: StreamPipeline? = null
            val callbacks = StreamPipelineCallbacks(
                onPreviewReady = {
                    callbackPipeline?.let { pipeline ->
                        dispatchOrHandle(
                            token = token,
                            pipeline = pipeline,
                            action = { handleReady(token, pipeline, PipelineStage.PREVIEW) },
                            failure = StreamFailure(
                                StreamErrorKind.THREAD,
                                "preview ready callback dispatch rejected",
                            ),
                        )
                    }
                },
                onStreamReady = {
                    callbackPipeline?.let { pipeline ->
                        dispatchOrHandle(
                            token = token,
                            pipeline = pipeline,
                            action = { handleReady(token, pipeline, PipelineStage.STREAM) },
                            failure = StreamFailure(
                                StreamErrorKind.THREAD,
                                "stream ready callback dispatch rejected",
                            ),
                        )
                    }
                },
                onError = { stage, error ->
                    callbackPipeline?.let { pipeline ->
                        dispatchOrHandle(
                            token = token,
                            pipeline = pipeline,
                            action = { handleError(token, pipeline, stage, error) },
                            failure = StreamFailure(
                                StreamErrorKind.THREAD,
                                "${stage.name.lowercase()} error callback dispatch rejected",
                            ),
                        )
                    }
                },
                onPreviewDiagnostic = { error ->
                    callbackPipeline?.let { pipeline ->
                        if (!dispatch {
                                handlePreviewDiagnostic(token, pipeline, error)
                            }
                        ) {
                            StreamErrorLogger.error(
                                StreamFailure(
                                    StreamErrorKind.THREAD,
                                    "preview diagnostic callback dispatch rejected",
                                ),
                            )
                        }
                    }
                },
                onPreviewRecovered = {
                    callbackPipeline?.let { pipeline ->
                        if (!dispatch {
                                handlePreviewRecovered(token, pipeline)
                            }
                        ) {
                            StreamErrorLogger.error(
                                StreamFailure(
                                    StreamErrorKind.THREAD,
                                    "preview recovery callback dispatch rejected",
                                ),
                            )
                        }
                    }
                },
                onSubsystemStateChanged = { subsystem, state ->
                    callbackPipeline?.let { pipeline ->
                        if (!dispatch {
                                handleSubsystemState(token, pipeline, subsystem, state)
                            }
                        ) {
                            StreamErrorLogger.error(
                                StreamFailure(
                                    StreamErrorKind.THREAD,
                                    "diagnostic state callback dispatch rejected",
                                ),
                            )
                        }
                    }
                },
                onOrientationChanged = { orientation ->
                    callbackPipeline?.let { pipeline ->
                        if (!dispatch {
                                handleOrientation(token, pipeline, orientation)
                            }
                        ) {
                            StreamErrorLogger.error(
                                StreamFailure(
                                    StreamErrorKind.THREAD,
                                    "orientation callback dispatch rejected",
                                ),
                            )
                        }
                    }
                },
            )
            try {
                val pipeline = factory.create(request, callbacks)
                callbackPipeline = pipeline
                created = pipeline
                active = pipeline
                pipeline.startPreview()
                starting = false
            } catch (error: StreamFailureException) {
                starting = false
                if (active === created) {
                    active = null
                    generation.incrementAndGet()
                    failure = failureWithCleanup(error.failure, created)
                } else if (created == null) {
                    generation.incrementAndGet()
                    failure = error.failure
                }
            } catch (error: Exception) {
                starting = false
                if (active === created) {
                    active = null
                    generation.incrementAndGet()
                    failure = failureWithCleanup(
                        StreamErrorFormatter.fromThrowable(StreamErrorKind.CONFIGURATION, error),
                        created,
                    )
                } else if (created == null) {
                    generation.incrementAndGet()
                    failure = StreamErrorFormatter.fromThrowable(StreamErrorKind.CONFIGURATION, error)
                }
            }
        }
        failure?.let { notifyError(PipelineStage.PREVIEW, it) }
        return true
    }

    fun startStreaming(bindAddress: InetAddress): Boolean {
        val pipeline = synchronized(lock) { active } ?: return false
        try {
            pipeline.startStreaming(bindAddress)
        } catch (error: StreamFailureException) {
            notifyError(PipelineStage.STREAM, error.failure)
        } catch (error: Exception) {
            notifyError(
                PipelineStage.STREAM,
                StreamErrorFormatter.fromThrowable(StreamErrorKind.MEDIACODEC, error),
            )
        }
        return true
    }

    fun stopStreaming(): CleanupReport {
        val pipeline = synchronized(lock) { active } ?: return CleanupReport()
        val cleanup = CleanupCollector()
        cleanup.run("stream output") { pipeline.stopStreaming() }
        return cleanup.report()
    }

    fun stop(): CleanupReport {
        synchronized(lock) {
            generation.incrementAndGet()
            starting = false
            val pipeline = active.also { active = null }
            val cleanup = CleanupCollector()
            if (pipeline != null) {
                cleanup.run("pipeline") { pipeline.stop() }
            }
            return cleanup.report()
        }
    }

    private fun dispatchOrHandle(
        token: Long,
        pipeline: StreamPipeline,
        action: () -> Unit,
        failure: StreamFailure,
    ) {
        if (!dispatch(action)) handleError(token, pipeline, failureStage(failure), failure)
    }

    private fun failureStage(failure: StreamFailure): PipelineStage =
        if (failure.kind == StreamErrorKind.MEDIACODEC || failure.kind == StreamErrorKind.ENCODER) {
            PipelineStage.STREAM
        } else {
            PipelineStage.PREVIEW
        }

    private fun handleReady(token: Long, pipeline: StreamPipeline, stage: PipelineStage) {
        val accepted = synchronized(lock) { token == generation.get() && active === pipeline }
        if (!accepted) return
        try {
            if (stage == PipelineStage.PREVIEW) onPreviewReady() else onStreamReady()
        } catch (error: Exception) {
            StreamErrorLogger.observer(error)
        }
    }

    private fun handleError(
        token: Long,
        pipeline: StreamPipeline,
        stage: PipelineStage,
        failure: StreamFailure,
    ) {
        if (stage == PipelineStage.STREAM) {
            val accepted = synchronized(lock) { token == generation.get() && active === pipeline }
            if (accepted) notifyError(stage, failure)
            return
        }
        val finalFailure: StreamFailure?
        synchronized(lock) {
            if (token != generation.get() || active !== pipeline) return
            active = null
            starting = false
            generation.incrementAndGet()
            finalFailure = failureWithCleanup(failure, pipeline)
        }
        finalFailure?.let { notifyError(PipelineStage.PREVIEW, it) }
    }

    private fun handlePreviewDiagnostic(
        token: Long,
        pipeline: StreamPipeline,
        failure: StreamFailure,
    ) {
        val accepted = synchronized(lock) { token == generation.get() && active === pipeline }
        if (!accepted) return
        try {
            onPreviewDiagnostic(failure)
        } catch (error: Exception) {
            StreamErrorLogger.observer(error)
        }
    }

    private fun handlePreviewRecovered(token: Long, pipeline: StreamPipeline) {
        val accepted = synchronized(lock) { token == generation.get() && active === pipeline }
        if (!accepted) return
        try {
            onPreviewRecovered()
        } catch (error: Exception) {
            StreamErrorLogger.observer(error)
        }
    }

    private fun handleSubsystemState(
        token: Long,
        pipeline: StreamPipeline,
        subsystem: StreamSubsystem,
        state: SubsystemState,
    ) {
        val accepted = synchronized(lock) { token == generation.get() && active === pipeline }
        if (!accepted) return
        try {
            onSubsystemStateChanged(subsystem, state)
        } catch (error: Exception) {
            StreamErrorLogger.observer(error)
        }
    }

    private fun handleOrientation(
        token: Long,
        pipeline: StreamPipeline,
        orientation: CameraOrientationState,
    ) {
        val accepted = synchronized(lock) { token == generation.get() && active === pipeline }
        if (!accepted) return
        try {
            onOrientationChanged(orientation)
        } catch (error: Exception) {
            StreamErrorLogger.observer(error)
        }
    }

    private fun failureWithCleanup(failure: StreamFailure, pipeline: StreamPipeline?): StreamFailure {
        if (pipeline == null) return failure
        val cleanup = CleanupCollector()
        cleanup.run("pipeline") { pipeline.stop() }
        return StreamErrorFormatter.withCleanup(failure, cleanup.report())
    }

    private fun notifyError(stage: PipelineStage, failure: StreamFailure) {
        StreamErrorLogger.error(failure)
        try {
            onError(stage, failure)
        } catch (error: Exception) {
            StreamErrorLogger.observer(error)
        }
    }
}
