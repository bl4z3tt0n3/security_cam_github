package com.localsecuritycam.android.encoder

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.os.Bundle
import android.view.Surface
import com.localsecuritycam.android.diagnostics.CleanupCollector
import com.localsecuritycam.android.diagnostics.CleanupReport
import com.localsecuritycam.android.diagnostics.StreamErrorFormatter
import com.localsecuritycam.android.diagnostics.StreamErrorKind
import com.localsecuritycam.android.diagnostics.StreamErrorLogger
import com.localsecuritycam.android.diagnostics.StreamFailure
import com.localsecuritycam.android.diagnostics.StreamFailureException
import com.localsecuritycam.android.settings.StreamSettings
import com.localsecuritycam.android.streaming.EncodedAccessUnit
import com.localsecuritycam.android.streaming.H264ParameterSets
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

private const val KEY_PIXEL_ASPECT_RATIO_WIDTH = "pixel-aspect-ratio-width"
private const val KEY_PIXEL_ASPECT_RATIO_HEIGHT = "pixel-aspect-ratio-height"

internal class H264Encoder(
    private val settings: StreamSettings,
    private val onAccessUnit: (EncodedAccessUnit) -> Unit,
    private val onParameterSets: (H264ParameterSets) -> Unit,
    private val onOutputFormat: (H264OutputFormat) -> Unit,
    private val onError: (StreamFailure) -> Unit,
) {
    private val running = AtomicBoolean(false)
    private var codec: MediaCodec? = null
    private var inputSurface: Surface? = null
    private var outputThread: Thread? = null
    private var outputStopped: CountDownLatch? = null
    private var encodedFrameCount = 0L
    private var encodedByteCount = 0L
    private var firstOutputLogged = false

    @Synchronized
    fun start(): Surface {
        if (running.get()) {
            throw StreamFailureException(StreamFailure(StreamErrorKind.ENCODER, "H264 encoder already started"))
        }
        val requestedConfiguration = H264EncoderConfiguration.from(settings)
        encodedFrameCount = 0L
        encodedByteCount = 0L
        firstOutputLogged = false
        StreamErrorLogger.info("H.264 encoder configuration requested: ${requestedConfiguration.description()}")
        StreamErrorLogger.info(
            "Encoder resolution ${requestedConfiguration.width}x${requestedConfiguration.height}",
        )
        var encoder: MediaCodec? = null
        try {
            encoder = MediaCodec.createEncoderByType(requestedConfiguration.mimeType)
            val codecInstance = encoder
            StreamErrorLogger.info("H.264 codec selected: ${codecInstance.name}")
            val format = createFormat(requestedConfiguration)
            StreamErrorLogger.info("H.264 MediaFormat requested: $format")
            try {
                codecInstance.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
                StreamErrorLogger.info("H.264 MediaCodec.configure succeeded")
            } catch (error: IllegalArgumentException) {
                StreamErrorLogger.info(
                    "H.264 MediaCodec.configure rejected bitrate mode: " +
                        "${error.message ?: error.javaClass.simpleName}; " +
                        "retrying without bitrate mode; requested=${requestedConfiguration.description()} " +
                        "alternative=${requestedConfiguration.withoutBitrateMode().description()}",
                )
                try {
                    codecInstance.release()
                } catch (releaseError: Exception) {
                    StreamErrorLogger.cleanup(
                        StreamErrorFormatter.cleanupFailure("MediaCodec rejected-config release", releaseError),
                    )
                } finally {
                    encoder = null
                }
                val fallbackConfiguration = requestedConfiguration.withoutBitrateMode()
                val fallback = MediaCodec.createEncoderByType(fallbackConfiguration.mimeType)
                encoder = fallback
                StreamErrorLogger.info("H.264 fallback codec selected: ${fallback.name}")
                fallback.configure(
                    createFormat(fallbackConfiguration),
                    null,
                    null,
                    MediaCodec.CONFIGURE_FLAG_ENCODE,
                )
                StreamErrorLogger.info("H.264 MediaCodec.configure succeeded without bitrate mode")
            }
            val activeEncoder = encoder ?: error("MediaCodec instance unavailable")
            inputSurface = activeEncoder.createInputSurface()
            StreamErrorLogger.info("H.264 input Surface created")
            activeEncoder.start()
            StreamErrorLogger.info("H.264 encoder started")
            codec = activeEncoder
            running.set(true)
            try {
                val stopped = CountDownLatch(1)
                val output = Thread(
                    {
                        try {
                            drainOutput()
                        } finally {
                            stopped.countDown()
                        }
                    },
                    "h264-output",
                ).apply { isDaemon = true }
                outputThread = output
                outputStopped = stopped
                output.start()
                StreamErrorLogger.info("H.264 output drain thread started")
            } catch (error: Exception) {
                outputThread = null
                outputStopped = null
                throw StreamFailureException(StreamErrorFormatter.fromThrowable(StreamErrorKind.THREAD, error))
            }
            return inputSurface ?: error("MediaCodec input Surface unavailable")
        } catch (error: StreamFailureException) {
            val cleanup = cleanupResources(encoder)
            throw StreamFailureException(StreamErrorFormatter.withCleanup(error.failure, cleanup))
        } catch (error: Exception) {
            val cleanup = cleanupResources(encoder)
            throw StreamFailureException(
                StreamErrorFormatter.withCleanup(
                    StreamErrorFormatter.fromThrowable(StreamErrorKind.MEDIACODEC, error),
                    cleanup,
                ),
            )
        }
    }

    @Synchronized
    fun stop(): CleanupReport {
        val hadResources = running.get() || codec != null || inputSurface != null || outputThread != null
        if (!hadResources) return CleanupReport()
        StreamErrorLogger.info("H.264 encoder stopping")
        val report = cleanupResources(null)
        if (report.isSuccessful) {
            StreamErrorLogger.info(
                "H.264 encoder stopped: encodedFrames=$encodedFrameCount encodedBytes=$encodedByteCount",
            )
        }
        return report
    }

    @Synchronized
    fun requestSyncFrame() {
        val activeCodec = codec ?: return
        if (!running.get()) return
        try {
            activeCodec.setParameters(Bundle().apply {
                putInt(MediaCodec.PARAMETER_KEY_REQUEST_SYNC_FRAME, 0)
            })
            StreamErrorLogger.info("H.264 sync frame requested")
        } catch (error: Exception) {
            // The server already has a valid generation and will wait for the
            // next IDR. A vendor refusing this best-effort request must not
            // tear down the otherwise healthy Camera2 pipeline.
            StreamErrorLogger.info(
                "H.264 sync frame request unavailable: ${error.message ?: error.javaClass.simpleName}",
            )
        }
    }

    private fun cleanupResources(unassignedEncoder: MediaCodec?): CleanupReport {
        val cleanup = CleanupCollector()
        running.set(false)
        val thread = outputThread.also { outputThread = null }
        val stopped = outputStopped.also { outputStopped = null }
        cleanup.runUnit("encoder thread interrupt") { thread?.interrupt() }

        // Stop the codec before waiting for the drain loop. dequeueOutputBuffer
        // is a native call and may not unblock from Thread.interrupt() alone;
        // stopping the codec is the lifecycle signal that releases it safely.
        val encoder = codec
        cleanup.runUnit("MediaCodec stop") { encoder?.stop() }

        if (thread != null && Thread.currentThread() !== thread) {
            cleanup.runUnit("encoder thread join") {
                val completed = stopped?.await(1_000, TimeUnit.MILLISECONDS) ?: run {
                    thread.join(1_000)
                    true
                }
                if (!completed) throw IllegalStateException("encoder thread did not stop")
            }
            if (thread.isAlive) {
                cleanup.add(
                    com.localsecuritycam.android.diagnostics.StreamErrorFormatter.cleanupFailure(
                        "encoder thread join",
                        IllegalStateException("encoder thread did not stop"),
                    ),
                )
            }
        }
        codec = null
        cleanup.runUnit("MediaCodec release") { encoder?.release() }
        if (encoder == null) cleanup.runUnit("MediaCodec release") { unassignedEncoder?.release() }
        val surface = inputSurface.also { inputSurface = null }
        cleanup.runUnit("encoder Surface") { surface?.release() }
        return cleanup.report()
    }

    private fun drainOutput() {
        val info = MediaCodec.BufferInfo()
        while (running.get()) {
            val encoder = codec ?: break
            try {
                when (val index = encoder.dequeueOutputBuffer(info, 10_000)) {
                    MediaCodec.INFO_TRY_AGAIN_LATER -> Unit
                    MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> publishFormat(encoder.outputFormat)
                    MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED -> Unit
                    else -> if (index >= 0) {
                        withH264OutputBuffer(
                            process = { processOutputBuffer(encoder, index, info) },
                            release = { encoder.releaseOutputBuffer(index, false) },
                            onReleaseError = { error ->
                                StreamErrorLogger.error(StreamErrorFormatter.fromThrowable(StreamErrorKind.ENCODER, error))
                            },
                        )
                    }
                }
            } catch (error: Exception) {
                val failure = StreamErrorFormatter.fromThrowable(StreamErrorKind.ENCODER, error)
                if (running.get()) reportFailure(failure)
                else StreamErrorLogger.info("H.264 drain stopped during cleanup: ${failure.detail}")
                break
            }
        }
    }

    private fun processOutputBuffer(encoder: MediaCodec, index: Int, info: MediaCodec.BufferInfo) {
        val buffer = encoder.getOutputBuffer(index)
        if (buffer == null || info.size <= 0) return
        val bytes = readBytes(buffer, info.offset, info.size)
        val result = H264OutputProcessor.process(
            H264OutputSample(
                bytes = bytes,
                ptsUs = info.presentationTimeUs,
                codecConfig = info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0,
                keyFrame = info.flags and MediaCodec.BUFFER_FLAG_KEY_FRAME != 0,
            ),
        )
        result.parameterSets?.let(::publishParameterSets)
        val accessUnit = result.accessUnit ?: return
        encodedFrameCount++
        encodedByteCount += accessUnit.byteCount.toLong()
        if (!firstOutputLogged) {
            firstOutputLogged = true
            StreamErrorLogger.info(
                "H.264 first encoded access unit: bytes=${accessUnit.byteCount} ptsUs=${accessUnit.ptsUs}",
            )
        }
        onAccessUnit(accessUnit)
    }

    @Suppress("InlinedApi")
    private fun publishFormat(format: MediaFormat) {
        StreamErrorLogger.info("H.264 output format changed: $format")
        fun integer(key: String): Int? = runCatching { format.getInteger(key) }.getOrNull()
        val width = integer(MediaFormat.KEY_WIDTH)
        val height = integer(MediaFormat.KEY_HEIGHT)
        if (width == null || height == null || width <= 0 || height <= 0) {
            reportFailure(
                StreamFailure(StreamErrorKind.MEDIACODEC, "MediaCodec output format omitted valid dimensions"),
            )
            return
        }
        val requested = H264EncoderConfiguration.from(settings)
        val outputFormat = H264OutputFormat(
            width = width,
            height = height,
            cropLeft = integer(MediaFormat.KEY_CROP_LEFT) ?: 0,
            cropTop = integer(MediaFormat.KEY_CROP_TOP) ?: 0,
            cropRight = integer(MediaFormat.KEY_CROP_RIGHT) ?: width - 1,
            cropBottom = integer(MediaFormat.KEY_CROP_BOTTOM) ?: height - 1,
            stride = integer(MediaFormat.KEY_STRIDE),
            sliceHeight = integer(MediaFormat.KEY_SLICE_HEIGHT),
            pixelAspectRatioWidth = integer(KEY_PIXEL_ASPECT_RATIO_WIDTH) ?: 1,
            pixelAspectRatioHeight = integer(KEY_PIXEL_ASPECT_RATIO_HEIGHT) ?: 1,
        )
        StreamErrorLogger.info(
            "H.264 output geometry raw=${outputFormat.width}x${outputFormat.height} " +
                "visible=${outputFormat.visibleWidth}x${outputFormat.visibleHeight} " +
                "crop=${outputFormat.cropLeft},${outputFormat.cropTop}-" +
                "${outputFormat.cropRight},${outputFormat.cropBottom} " +
                "stride=${outputFormat.stride ?: "n/a"} slice_height=${outputFormat.sliceHeight ?: "n/a"} " +
                "pixel_aspect=${outputFormat.pixelAspectRatioWidth}:${outputFormat.pixelAspectRatioHeight}",
        )
        val geometryError = outputFormat.validationError(requested.width, requested.height)
        if (geometryError != null || !requested.matches(outputFormat)) {
            reportFailure(
                StreamFailure(
                    StreamErrorKind.MEDIACODEC,
                    "MediaCodec output geometry mismatch requested=${requested.width}x${requested.height} " +
                        "actual=${outputFormat.width}x${outputFormat.height} " +
                        "detail=${geometryError ?: "visible dimensions do not match"}",
                    retryable = false,
                ),
            )
            return
        }
        StreamErrorLogger.info("Encoder resolution ${width}x${height}")
        val sps = format.getByteBuffer("csd-0")?.let(::copyBuffer)
        val pps = format.getByteBuffer("csd-1")?.let(::copyBuffer)
        H264OutputProcessor.parameterSetsFromFormat(sps, pps)?.let(::publishParameterSets)
        // Make parameter sets available to the broadcaster before opening the
        // new RTSP generation and exposing its SDP to a client.
        onOutputFormat(outputFormat)
    }

    private fun publishParameterSets(parameterSets: H264ParameterSets) {
        StreamErrorLogger.info(
            "H.264 SPS/PPS received: spsBytes=${parameterSets.sps.size} ppsBytes=${parameterSets.pps.size}",
        )
        onParameterSets(parameterSets)
    }

    private fun reportFailure(failure: StreamFailure) {
        try {
            onError(failure)
        } catch (callbackError: Exception) {
            StreamErrorLogger.observer(callbackError)
        }
    }

    private fun createFormat(configuration: H264EncoderConfiguration): MediaFormat = MediaFormat.createVideoFormat(
        configuration.mimeType,
        configuration.width,
        configuration.height,
    ).apply {
        check(configuration.surfaceInput) { "H.264 encoder requires Surface input" }
        setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
        setInteger(MediaFormat.KEY_BIT_RATE, configuration.bitrate)
        setInteger(MediaFormat.KEY_FRAME_RATE, configuration.fps)
        setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, configuration.keyframeIntervalSeconds)
        if (configuration.bitrateMode == H264BitrateMode.CBR) {
            setInteger(MediaFormat.KEY_BITRATE_MODE, MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_CBR)
        }
    }

    private fun copyBuffer(source: ByteBuffer): ByteArray {
        val duplicate = source.duplicate()
        val output = ByteArray(duplicate.remaining())
        duplicate.get(output)
        return output
    }

    private fun readBytes(source: ByteBuffer, offset: Int, size: Int): ByteArray {
        val duplicate = source.duplicate()
        val start = offset.coerceIn(0, duplicate.limit())
        val end = (offset.toLong() + size.toLong())
            .coerceAtMost(duplicate.limit().toLong())
            .toInt()
        if (end <= start) return ByteArray(0)
        duplicate.position(start)
        duplicate.limit(end)
        val output = ByteArray(duplicate.remaining())
        duplicate.get(output)
        return output
    }
}
