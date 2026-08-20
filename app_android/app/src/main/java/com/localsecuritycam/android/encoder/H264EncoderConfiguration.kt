package com.localsecuritycam.android.encoder

import com.localsecuritycam.android.settings.StreamSettings

internal data class H264OutputFormat(
    val width: Int,
    val height: Int,
    val cropLeft: Int = 0,
    val cropTop: Int = 0,
    val cropRight: Int = width - 1,
    val cropBottom: Int = height - 1,
    val stride: Int? = null,
    val sliceHeight: Int? = null,
    val pixelAspectRatioWidth: Int = 1,
    val pixelAspectRatioHeight: Int = 1,
)

{
    val visibleWidth: Int
        get() = cropRight - cropLeft + 1

    val visibleHeight: Int
        get() = cropBottom - cropTop + 1

    /**
     * The GL pass is the only place allowed to transform pixels. A codec
     * crop/pixel-aspect mismatch is therefore a hard generation failure.
     */
    fun validationError(requestedWidth: Int, requestedHeight: Int): String? {
        if (width != requestedWidth || height != requestedHeight) {
            return "raw dimensions ${width}x$height do not match requested ${requestedWidth}x$requestedHeight"
        }
        if (cropLeft != 0 || cropTop != 0 || cropRight != width - 1 || cropBottom != height - 1) {
            return "crop ${cropLeft},${cropTop}-${cropRight},${cropBottom} is not the full requested frame"
        }
        if (visibleWidth != requestedWidth || visibleHeight != requestedHeight) {
            return "visible dimensions ${visibleWidth}x$visibleHeight do not match requested ${requestedWidth}x$requestedHeight"
        }
        if (stride != null && stride < visibleWidth) return "stride $stride is smaller than visible width $visibleWidth"
        if (sliceHeight != null && sliceHeight < visibleHeight) {
            return "slice height $sliceHeight is smaller than visible height $visibleHeight"
        }
        if (pixelAspectRatioWidth <= 0 || pixelAspectRatioHeight <= 0) {
            return "pixel aspect ratio is invalid ${pixelAspectRatioWidth}:$pixelAspectRatioHeight"
        }
        if (pixelAspectRatioWidth != pixelAspectRatioHeight) {
            return "non-square pixel aspect ratio ${pixelAspectRatioWidth}:$pixelAspectRatioHeight is unsupported"
        }
        return null
    }
}

internal fun H264EncoderConfiguration.matches(format: H264OutputFormat): Boolean =
    width == format.visibleWidth && height == format.visibleHeight

internal enum class H264BitrateMode {
    CBR,
}

/**
 * Android-free description of the requested AVC encoder configuration.
 * MediaFormat creation remains in H264Encoder; this value is the diagnostic
 * and testable contract between settings and MediaCodec configuration.
 */
internal data class H264EncoderConfiguration(
    val mimeType: String,
    val width: Int,
    val height: Int,
    val fps: Int,
    val bitrate: Int,
    val keyframeIntervalSeconds: Int,
    val surfaceInput: Boolean,
    val bitrateMode: H264BitrateMode?,
) {
    fun withoutBitrateMode(): H264EncoderConfiguration = copy(bitrateMode = null)

    fun description(): String = buildString {
        append(mimeType)
            .append(' ')
            .append(width)
            .append('x')
            .append(height)
            .append('@')
            .append(fps)
            .append("fps bitrate=")
            .append(bitrate)
            .append("bps keyframe=")
            .append(keyframeIntervalSeconds)
            .append("s input=")
            .append(if (surfaceInput) "Surface" else "unsupported")
        bitrateMode?.let { append(" bitrateMode=").append(it.name) }
    }

    companion object {
        fun from(settings: StreamSettings, bitrateMode: H264BitrateMode? = H264BitrateMode.CBR) =
            H264EncoderConfiguration(
                mimeType = "video/avc",
                width = settings.resolution.width,
                height = settings.resolution.height,
                fps = settings.fps,
                bitrate = settings.bitrate,
                keyframeIntervalSeconds = settings.keyframeIntervalSeconds,
                surfaceInput = true,
                bitrateMode = bitrateMode,
            )
    }
}
