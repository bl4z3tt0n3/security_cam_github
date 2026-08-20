package com.localsecuritycam.android.diagnostics

import java.util.Locale

/** Pixel-level facts captured from the actual preview SurfaceView. */
data class PreviewPixelMetrics(
    val width: Int,
    val height: Int,
    val sampledPixels: Int,
    val meanLuma: Double,
    val nonBlackRatio: Double,
    val frameHash: Long,
    val changedFromPrevious: Boolean?,
)

/** Pure bitmap-independent analyzer so the thresholds can be unit tested. */
object PreviewPixelAnalyzer {
    private const val NON_BLACK_THRESHOLD = 8
    private const val FNV_OFFSET = -0x340d631b8c467dL
    private const val FNV_PRIME = 0x100000001b3L

    fun analyze(
        width: Int,
        height: Int,
        pixels: IntArray,
        previousHash: Long? = null,
    ): PreviewPixelMetrics {
        require(width > 0 && height > 0) { "pixel dimensions must be positive" }
        require(pixels.size == width * height) {
            "pixel count ${pixels.size} does not match ${width}x${height}"
        }
        var lumaSum = 0.0
        var nonBlack = 0
        var hash = FNV_OFFSET
        pixels.forEach { pixel ->
            val red = pixel ushr 16 and 0xff
            val green = pixel ushr 8 and 0xff
            val blue = pixel and 0xff
            lumaSum += 0.2126 * red + 0.7152 * green + 0.0722 * blue
            if (maxOf(red, green, blue) > NON_BLACK_THRESHOLD) nonBlack++
            hash = (hash xor (pixel.toLong() and 0xffffffffL)) * FNV_PRIME
        }
        return PreviewPixelMetrics(
            width = width,
            height = height,
            sampledPixels = pixels.size,
            meanLuma = lumaSum / pixels.size.toDouble(),
            nonBlackRatio = nonBlack.toDouble() / pixels.size.toDouble(),
            frameHash = hash,
            changedFromPrevious = previousHash?.let { it != hash },
        )
    }

    fun toLogLine(metrics: PreviewPixelMetrics): String =
        "PREVIEW_PIXEL_METRICS width=${metrics.width} height=${metrics.height} " +
            "sampled=${metrics.sampledPixels} mean_luma=${"%.3f".format(Locale.US, metrics.meanLuma)} " +
            "non_black_ratio=${"%.5f".format(Locale.US, metrics.nonBlackRatio)} " +
            "frame_hash=${metrics.frameHash} changed=${metrics.changedFromPrevious}"
}
