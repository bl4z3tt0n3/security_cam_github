package com.localsecuritycam.android.diagnostics

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PreviewPixelMetricsTest {
    @Test
    fun blackPixelsProduceZeroLumaAndNoNonBlackContent() {
        val metrics = PreviewPixelAnalyzer.analyze(
            width = 2,
            height = 2,
            pixels = intArrayOf(0xff000000.toInt(), 0xff000000.toInt(), 0xff000000.toInt(), 0xff000000.toInt()),
        )

        assertEquals(4, metrics.sampledPixels)
        assertEquals(0.0, metrics.meanLuma, 0.0)
        assertEquals(0.0, metrics.nonBlackRatio, 0.0)
        assertNull(metrics.changedFromPrevious)
    }

    @Test
    fun changingPixelsAreDetectedByTheFrameHash() {
        val first = PreviewPixelAnalyzer.analyze(
            width = 2,
            height = 1,
            pixels = intArrayOf(0xffff0000.toInt(), 0xff00ff00.toInt()),
        )
        val second = PreviewPixelAnalyzer.analyze(
            width = 2,
            height = 1,
            pixels = intArrayOf(0xff0000ff.toInt(), 0xffffff00.toInt()),
            previousHash = first.frameHash,
        )

        assertTrue(first.nonBlackRatio > 0.99)
        assertTrue(first.meanLuma > 0.0)
        assertFalse(first.frameHash == second.frameHash)
        assertTrue(second.changedFromPrevious == true)
    }

    @Test(expected = IllegalArgumentException::class)
    fun mismatchedDimensionsAreRejected() {
        PreviewPixelAnalyzer.analyze(width = 2, height = 2, pixels = intArrayOf(0))
    }
}
