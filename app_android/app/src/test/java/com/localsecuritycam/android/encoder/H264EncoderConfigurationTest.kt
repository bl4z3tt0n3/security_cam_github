package com.localsecuritycam.android.encoder

import com.localsecuritycam.android.settings.Resolution
import com.localsecuritycam.android.settings.StreamSettings
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class H264EncoderConfigurationTest {
    @Test
    fun mapsTheRequested1080pConfigurationWithoutChangingValues() {
        val configuration = H264EncoderConfiguration.from(
            StreamSettings(
                resolution = Resolution(1920, 1080),
                fps = 20,
                bitrate = 2_000_000,
                authEnabled = false,
            ),
        )

        assertEquals("video/avc", configuration.mimeType)
        assertEquals(1920, configuration.width)
        assertEquals(1080, configuration.height)
        assertEquals(20, configuration.fps)
        assertEquals(2_000_000, configuration.bitrate)
        assertEquals(2, configuration.keyframeIntervalSeconds)
        assertTrue(configuration.surfaceInput)
        assertEquals(H264BitrateMode.CBR, configuration.bitrateMode)
    }

    @Test
    fun fallbackOnlyRemovesBitrateMode() {
        val requested = H264EncoderConfiguration.from(
            StreamSettings(
                resolution = Resolution(1920, 1080),
                fps = 20,
                bitrate = 2_000_000,
                authEnabled = false,
            ),
        )

        val fallback = requested.withoutBitrateMode()

        assertEquals(requested.copy(bitrateMode = null), fallback)
        assertEquals("video/avc 1920x1080@20fps bitrate=2000000bps keyframe=2s input=Surface", fallback.description())
    }

    @Test
    fun acceptsPortraitOutputDimensionsWithoutSwappingThemBack() {
        val configuration = H264EncoderConfiguration.from(
            StreamSettings(
                resolution = Resolution(720, 1280),
                authEnabled = false,
            ),
        )

        assertEquals(720, configuration.width)
        assertEquals(1280, configuration.height)
    }

    @Test
    fun outputFormatMustMatchTheRequestedMediaCodecGeometry() {
        val configuration = H264EncoderConfiguration.from(
            StreamSettings(resolution = Resolution(720, 1280), authEnabled = false),
        )

        assertTrue(configuration.matches(H264OutputFormat(720, 1280)))
        assertTrue(!configuration.matches(H264OutputFormat(1280, 720)))
    }

    @Test
    fun rejectsCodecCropEvenWhenVisibleDimensionsWouldAppearCorrect() {
        val format = H264OutputFormat(
            width = 1920,
            height = 1088,
            cropBottom = 1079,
        )

        assertEquals(
            "raw dimensions 1920x1088 do not match requested 1920x1080",
            format.validationError(1920, 1080),
        )
    }

    @Test
    fun rejectsNonSquarePixelAspectAndInvalidStride() {
        val nonSquare = H264OutputFormat(
            width = 1280,
            height = 720,
            pixelAspectRatioWidth = 4,
            pixelAspectRatioHeight = 3,
        )
        val invalidStride = H264OutputFormat(width = 1280, height = 720, stride = 1200)

        assertTrue(nonSquare.validationError(1280, 720)!!.contains("non-square pixel aspect ratio"))
        assertTrue(invalidStride.validationError(1280, 720)!!.contains("stride"))
    }
}
