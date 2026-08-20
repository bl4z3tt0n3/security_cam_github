package com.localsecuritycam.android.camera

import com.localsecuritycam.android.settings.CameraLens
import com.localsecuritycam.android.settings.Resolution
import com.localsecuritycam.android.settings.StreamSettings
import org.junit.Assert.assertTrue
import org.junit.Assert.assertFalse
import org.junit.Test

class CameraCapabilitiesTest {
    private val resolution = Resolution(1280, 720)
    private val capabilities = CameraCapabilities(
        cameraId = "0",
        lens = CameraLens.BACK,
        resolutions = listOf(resolution),
        fpsValues = listOf(20, 30),
        fpsByResolution = mapOf(resolution to listOf(20, 30)),
        minBitrate = 500_000,
        maxBitrate = 5_000_000,
    )

    @Test
    fun acceptsOnlyTheResolutionAndFpsPairsReportedByTheEncoder() {
        assertTrue(
            CameraCapabilitiesProvider.validationErrors(
                capabilities,
                StreamSettings(resolution = resolution, fps = 30, bitrate = 2_000_000),
            ).isEmpty(),
        )
        assertTrue(
            CameraCapabilitiesProvider.validationErrors(
                capabilities,
                StreamSettings(resolution = Resolution(1920, 1080), fps = 30, bitrate = 2_000_000),
            ).any { it.contains("resolution/FPS") },
        )
    }

    @Test
    fun rejectsBitrateOutsideTheDetectedEncoderRange() {
        val errors = CameraCapabilitiesProvider.validationErrors(
            capabilities,
            StreamSettings(resolution = resolution, fps = 20, bitrate = 5_000_001),
        )
        assertTrue(errors.any { it.contains("bitrate") })
    }

    @Test
    fun requiresAvcSupportForBothLandscapeAndPortraitOutputGeometries() {
        assertTrue(
            CameraCapabilitiesProvider.supportsBothOutputGeometries(resolution, 20) { width, height, _ ->
                (width == 1280 && height == 720) || (width == 720 && height == 1280)
            },
        )
        assertFalse(
            CameraCapabilitiesProvider.supportsBothOutputGeometries(resolution, 20) { width, height, _ ->
                width == 1280 && height == 720
            },
        )
    }
}
