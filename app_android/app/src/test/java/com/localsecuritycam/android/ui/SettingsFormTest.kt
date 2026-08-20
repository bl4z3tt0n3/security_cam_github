package com.localsecuritycam.android.ui

import com.localsecuritycam.android.camera.CameraCapabilities
import com.localsecuritycam.android.settings.CameraLens
import com.localsecuritycam.android.settings.Resolution
import com.localsecuritycam.android.settings.StreamSettings
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SettingsFormTest {
    @Test
    fun emptyPasswordKeepsTheConfiguredCredentialOutOfTheFormState() {
        val resolution = Resolution(640, 360)
        val stream = StreamSettings(
            cameraName = "Huawei",
            lens = CameraLens.BACK,
            resolution = resolution,
            fps = 15,
            bitrate = 700_000,
            keyframeIntervalSeconds = 2,
            username = "camera",
            authEnabled = true,
        )
        val result = SettingsFormState.from(stream).toValidatedSettings(
            capabilities = capabilities(resolution),
            existingPassword = "local-pass-123",
            typedPassword = "",
        )

        assertTrue(result.errors.isEmpty())
        assertEquals("local-pass-123", result.settings?.password)
    }

    @Test
    fun unsupportedResolutionAndFpsAreRejectedBeforeSave() {
        val supported = Resolution(640, 360)
        val form = SettingsFormState.from(
            StreamSettings(
                resolution = Resolution(1920, 1080),
                fps = 30,
                authEnabled = false,
            ),
        )
        val result = form.toValidatedSettings(
            capabilities = capabilities(supported),
            existingPassword = null,
            typedPassword = "",
        )

        assertTrue(result.settings == null)
        assertTrue(result.errors.any { it.contains("resolution/FPS") })
    }

    private fun capabilities(resolution: Resolution) = CameraCapabilities(
        cameraId = "0",
        lens = CameraLens.BACK,
        resolutions = listOf(resolution),
        fpsValues = listOf(15),
        fpsByResolution = mapOf(resolution to listOf(15)),
        minBitrate = 100_000,
        maxBitrate = 50_000_000,
    )
}

