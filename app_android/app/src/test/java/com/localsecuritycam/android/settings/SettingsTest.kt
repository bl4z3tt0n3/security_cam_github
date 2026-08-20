package com.localsecuritycam.android.settings

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SettingsTest {
    @Test
    fun normalizesPathAndRedactsPasswordInDisplayedUrl() {
        val settings = StreamSettings(streamPath = "/live/")
        assertEquals("/live", settings.normalizedPath)
        assertEquals("rtsp://admin:***@192.168.1.20:8554/live", StreamUrlBuilder.sanitizedUrl(settings.copy(authEnabled = true, username = "admin"), "192.168.1.20"))
    }

    @Test
    fun rejectsInvalidPortAndMissingAuthenticatedPassword() {
        val settings = StreamSettings(port = 80, authEnabled = true)
        val errors = settings.validate(null)
        assertTrue(errors.any { it.contains("port") })
        assertTrue(errors.any { it.contains("password") })
    }

    @Test
    fun rejectsWeakAuthenticatedPassword() {
        val errors = StreamSettings(authEnabled = true, username = "camera").validate("short")
        assertTrue(errors.any { it.contains("8 characters") })
    }

    @Test
    fun rejectsPathTraversalAndUnsafeBasicAuthUsername() {
        val settings = StreamSettings(
            streamPath = "live/../private",
            authEnabled = true,
            username = "camera@example",
        )
        val errors = settings.validate("local-password")
        assertTrue(errors.any { it.contains("stream path") })
        assertTrue(errors.any { it.contains("username") })
    }

    @Test
    fun acceptsDefaultSettings() {
        assertTrue(StreamSettings(authEnabled = false).validate().isEmpty())
    }

    @Test
    fun enablesAuthenticationByDefaultForRealListeners() {
        assertTrue(StreamSettings().authEnabled)
    }

    @Test
    fun displayLineIdentifiesSelectedLens() {
        assertTrue(StreamSettings().displayVideoLine().contains("posteriore"))
        assertTrue(StreamSettings(lens = CameraLens.FRONT).displayVideoLine().contains("anteriore"))
    }

    @Test
    fun exposesBothSelectableOutputAspectRatios() {
        assertEquals("16:9", VideoAspectRatio.LANDSCAPE_16_9.label)
        assertEquals("9:16", VideoAspectRatio.PORTRAIT_9_16.label)
        assertEquals(VideoAspectRatio.LANDSCAPE_16_9, StreamSettings().aspectRatio)
        assertTrue(StreamSettings(aspectRatio = VideoAspectRatio.PORTRAIT_9_16).displayVideoLine().contains("9:16"))
    }
}
