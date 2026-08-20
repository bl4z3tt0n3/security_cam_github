package com.localsecuritycam.android.diagnostics

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class LogSanitizerTest {
    @Test
    fun redactsUserInfoAndQuerySecrets() {
        val safe = LogSanitizer.text("open rtsp://admin:secret@camera.local/live?password=query-secret")
        assertTrue(safe.contains("admin:***@"))
        assertTrue(safe.contains("password=***"))
        assertFalse(safe.contains("secret"))
        assertFalse(safe.contains("query-secret"))
    }

    @Test
    fun redactsBasicAuthorizationHeaders() {
        val safe = LogSanitizer.text("Authorization: Basic YWRtaW46dmVyeS1zZWNyZXQ=")
        assertTrue(safe.contains("Authorization: Basic ***"))
        assertFalse(safe.contains("YWRtaW46"))
    }

    @Test
    fun redactsEncodedReservedPasswordCharactersInUrlUserInfo() {
        val safe = LogSanitizer.text("open rtsp://admin:pa%2Fss@camera.local/live")
        assertTrue(safe.contains("admin:***@"))
        assertFalse(safe.contains("pa%2Fss"))
    }
}
