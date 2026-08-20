package com.localsecuritycam.android.streaming

import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.Base64

data class RtspCredentials(
    val enabled: Boolean,
    val username: String,
    val password: String,
) {
    override fun toString(): String =
        "RtspCredentials(enabled=$enabled, username=$username, passwordConfigured=${password.isNotEmpty()})"
}

object RtspAuth {
    fun isAuthorized(header: String?, credentials: RtspCredentials?): Boolean {
        if (credentials == null || !credentials.enabled) return true
        val value = header?.trim() ?: return false
        if (!value.startsWith("Basic ", ignoreCase = true)) return false
        val encoded = value.substringAfter(' ', "").trim()
        val decoded = runCatching { String(Base64.getDecoder().decode(encoded), StandardCharsets.UTF_8) }.getOrNull()
            ?: return false
        val expected = "${credentials.username}:${credentials.password}"
        return MessageDigest.isEqual(
            decoded.toByteArray(StandardCharsets.UTF_8),
            expected.toByteArray(StandardCharsets.UTF_8),
        )
    }

    fun challenge(): String = "Basic realm=\"Huawei LAN Camera\""
}
