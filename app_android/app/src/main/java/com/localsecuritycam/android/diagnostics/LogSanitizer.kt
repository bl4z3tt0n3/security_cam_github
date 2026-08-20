package com.localsecuritycam.android.diagnostics

object LogSanitizer {
    private val userInfoPattern = Regex("(?i)(rtsp://[^/?#:@\\s]+):[^/?#@\\s]*@")
    private val basicAuthPattern = Regex("(?i)(\\bAuthorization\\s*:\\s*Basic\\s+)[^\\s]+")
    private val secretPattern = Regex("(?i)(\\b(?:password|passwd|pwd|secret|token|api[_-]?key)\\b\\s*[:=]\\s*)[^\\s,;]+")

    fun url(value: String): String = value
        .replace(userInfoPattern, "$1:***@")
        .replace(Regex("(?i)([?&](?:password|passwd|pwd|secret|token)=)[^&\\s]+"), "$1***")

    fun text(value: Any?): String = secretPattern.replace(
        basicAuthPattern.replace(url(value?.toString().orEmpty()).replace('\n', ' '), "$1***"),
    ) { "${it.groupValues[1]}***" }

    fun error(error: Throwable): String = text(error.message ?: error::class.java.simpleName)
}
