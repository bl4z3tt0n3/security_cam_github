package com.localsecuritycam.android.settings

import java.util.Locale

enum class CameraLens(val wireValue: String) {
    BACK("back"),
    FRONT("front"),
}

enum class VideoAspectRatio(
    val label: String,
    val isPortrait: Boolean,
) {
    LANDSCAPE_16_9("16:9", isPortrait = false),
    PORTRAIT_9_16("9:16", isPortrait = true),
}

data class Resolution(val width: Int, val height: Int) {
    init {
        require(width > 0 && height > 0) { "resolution must be positive" }
    }

    override fun toString(): String = "${width}x$height"
}

data class StreamSettings(
    val cameraName: String = "Camera 1",
    val lens: CameraLens = CameraLens.BACK,
    val resolution: Resolution = Resolution(1280, 720),
    val aspectRatio: VideoAspectRatio = VideoAspectRatio.LANDSCAPE_16_9,
    val fps: Int = 20,
    val bitrate: Int = 2_000_000,
    val keyframeIntervalSeconds: Int = 2,
    val port: Int = 8554,
    val streamPath: String = "stream",
    /** Basic authentication is enabled by default; disabling it explicitly exposes the RTSP listener openly. */
    val authEnabled: Boolean = true,
    val username: String = "",
    val autoStart: Boolean = false,
    val keepScreenAwake: Boolean = false,
    /** An active local stream resumes automatically when its Wi-Fi returns. */
    val autoRestartOnNetwork: Boolean = true,
) {
    val normalizedPath: String
        get() = "/" + streamPath.trim().trim('/').ifBlank { "stream" }

    fun validate(password: String? = null): List<String> {
        val errors = mutableListOf<String>()
        if (cameraName.trim().isEmpty()) errors += "camera name cannot be empty"
        if (resolution.width < 160 || resolution.height < 120) errors += "resolution is too small"
        if (fps !in 1..60) errors += "FPS must be between 1 and 60"
        if (bitrate !in 100_000..50_000_000) errors += "bitrate must be between 100 kbps and 50 Mbps"
        if (keyframeIntervalSeconds !in 1..10) errors += "keyframe interval must be between 1 and 10 seconds"
        if (port !in 1024..65535) errors += "RTSP port must be between 1024 and 65535"
        if (streamPath.trim().trim('/').isEmpty()) errors += "stream path cannot be empty"
        val pathSegments = streamPath.trim().trim('/').split('/')
        if (streamPath.any { it.code < 0x20 || it.code == 0x7f || it.isWhitespace() || it == '?' || it == '#' || it == '\\' } ||
            pathSegments.any { it == "." || it == ".." || it.isEmpty() }
        ) {
            errors += "stream path contains unsupported characters"
        }
        if (authEnabled) {
            if (username.trim().isEmpty()) errors += "username is required when authentication is enabled"
            if (username.any { it.code < 0x20 || it.code == 0x7f || it.isWhitespace() || it in ":/@?#%\\" }) {
                errors += "username contains unsupported URL characters"
            }
            if (password.isNullOrEmpty()) errors += "password is required when authentication is enabled"
            if (!password.isNullOrEmpty() && password.length < 8) {
                errors += "password must contain at least 8 characters"
            }
        }
        return errors
    }

    fun requireValid(password: String? = null): StreamSettings {
        val errors = validate(password)
        require(errors.isEmpty()) { errors.joinToString("; ") }
        return this
    }

    fun displayVideoLine(): String {
        val lensLabel = if (lens == CameraLens.BACK) "posteriore" else "anteriore"
        return String.format(
            Locale.US,
            "%s • %dx%d • %d FPS • H.264 • %d kbps",
            lensLabel,
            resolution.width,
            resolution.height,
            fps,
            bitrate / 1000,
        ).let { base -> "$base • ${aspectRatio.label}" }
    }
}

data class AppSettings(
    val stream: StreamSettings = StreamSettings(),
    val password: String? = null,
) {
    override fun toString(): String =
        "AppSettings(stream=$stream, passwordConfigured=${!password.isNullOrEmpty()})"
}

enum class StreamPreset(val label: String, val settings: StreamSettings) {
    LOW("LOW", StreamSettings(resolution = Resolution(640, 360), fps = 15, bitrate = 700_000)),
    MEDIUM("MEDIUM", StreamSettings(resolution = Resolution(1280, 720), fps = 20, bitrate = 2_000_000)),
    HIGH("HIGH", StreamSettings(resolution = Resolution(1920, 1080), fps = 30, bitrate = 5_000_000)),
}

object StreamUrlBuilder {
    fun sanitizedUrl(settings: StreamSettings, ipAddress: String?, includeUsername: Boolean = true): String? {
        val ip = ipAddress?.trim().orEmpty()
        if (ip.isEmpty()) return null
        val userInfo = if (settings.authEnabled && includeUsername && settings.username.isNotBlank()) {
            "${settings.username}:***@"
        } else {
            ""
        }
        return "rtsp://$userInfo$ip:${settings.port}${settings.normalizedPath}"
    }
}
