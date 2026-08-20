package com.localsecuritycam.android.settings

import android.content.Context

class SettingsRepository(context: Context) {
    private val preferences = context.applicationContext.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
    private val credentialStore = CredentialStore(context)

    fun load(): AppSettings {
        val lens = runCatching { CameraLens.valueOf(preferences.getString(KEY_LENS, CameraLens.BACK.name)!!) }
            .getOrDefault(CameraLens.BACK)
        // Older builds persisted a manual override. Orientation is now owned
        // by the foreground service, so this preference is deliberately ignored.
        preferences.edit().remove(KEY_ROTATION).apply()
        val resolution = runCatching {
            Resolution(
                preferences.getInt(KEY_WIDTH, 1280),
                preferences.getInt(KEY_HEIGHT, 720),
            )
        }.getOrDefault(Resolution(1280, 720))
        val aspectRatio = runCatching {
            VideoAspectRatio.valueOf(
                preferences.getString(KEY_ASPECT_RATIO, VideoAspectRatio.LANDSCAPE_16_9.name)
                    ?: VideoAspectRatio.LANDSCAPE_16_9.name,
            )
        }.getOrDefault(VideoAspectRatio.LANDSCAPE_16_9)
        val stream = StreamSettings(
            cameraName = preferences.getString(KEY_NAME, "Camera 1") ?: "Camera 1",
            lens = lens,
            resolution = resolution,
            aspectRatio = aspectRatio,
            fps = preferences.getInt(KEY_FPS, 20),
            bitrate = preferences.getInt(KEY_BITRATE, 2_000_000),
            keyframeIntervalSeconds = preferences.getInt(KEY_KEYFRAME, 2),
            port = preferences.getInt(KEY_PORT, 8554),
            streamPath = preferences.getString(KEY_PATH, "stream") ?: "stream",
            authEnabled = preferences.getBoolean(KEY_AUTH, true),
            username = preferences.getString(KEY_USERNAME, "") ?: "",
            autoStart = preferences.getBoolean(KEY_AUTOSTART, false),
            keepScreenAwake = preferences.getBoolean(KEY_KEEP_AWAKE, false),
            autoRestartOnNetwork = true,
        )
        return AppSettings(stream = stream, password = credentialStore.loadPassword())
    }

    fun save(settings: AppSettings) {
        settings.stream.requireValid(if (settings.stream.authEnabled) settings.password else null)
        preferences.edit()
            .putString(KEY_NAME, settings.stream.cameraName.trim())
            .putString(KEY_LENS, settings.stream.lens.name)
            .putInt(KEY_WIDTH, settings.stream.resolution.width)
            .putInt(KEY_HEIGHT, settings.stream.resolution.height)
            .putString(KEY_ASPECT_RATIO, settings.stream.aspectRatio.name)
            .putInt(KEY_FPS, settings.stream.fps)
            .putInt(KEY_BITRATE, settings.stream.bitrate)
            .putInt(KEY_KEYFRAME, settings.stream.keyframeIntervalSeconds)
            .putInt(KEY_PORT, settings.stream.port)
            .putString(KEY_PATH, settings.stream.streamPath.trim().trim('/'))
            .putBoolean(KEY_AUTH, settings.stream.authEnabled)
            .putString(KEY_USERNAME, settings.stream.username.trim())
            .remove(KEY_ROTATION)
            .putBoolean(KEY_AUTOSTART, settings.stream.autoStart)
            .putBoolean(KEY_KEEP_AWAKE, settings.stream.keepScreenAwake)
            .putBoolean(KEY_AUTORESTART, true)
            .apply()
        credentialStore.savePassword(if (settings.stream.authEnabled) settings.password else null)
    }

    private companion object {
        const val PREFERENCES = "camera_settings"
        const val KEY_NAME = "camera_name"
        const val KEY_LENS = "lens"
        const val KEY_WIDTH = "width"
        const val KEY_HEIGHT = "height"
        const val KEY_ASPECT_RATIO = "aspect_ratio"
        const val KEY_FPS = "fps"
        const val KEY_BITRATE = "bitrate"
        const val KEY_KEYFRAME = "keyframe_interval"
        const val KEY_PORT = "port"
        const val KEY_PATH = "path"
        const val KEY_AUTH = "auth_enabled"
        const val KEY_USERNAME = "username"
        const val KEY_ROTATION = "rotation"
        const val KEY_AUTOSTART = "auto_start"
        const val KEY_KEEP_AWAKE = "keep_awake"
        const val KEY_AUTORESTART = "auto_restart_network"
    }
}
