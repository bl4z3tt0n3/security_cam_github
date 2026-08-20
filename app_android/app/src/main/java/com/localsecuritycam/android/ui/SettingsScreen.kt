package com.localsecuritycam.android.ui

import androidx.compose.runtime.Immutable
import com.localsecuritycam.android.camera.CameraCapabilities
import com.localsecuritycam.android.camera.CameraCapabilitiesProvider
import com.localsecuritycam.android.settings.AppSettings
import com.localsecuritycam.android.settings.CameraLens
import com.localsecuritycam.android.settings.Resolution
import com.localsecuritycam.android.settings.StreamSettings
import com.localsecuritycam.android.settings.VideoAspectRatio

/** Non-sensitive settings projection used by Compose. The password never enters this state. */
@Immutable
internal data class SettingsUiState(
    val stream: StreamSettings = StreamSettings(),
    val passwordConfigured: Boolean = false,
) {
    companion object {
        fun from(value: AppSettings): SettingsUiState = SettingsUiState(
            stream = value.stream,
            passwordConfigured = !value.password.isNullOrEmpty(),
        )
    }
}

@Immutable
internal data class SettingsFormState(
    val cameraName: String,
    val lens: CameraLens,
    val resolution: Resolution,
    val aspectRatio: VideoAspectRatio,
    val fps: Int,
    val bitrate: String,
    val keyframeIntervalSeconds: String,
    val port: String,
    val streamPath: String,
    val authEnabled: Boolean,
    val username: String,
    val autoStart: Boolean,
    val keepScreenAwake: Boolean,
) {
    companion object {
        fun from(stream: StreamSettings): SettingsFormState = SettingsFormState(
            cameraName = stream.cameraName,
            lens = stream.lens,
            resolution = stream.resolution,
            aspectRatio = stream.aspectRatio,
            fps = stream.fps,
            bitrate = stream.bitrate.toString(),
            keyframeIntervalSeconds = stream.keyframeIntervalSeconds.toString(),
            port = stream.port.toString(),
            streamPath = stream.streamPath,
            authEnabled = stream.authEnabled,
            username = stream.username,
            autoStart = stream.autoStart,
            keepScreenAwake = stream.keepScreenAwake,
        )
    }
}

internal data class SettingsValidationResult(
    val settings: AppSettings? = null,
    val errors: List<String> = emptyList(),
)

internal fun SettingsFormState.toValidatedSettings(
    capabilities: CameraCapabilities?,
    existingPassword: String?,
    typedPassword: String,
): SettingsValidationResult {
    if (capabilities == null) {
        return SettingsValidationResult(errors = listOf("Attendere le capability reali della camera e dell'encoder AVC"))
    }
    val selectedResolution = capabilities.resolutions.firstOrNull { it == resolution }
        ?: capabilities.resolutions.firstOrNull()
    if (selectedResolution == null) {
        return SettingsValidationResult(errors = listOf("Nessuna risoluzione camera disponibile"))
    }
    val finalPassword = typedPassword.ifEmpty { existingPassword }
    val stream = StreamSettings(
        cameraName = cameraName.trim(),
        lens = lens,
        resolution = selectedResolution,
        aspectRatio = aspectRatio,
        fps = fps,
        bitrate = bitrate.toIntOrNull() ?: 0,
        keyframeIntervalSeconds = keyframeIntervalSeconds.toIntOrNull() ?: 0,
        port = port.toIntOrNull() ?: 0,
        streamPath = streamPath,
        authEnabled = authEnabled,
        username = username.trim(),
        autoStart = autoStart,
        keepScreenAwake = keepScreenAwake,
        autoRestartOnNetwork = true,
    )
    val errors = buildList {
        addAll(stream.validate(if (stream.authEnabled) finalPassword else null))
        addAll(CameraCapabilitiesProvider.validationErrors(capabilities, stream))
    }
    return if (errors.isEmpty()) {
        SettingsValidationResult(
            settings = AppSettings(stream, if (stream.authEnabled) finalPassword else null),
        )
    } else {
        SettingsValidationResult(errors = errors)
    }
}
