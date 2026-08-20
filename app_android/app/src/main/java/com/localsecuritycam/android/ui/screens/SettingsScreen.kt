package com.localsecuritycam.android.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.text.KeyboardOptions
import com.localsecuritycam.android.R
import com.localsecuritycam.android.camera.CameraCapabilities
import com.localsecuritycam.android.camera.CameraCapabilitiesProvider
import com.localsecuritycam.android.settings.CameraLens
import com.localsecuritycam.android.settings.Resolution
import com.localsecuritycam.android.settings.StreamPreset
import com.localsecuritycam.android.ui.SettingsFormState
import com.localsecuritycam.android.ui.SettingsUiState
import com.localsecuritycam.android.ui.toValidatedSettings
import com.localsecuritycam.android.ui.components.ChoiceField
import com.localsecuritycam.android.ui.components.FormSurface
import com.localsecuritycam.android.ui.components.SectionDivider
import com.localsecuritycam.android.ui.components.SectionTitle
import com.localsecuritycam.android.ui.components.SettingRow
import com.localsecuritycam.android.ui.components.ToggleRow
import com.localsecuritycam.android.ui.theme.LocalCamAction
import com.localsecuritycam.android.ui.theme.LocalCamGreen
import com.localsecuritycam.android.ui.theme.LocalCamRed

@Composable
internal fun SettingsScreen(
    settings: SettingsUiState,
    capabilities: CameraCapabilities?,
    capabilitiesLoading: Boolean,
    message: String?,
    messageIsError: Boolean,
    existingPassword: String?,
    onLensChanged: (CameraLens) -> Unit,
    onSave: (com.localsecuritycam.android.settings.AppSettings) -> Unit,
    onValidationError: (String) -> Unit,
) {
    var form by remember(settings, existingPassword) {
        mutableStateOf(SettingsFormState.from(settings.stream))
    }
    var password by remember(settings, existingPassword) { mutableStateOf("") }
    var passwordVisible by remember { mutableStateOf(false) }
    val passwordDescription = stringResource(
        if (passwordVisible) R.string.setup_hide_password else R.string.setup_show_password,
    )

    LaunchedEffect(capabilities) {
        capabilities?.let { value ->
            val selectedResolution = value.resolutions.firstOrNull { it == form.resolution }
                ?: value.resolutions.firstOrNull()
            val fpsOptions = selectedResolution?.let { value.fpsByResolution[it] }.orEmpty()
            val selectedFps = when {
                form.fps in fpsOptions -> form.fps
                fpsOptions.isNotEmpty() -> fpsOptions.first()
                else -> form.fps
            }
            if (selectedResolution != null) {
                form = form.copy(resolution = selectedResolution, fps = selectedFps)
            }
        }
    }

    val resolutions = capabilities?.resolutions.orEmpty()
    val selectedResolution = resolutions.firstOrNull { it == form.resolution } ?: form.resolution
    val fpsOptions = capabilities?.fpsByResolution?.get(selectedResolution)
        ?: capabilities?.fpsValues.orEmpty()
    val supportedPresets = capabilities?.let { value ->
        StreamPreset.entries.filter { preset ->
            CameraCapabilitiesProvider.validationErrors(value, preset.settings).isEmpty()
        }
    }.orEmpty()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp, vertical = 4.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        SectionTitle(stringResource(R.string.setup_section_stream_settings))
        FormSurface {
            SettingRow(stringResource(R.string.setup_camera_name)) {
                TextInput(
                    value = form.cameraName,
                    onValueChange = { form = form.copy(cameraName = it) },
                    label = stringResource(R.string.setup_camera_name),
                    modifier = Modifier.testTag("setting_camera_name"),
                )
            }
            SectionDivider()
            SettingRow(stringResource(R.string.setup_camera_lens)) {
                ChoiceField(
                    label = stringResource(R.string.setup_camera_lens),
                    value = lensLabel(form.lens),
                    options = CameraLens.entries.map(::lensLabel),
                    onSelected = { index ->
                        val lens = CameraLens.entries[index]
                        form = form.copy(lens = lens)
                        onLensChanged(lens)
                    },
                    modifier = Modifier.testTag("setting_camera_lens"),
                )
            }
            SectionDivider()
            SettingRow(stringResource(R.string.setup_resolution)) {
                ChoiceField(
                    label = stringResource(R.string.setup_resolution),
                    value = selectedResolution.toString(),
                    options = resolutions.map(Resolution::toString),
                    enabled = !capabilitiesLoading,
                    onSelected = { index ->
                        resolutions.getOrNull(index)?.let { next ->
                            val nextFps = capabilities?.fpsByResolution?.get(next).orEmpty()
                                .let { values -> if (form.fps in values) form.fps else values.firstOrNull() ?: form.fps }
                            form = form.copy(resolution = next, fps = nextFps)
                        }
                    },
                    modifier = Modifier.testTag("setting_resolution"),
                )
            }
            SectionDivider()
            SettingRow(stringResource(R.string.setup_aspect_ratio)) {
                ChoiceField(
                    label = stringResource(R.string.setup_aspect_ratio),
                    value = form.aspectRatio.label,
                    options = com.localsecuritycam.android.settings.VideoAspectRatio.entries.map { it.label },
                    onSelected = { index ->
                        form = form.copy(
                            aspectRatio = com.localsecuritycam.android.settings.VideoAspectRatio.entries[index],
                        )
                    },
                    modifier = Modifier.testTag("setting_aspect_ratio"),
                )
            }
            SectionDivider()
            SettingRow(stringResource(R.string.setup_fps)) {
                ChoiceField(
                    label = stringResource(R.string.setup_fps),
                    value = form.fps.toString(),
                    options = fpsOptions.map(Int::toString),
                    enabled = !capabilitiesLoading,
                    onSelected = { index ->
                        fpsOptions.getOrNull(index)?.let { form = form.copy(fps = it) }
                    },
                    modifier = Modifier.testTag("setting_fps"),
                )
            }
            SectionDivider()
            SettingRow(stringResource(R.string.setup_bitrate)) {
                TextInput(
                    value = form.bitrate,
                    onValueChange = { form = form.copy(bitrate = it.filter(Char::isDigit)) },
                    label = stringResource(R.string.setup_bitrate),
                    keyboardType = KeyboardType.Number,
                    modifier = Modifier.testTag("setting_bitrate"),
                )
            }
            SectionDivider()
            SettingRow(stringResource(R.string.setup_gop)) {
                ChoiceField(
                    label = stringResource(R.string.setup_gop),
                    value = form.keyframeIntervalSeconds,
                    options = (1..10).map(Int::toString),
                    onSelected = { index -> form = form.copy(keyframeIntervalSeconds = (index + 1).toString()) },
                    modifier = Modifier.testTag("setting_gop"),
                )
            }
            SectionDivider()
            SettingRow(stringResource(R.string.setup_port)) {
                TextInput(
                    value = form.port,
                    onValueChange = { form = form.copy(port = it.filter(Char::isDigit)) },
                    label = stringResource(R.string.setup_port),
                    keyboardType = KeyboardType.Number,
                    modifier = Modifier.testTag("setting_port"),
                )
            }
            SectionDivider()
            SettingRow(stringResource(R.string.setup_path)) {
                TextInput(
                    value = form.streamPath,
                    onValueChange = { form = form.copy(streamPath = it) },
                    label = stringResource(R.string.setup_path),
                    modifier = Modifier.testTag("setting_path"),
                )
            }
        }

        SectionTitle(stringResource(R.string.setup_section_quality_preset))
        FormSurface {
            Row(
                modifier = Modifier.fillMaxWidth().padding(4.dp),
                horizontalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                supportedPresets.forEach { preset ->
                    val selected = presetMatches(form, preset)
                    Button(
                        onClick = {
                            val value = preset.settings
                            form = form.copy(
                                resolution = value.resolution,
                                fps = value.fps,
                                bitrate = value.bitrate.toString(),
                                keyframeIntervalSeconds = value.keyframeIntervalSeconds.toString(),
                            )
                        },
                        modifier = Modifier
                            .weight(1f)
                            .heightIn(min = 48.dp)
                            .semantics { contentDescription = preset.label },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (selected) LocalCamAction else MaterialTheme.colorScheme.surfaceVariant,
                            contentColor = MaterialTheme.colorScheme.onSurface,
                        ),
                    ) {
                        Text(preset.label)
                    }
                }
                if (supportedPresets.isEmpty()) {
                    Text(
                        text = if (capabilitiesLoading) "Lettura capability..." else "Nessun preset disponibile",
                        modifier = Modifier.padding(16.dp),
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }

        SectionTitle(stringResource(R.string.setup_section_authentication))
        FormSurface {
            ToggleRow(
                label = stringResource(R.string.setup_enable_authentication),
                checked = form.authEnabled,
                onCheckedChange = { form = form.copy(authEnabled = it) },
            )
            if (form.authEnabled) {
                SectionDivider()
                SettingRow(stringResource(R.string.setup_username)) {
                    TextInput(
                        value = form.username,
                        onValueChange = { form = form.copy(username = it) },
                        label = stringResource(R.string.setup_username),
                        modifier = Modifier.testTag("setting_username"),
                    )
                }
                SectionDivider()
                SettingRow(stringResource(R.string.setup_password)) {
                    OutlinedTextField(
                        value = password,
                        onValueChange = { password = it },
                        modifier = Modifier
                            .fillMaxWidth()
                            .testTag("setting_password"),
                        singleLine = true,
                        label = {
                            Text(
                                text = stringResource(R.string.setup_password),
                                maxLines = 1,
                                softWrap = false,
                            )
                        },
                        placeholder = {
                            if (settings.passwordConfigured) {
                                Text(stringResource(R.string.setup_password_configured))
                            }
                        },
                        visualTransformation = if (passwordVisible) {
                            VisualTransformation.None
                        } else {
                            PasswordVisualTransformation()
                        },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                        trailingIcon = {
                            IconButton(
                                onClick = { passwordVisible = !passwordVisible },
                                modifier = Modifier.semantics {
                                    contentDescription = passwordDescription
                                },
                            ) {
                                Icon(
                                    painter = painterResource(
                                        if (passwordVisible) R.drawable.ic_setup_visibility_off
                                        else R.drawable.ic_setup_visibility,
                                    ),
                                    contentDescription = null,
                                )
                            }
                        },
                    )
                }
            }
        }

        SectionTitle(stringResource(R.string.setup_section_behavior))
        FormSurface {
            ToggleRow(
                label = stringResource(R.string.setup_auto_start),
                checked = form.autoStart,
                onCheckedChange = { form = form.copy(autoStart = it) },
            )
            SectionDivider()
            ToggleRow(
                label = stringResource(R.string.setup_keep_awake),
                checked = form.keepScreenAwake,
                onCheckedChange = { form = form.copy(keepScreenAwake = it) },
            )
        }

        message?.takeIf { it.isNotBlank() }?.let {
            Text(
                text = it,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 8.dp)
                    .testTag("settings_message"),
                color = if (messageIsError) LocalCamRed else LocalCamGreen,
                style = MaterialTheme.typography.bodyMedium,
            )
        }

        Button(
            onClick = {
                val result = form.toValidatedSettings(capabilities, existingPassword, password)
                if (result.settings != null) onSave(result.settings)
                else onValidationError(result.errors.joinToString("\n"))
            },
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 12.dp, bottom = 20.dp)
                .heightIn(min = 56.dp)
                .testTag("setup_save_settings")
                .semantics { contentDescription = "Salva impostazioni" },
            colors = ButtonDefaults.buttonColors(containerColor = LocalCamAction),
        ) {
            Text(stringResource(R.string.setup_save_settings), style = MaterialTheme.typography.titleMedium)
        }
        Spacer(modifier = Modifier.height(88.dp))
    }
}

@Composable
private fun TextInput(
    value: String,
    onValueChange: (String) -> Unit,
    label: String,
    modifier: Modifier = Modifier,
    keyboardType: KeyboardType = KeyboardType.Text,
) {
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        modifier = modifier.fillMaxWidth(),
        singleLine = true,
        label = { Text(label) },
        keyboardOptions = KeyboardOptions(keyboardType = keyboardType),
    )
}

private fun presetMatches(form: SettingsFormState, preset: StreamPreset): Boolean =
    form.resolution == preset.settings.resolution &&
        form.fps == preset.settings.fps &&
        form.bitrate.toIntOrNull() == preset.settings.bitrate &&
        form.keyframeIntervalSeconds.toIntOrNull() == preset.settings.keyframeIntervalSeconds

private fun lensLabel(lens: CameraLens): String = when (lens) {
    CameraLens.BACK -> "Back"
    CameraLens.FRONT -> "Front"
}
