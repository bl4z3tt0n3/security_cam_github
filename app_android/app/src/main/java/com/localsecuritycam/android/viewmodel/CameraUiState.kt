package com.localsecuritycam.android.viewmodel

import android.os.Bundle
import com.localsecuritycam.android.ControlPanelState
import com.localsecuritycam.android.LocalCamUiState
import com.localsecuritycam.android.PlaceholderPanel
import com.localsecuritycam.android.camera.CameraCapabilities
import com.localsecuritycam.android.presentation.StreamingScreenState
import com.localsecuritycam.android.settings.AppSettings
import com.localsecuritycam.android.settings.CameraLens
import com.localsecuritycam.android.ui.DiagnosticsSectionUiState
import com.localsecuritycam.android.ui.SettingsUiState
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import androidx.lifecycle.ViewModel

enum class CameraDestination {
    PREVIEW,
    SETUP,
    DIAGNOSTICS,
}

internal data class CameraUiState(
    val local: LocalCamUiState = LocalCamUiState(),
    val streaming: StreamingScreenState? = null,
    val settings: SettingsUiState = SettingsUiState(),
    val capabilities: CameraCapabilities? = null,
    val capabilitiesLoading: Boolean = false,
    val settingsMessage: String? = null,
    val settingsMessageIsError: Boolean = false,
    val diagnostics: List<DiagnosticsSectionUiState> = emptyList(),
) {
    val activeDestination: CameraDestination
        get() = when (local.placeholderPanel) {
            PlaceholderPanel.NONE -> CameraDestination.PREVIEW
            PlaceholderPanel.SETUP -> CameraDestination.SETUP
            PlaceholderPanel.DIAGNOSTICS -> CameraDestination.DIAGNOSTICS
        }
}

internal sealed interface CameraUiEffect {
    data object StartStream : CameraUiEffect
    data object StopStream : CameraUiEffect
    data class SaveSettings(val settings: AppSettings) : CameraUiEffect
    data class QueryCapabilities(val lens: CameraLens) : CameraUiEffect
}

internal class CameraViewModel : ViewModel() {
    private val _state = MutableStateFlow(CameraUiState())
    val state: StateFlow<CameraUiState> = _state.asStateFlow()

    private val _effects = MutableSharedFlow<CameraUiEffect>(extraBufferCapacity = 8)
    val effects: SharedFlow<CameraUiEffect> = _effects.asSharedFlow()

    private val presenter = com.localsecuritycam.android.presentation.LocalCamStreamPresenter(
        onStartRequested = { _effects.tryEmit(CameraUiEffect.StartStream) },
        onStopRequested = { _effects.tryEmit(CameraUiEffect.StopStream) },
        onStateChanged = { next -> _state.update { it.copy(streaming = next) } },
    )

    fun onSnapshot(snapshot: com.localsecuritycam.android.service.ServiceSnapshot) {
        presenter.onSnapshot(snapshot)
        _state.update {
            it.copy(
                // ServiceSnapshot is redacted. Preserve only the in-memory
                // configured flag received from LocalBinder.settings().
                settings = it.settings.copy(stream = snapshot.settings.stream),
                diagnostics = com.localsecuritycam.android.ui.diagnosticsSections(snapshot),
            )
        }
    }

    fun onSettings(value: AppSettings) {
        _state.update { it.copy(settings = SettingsUiState.from(value)) }
    }

    fun onStreamActionClicked() = presenter.onStreamActionClicked()

    fun toggleControlPanel() {
        _state.update { it.copy(local = it.local.toggleControlPanel()) }
    }

    fun openDestination(destination: CameraDestination) {
        val panel = when (destination) {
            CameraDestination.PREVIEW -> PlaceholderPanel.NONE
            CameraDestination.SETUP -> PlaceholderPanel.SETUP
            CameraDestination.DIAGNOSTICS -> PlaceholderPanel.DIAGNOSTICS
        }
        _state.update { it.copy(local = it.local.copy(placeholderPanel = panel)) }
    }

    fun closeDestination() {
        _state.update { it.copy(local = it.local.closePlaceholder()) }
    }

    fun requestCapabilities(lens: CameraLens) {
        _state.update { it.copy(capabilities = null, capabilitiesLoading = true, settingsMessage = null) }
        _effects.tryEmit(CameraUiEffect.QueryCapabilities(lens))
    }

    fun onCapabilities(value: CameraCapabilities) {
        _state.update { it.copy(capabilities = value, capabilitiesLoading = false) }
    }

    fun onCapabilitiesUnavailable(message: String = "Lettura capability camera/AVC in corso") {
        _state.update {
            it.copy(
                capabilities = null,
                capabilitiesLoading = false,
                settingsMessage = message,
                settingsMessageIsError = true,
            )
        }
    }

    fun submitSettings(value: AppSettings) {
        _state.update {
            it.copy(
                settings = SettingsUiState.from(value),
                settingsMessage = "Impostazioni salvate",
                settingsMessageIsError = false,
            )
        }
        _effects.tryEmit(CameraUiEffect.SaveSettings(value))
    }

    fun showSettingsError(message: String) {
        _state.update { it.copy(settingsMessage = message, settingsMessageIsError = true) }
    }

    fun saveUiState(outState: Bundle) {
        outState.putString(SAVED_CONTROL_PANEL_STATE, _state.value.local.controlPanelState.name)
        outState.putString(SAVED_PLACEHOLDER_PANEL, _state.value.local.placeholderPanel.name)
    }

    fun restoreUiState(savedState: Bundle?) {
        if (savedState == null) return
        val restored = LocalCamUiState.restore(
            savedState.getString(SAVED_CONTROL_PANEL_STATE),
            savedState.getString(SAVED_PLACEHOLDER_PANEL),
        )
        _state.update { it.copy(local = restored) }
    }

    private companion object {
        const val SAVED_CONTROL_PANEL_STATE = "control_panel_state"
        const val SAVED_PLACEHOLDER_PANEL = "placeholder_panel"
    }
}
