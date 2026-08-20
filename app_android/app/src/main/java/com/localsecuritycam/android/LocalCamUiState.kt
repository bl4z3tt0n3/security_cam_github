package com.localsecuritycam.android

enum class ControlPanelState {
    EXPANDED,
    COLLAPSED,
}

enum class PlaceholderPanel {
    NONE,
    SETUP,
    DIAGNOSTICS,
}

/**
 * UI-only state. Streaming state is never stored here: it is observed from
 * CameraStreamingService through ServiceSnapshot.
 */
data class LocalCamUiState(
    val controlPanelState: ControlPanelState = ControlPanelState.EXPANDED,
    val placeholderPanel: PlaceholderPanel = PlaceholderPanel.NONE,
) {
    fun toggleControlPanel(): LocalCamUiState = copy(
        controlPanelState = when (controlPanelState) {
            ControlPanelState.EXPANDED -> ControlPanelState.COLLAPSED
            ControlPanelState.COLLAPSED -> ControlPanelState.EXPANDED
        },
    )

    fun openPlaceholder(panel: PlaceholderPanel): LocalCamUiState = copy(
        placeholderPanel = panel,
    )

    fun closePlaceholder(): LocalCamUiState = copy(
        placeholderPanel = PlaceholderPanel.NONE,
    )

    companion object {
        fun restore(
            controlPanelStateName: String?,
            placeholderPanelName: String?,
        ): LocalCamUiState = LocalCamUiState(
            controlPanelState = ControlPanelState.entries.firstOrNull {
                it.name == controlPanelStateName
            } ?: ControlPanelState.EXPANDED,
            placeholderPanel = PlaceholderPanel.entries.firstOrNull {
                it.name == placeholderPanelName
            } ?: PlaceholderPanel.NONE,
        )
    }
}
