package com.localsecuritycam.android

import org.junit.Assert.assertEquals
import org.junit.Test

class LocalCamUiStateTest {
    @Test
    fun panelAndSheetStateAreIndependentFromStreamLifecycle() {
        val initial = LocalCamUiState()
        val changed = initial
            .toggleControlPanel()
            .openPlaceholder(PlaceholderPanel.DIAGNOSTICS)

        assertEquals(ControlPanelState.COLLAPSED, changed.controlPanelState)
        assertEquals(PlaceholderPanel.DIAGNOSTICS, changed.placeholderPanel)
    }

    @Test
    fun restoresOnlyUiStateAfterActivityRecreation() {
        val restored = LocalCamUiState.restore(
            ControlPanelState.COLLAPSED.name,
            PlaceholderPanel.SETUP.name,
        )

        assertEquals(ControlPanelState.COLLAPSED, restored.controlPanelState)
        assertEquals(PlaceholderPanel.SETUP, restored.placeholderPanel)
    }
}
