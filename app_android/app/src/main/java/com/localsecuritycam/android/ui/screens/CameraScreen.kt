package com.localsecuritycam.android.ui.screens

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.systemBars
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.unit.dp
import com.localsecuritycam.android.ControlPanelState
import com.localsecuritycam.android.ui.SettingsUiState
import com.localsecuritycam.android.ui.components.PreviewBadge
import com.localsecuritycam.android.ui.components.StatusSection
import com.localsecuritycam.android.ui.components.StreamControls
import com.localsecuritycam.android.ui.theme.LocalCamShell
import com.localsecuritycam.android.viewmodel.CameraUiState

@Composable
internal fun CameraScreen(
    uiState: CameraUiState,
    landscape: Boolean,
    onTogglePanel: () -> Unit,
    onStreamAction: () -> Unit,
    onOpenSetup: () -> Unit,
    onOpenDiagnostics: () -> Unit,
) {
    val collapsed = uiState.local.controlPanelState == ControlPanelState.COLLAPSED
    Box(
        modifier = Modifier
            .fillMaxSize()
            .testTag("camera_screen"),
    ) {
        StatusSection(
            state = uiState.streaming,
            cameraLabel = when (uiState.settings.stream.lens) {
                com.localsecuritycam.android.settings.CameraLens.BACK -> "Rear 1x"
                com.localsecuritycam.android.settings.CameraLens.FRONT -> "Front"
            },
            modifier = Modifier
                .align(Alignment.TopCenter)
                .windowInsetsPadding(WindowInsets.systemBars),
        )
        PreviewBadge(
            state = uiState.streaming,
            modifier = Modifier
                .align(Alignment.TopStart)
                .windowInsetsPadding(WindowInsets.systemBars)
                .padding(start = 24.dp, top = if (landscape) 82.dp else 88.dp),
        )
        StreamControls(
            state = uiState.streaming,
            settings = uiState.settings,
            collapsed = collapsed,
            landscape = landscape,
            onToggleCollapsed = onTogglePanel,
            onStreamAction = onStreamAction,
            onOpenSetup = onOpenSetup,
            onOpenDiagnostics = onOpenDiagnostics,
            modifier = Modifier
                .align(if (landscape) Alignment.CenterEnd else Alignment.BottomCenter)
                .windowInsetsPadding(WindowInsets.systemBars)
                .padding(
                    start = if (landscape) 0.dp else 16.dp,
                    end = if (landscape) 16.dp else 16.dp,
                    bottom = if (landscape) 16.dp else 16.dp,
                ),
        )
    }
}
