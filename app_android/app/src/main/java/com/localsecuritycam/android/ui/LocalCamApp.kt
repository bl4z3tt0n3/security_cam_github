package com.localsecuritycam.android.ui

import android.app.Activity
import android.view.View
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.widthIn
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.testTagsAsResourceId
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.DialogWindowProvider
import kotlinx.coroutines.delay
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import com.localsecuritycam.android.PlaceholderPanel
import com.localsecuritycam.android.R
import com.localsecuritycam.android.settings.AppSettings
import com.localsecuritycam.android.settings.CameraLens
import com.localsecuritycam.android.ui.screens.CameraScreen
import com.localsecuritycam.android.ui.screens.DiagnosticsScreen
import com.localsecuritycam.android.ui.screens.SettingsScreen
import com.localsecuritycam.android.ui.theme.LocalCamSurface
import com.localsecuritycam.android.ui.theme.LocalCamTheme
import com.localsecuritycam.android.viewmodel.CameraDestination
import com.localsecuritycam.android.viewmodel.CameraUiState

@OptIn(ExperimentalMaterial3Api::class, ExperimentalComposeUiApi::class)
@Composable
internal fun LocalCamApp(
    uiState: CameraUiState,
    existingPassword: String?,
    onSurfaceAvailable: (android.view.Surface, Int, Int) -> Unit,
    onSurfaceDestroyed: () -> Unit,
    onTogglePanel: () -> Unit,
    onStreamAction: () -> Unit,
    onOpenSetup: () -> Unit,
    onOpenDiagnostics: () -> Unit,
    onClosePanel: () -> Unit,
    onLensChanged: (CameraLens) -> Unit,
    onSaveSettings: (AppSettings) -> Unit,
    onValidationError: (String) -> Unit,
) {
    LocalCamTheme {
        val view = LocalView.current
        val window = (view.context as? Activity)?.window
        SideEffect {
            window?.let {
                WindowCompat.getInsetsController(it, view).apply {
                    systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
                    hide(WindowInsetsCompat.Type.systemBars())
                }
            }
        }
        val landscape = LocalConfiguration.current.orientation ==
            android.content.res.Configuration.ORIENTATION_LANDSCAPE

        BackHandler(enabled = uiState.activeDestination != CameraDestination.PREVIEW) {
            onClosePanel()
        }

        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(MaterialTheme.colorScheme.background)
                .semantics { testTagsAsResourceId = true }
                .testTag("localcam_root"),
        ) {
            CameraPreviewLayer(
                onSurfaceAvailable = onSurfaceAvailable,
                onSurfaceDestroyed = onSurfaceDestroyed,
            )
            CameraScreen(
                uiState = uiState,
                landscape = landscape,
                onTogglePanel = onTogglePanel,
                onStreamAction = onStreamAction,
                onOpenSetup = onOpenSetup,
                onOpenDiagnostics = onOpenDiagnostics,
            )
            if (uiState.local.placeholderPanel != PlaceholderPanel.NONE) {
                if (landscape) {
                    LandscapePanel(
                        panel = uiState.local.placeholderPanel,
                        uiState = uiState,
                        existingPassword = existingPassword,
                        onClose = onClosePanel,
                        onLensChanged = onLensChanged,
                        onSaveSettings = onSaveSettings,
                        onValidationError = onValidationError,
                    )
                } else {
                    PortraitPanel(
                        panel = uiState.local.placeholderPanel,
                        uiState = uiState,
                        existingPassword = existingPassword,
                        onClose = onClosePanel,
                        onLensChanged = onLensChanged,
                        onSaveSettings = onSaveSettings,
                        onValidationError = onValidationError,
                    )
                }
            }
        }
    }
}

@Composable
private fun CameraPreviewLayer(
    onSurfaceAvailable: (android.view.Surface, Int, Int) -> Unit,
    onSurfaceDestroyed: () -> Unit,
) {
    com.localsecuritycam.android.ui.components.CameraPreview(
        modifier = Modifier.fillMaxSize(),
        onSurfaceAvailable = onSurfaceAvailable,
        onSurfaceDestroyed = onSurfaceDestroyed,
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun PortraitPanel(
    panel: PlaceholderPanel,
    uiState: CameraUiState,
    existingPassword: String?,
    onClose: () -> Unit,
    onLensChanged: (CameraLens) -> Unit,
    onSaveSettings: (AppSettings) -> Unit,
    onValidationError: (String) -> Unit,
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    ModalBottomSheet(
        sheetState = sheetState,
        onDismissRequest = onClose,
        modifier = Modifier.fillMaxHeight(),
        contentWindowInsets = { WindowInsets(0, 0, 0, 0) },
        containerColor = LocalCamSurface,
    ) {
        HideSystemBarsEffect()
        PanelContent(
            panel = panel,
            uiState = uiState,
            existingPassword = existingPassword,
            onClose = onClose,
            onLensChanged = onLensChanged,
            onSaveSettings = onSaveSettings,
            onValidationError = onValidationError,
        )
    }
}

@Composable
private fun HideSystemBarsEffect() {
    val view = LocalView.current
    fun hide() {
        val window = findWindow(view) ?: return
        val decor = window.decorView
        WindowCompat.getInsetsController(window, decor).apply {
            systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            hide(WindowInsetsCompat.Type.systemBars())
        }
        @Suppress("DEPRECATION")
        run {
            decor.systemUiVisibility = decor.systemUiVisibility or
                View.SYSTEM_UI_FLAG_FULLSCREEN or
                View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or
                View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN or
                View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION or
                View.SYSTEM_UI_FLAG_LAYOUT_STABLE
        }
    }
    SideEffect {
        hide()
    }
    LaunchedEffect(view) {
        repeat(6) {
            hide()
            delay(100)
        }
    }
}

private fun findWindow(view: View): android.view.Window? {
    var parent = view.parent
    while (parent != null) {
        if (parent is DialogWindowProvider) return parent.window
        parent = parent.parent
    }
    return (view.context as? Activity)?.window
}

@Composable
private fun LandscapePanel(
    panel: PlaceholderPanel,
    uiState: CameraUiState,
    existingPassword: String?,
    onClose: () -> Unit,
    onLensChanged: (CameraLens) -> Unit,
    onSaveSettings: (AppSettings) -> Unit,
    onValidationError: (String) -> Unit,
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black.copy(alpha = 0.48f))
            .clickable(onClick = onClose),
    ) {
        Surface(
            modifier = Modifier
                .align(Alignment.CenterEnd)
                .fillMaxHeight()
                .widthIn(max = 420.dp)
                .padding(top = 16.dp, end = 16.dp, bottom = 16.dp)
                .clickable(enabled = false, onClick = {}),
            color = LocalCamSurface,
            shape = MaterialTheme.shapes.large,
            tonalElevation = 6.dp,
        ) {
            PanelContent(
                panel = panel,
                uiState = uiState,
                existingPassword = existingPassword,
                onClose = onClose,
                onLensChanged = onLensChanged,
                onSaveSettings = onSaveSettings,
                onValidationError = onValidationError,
            )
        }
    }
}

@Composable
private fun PanelContent(
    panel: PlaceholderPanel,
    uiState: CameraUiState,
    existingPassword: String?,
    onClose: () -> Unit,
    onLensChanged: (CameraLens) -> Unit,
    onSaveSettings: (AppSettings) -> Unit,
    onValidationError: (String) -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 20.dp, vertical = 12.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = stringResource(
                    if (panel == PlaceholderPanel.SETUP) R.string.setup_title else R.string.diagnostics_title,
                ),
                modifier = Modifier
                    .weight(1f)
                    .testTag("placeholder_title"),
                color = MaterialTheme.colorScheme.onSurface,
                style = MaterialTheme.typography.headlineSmall,
            )
            Button(
                onClick = onClose,
                modifier = Modifier
                    .testTag("placeholder_close_button")
                    .semantics { contentDescription = "Chiudi pannello" },
                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant,
                ),
            ) {
                Text(stringResource(R.string.close))
            }
        }
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(top = 12.dp),
        ) {
            when (panel) {
                PlaceholderPanel.SETUP -> SettingsScreen(
                    settings = uiState.settings,
                    capabilities = uiState.capabilities,
                    capabilitiesLoading = uiState.capabilitiesLoading,
                    message = uiState.settingsMessage,
                    messageIsError = uiState.settingsMessageIsError,
                    existingPassword = existingPassword,
                    onLensChanged = onLensChanged,
                    onSave = onSaveSettings,
                    onValidationError = onValidationError,
                )

                PlaceholderPanel.DIAGNOSTICS -> DiagnosticsScreen(uiState)
                PlaceholderPanel.NONE -> Unit
            }
        }
    }
}
