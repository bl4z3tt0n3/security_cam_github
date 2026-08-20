package com.localsecuritycam.android.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import com.localsecuritycam.android.R
import com.localsecuritycam.android.presentation.StreamAction
import com.localsecuritycam.android.presentation.StreamingScreenState
import com.localsecuritycam.android.presentation.StreamingVisualState
import com.localsecuritycam.android.ui.SettingsUiState
import com.localsecuritycam.android.ui.theme.LocalCamAction
import com.localsecuritycam.android.ui.theme.LocalCamRed

@Composable
internal fun StreamControls(
    state: StreamingScreenState?,
    settings: SettingsUiState,
    collapsed: Boolean,
    landscape: Boolean,
    onToggleCollapsed: () -> Unit,
    onStreamAction: () -> Unit,
    onOpenSetup: () -> Unit,
    onOpenDiagnostics: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val action = state?.action ?: StreamAction.START
    val label = when (action) {
        StreamAction.START -> if (state?.visualState == StreamingVisualState.ERROR) "Riprova" else "Start Stream"
        StreamAction.STOP -> "Stop Stream"
        StreamAction.NONE -> "Arresto..."
    }
    val isStop = action == StreamAction.STOP || state?.visualState == StreamingVisualState.STOPPING
    val enabled = state?.actionEnabled ?: true

    Surface(
        modifier = modifier
            .then(if (landscape) Modifier.widthIn(max = if (collapsed) 72.dp else 252.dp) else Modifier.fillMaxWidth())
            .then(
                if (landscape) Modifier.fillMaxHeight()
                else Modifier.heightIn(min = 132.dp, max = 420.dp)
            )
            .testTag("control_panel"),
        color = if (isStop) LocalCamRed.copy(alpha = 0.93f) else Color.Black.copy(alpha = 0.78f),
        shape = MaterialTheme.shapes.large,
        tonalElevation = 4.dp,
    ) {
        if (landscape && collapsed) {
            CollapsedRail(label, enabled, isStop, onToggleCollapsed, onStreamAction)
        } else {
            Column(
                modifier = Modifier
                    .padding(16.dp)
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                if (!landscape) {
                    Text(
                        text = if (collapsed) "Espandi controlli" else "Comprimi controlli",
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(min = 40.dp)
                            .testTag("panel_handle")
                            .semantics { contentDescription = if (collapsed) "Espandi controlli" else "Comprimi controlli" },
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        style = MaterialTheme.typography.labelLarge,
                    )
                } else {
                    OutlinedButton(
                        onClick = onToggleCollapsed,
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(min = 44.dp)
                            .testTag("panel_handle"),
                    ) {
                        Text("Comprimi")
                    }
                }
                StreamActionButton(
                    label = label,
                    enabled = enabled,
                    stop = isStop,
                    onClick = onStreamAction,
                    modifier = Modifier.testTag("stream_action_button"),
                )
                Text(
                    text = "${settings.stream.resolution} • ${settings.stream.fps} FPS • " +
                        "${settings.stream.bitrate / 1000} kbps",
                    modifier = Modifier
                        .fillMaxWidth()
                        .testTag("stream_summary"),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    style = MaterialTheme.typography.bodyMedium,
                )
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    SecondaryAction(
                        text = "Preview",
                        modifier = Modifier.weight(1f).testTag("preview_action"),
                        onClick = {},
                    )
                    SecondaryAction(
                        text = "Setup",
                        modifier = Modifier.weight(1f).testTag("setup_action"),
                        onClick = onOpenSetup,
                    )
                    SecondaryAction(
                        text = "Diagnostics",
                        modifier = Modifier.weight(1f).testTag("diagnostics_action"),
                        onClick = onOpenDiagnostics,
                    )
                }
                Text(
                    text = state?.panelMessage ?: "Camera pronta",
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(Color.Black.copy(alpha = 0.22f), MaterialTheme.shapes.small)
                        .padding(horizontal = 12.dp, vertical = 12.dp)
                        .testTag("panel_info_text"),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
        }
    }
}

@Composable
private fun CollapsedRail(
    label: String,
    enabled: Boolean,
    stop: Boolean,
    onToggleCollapsed: () -> Unit,
    onStreamAction: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxHeight()
            .padding(vertical = 14.dp, horizontal = 8.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        StreamActionButton(
            label = label,
            enabled = enabled,
            stop = stop,
            onClick = onStreamAction,
            modifier = Modifier
                .width(56.dp)
                .heightIn(min = 80.dp)
                .testTag("collapsed_stream_action"),
            vertical = true,
        )
        OutlinedButton(
            onClick = onToggleCollapsed,
            modifier = Modifier
                .padding(top = 12.dp)
                .width(56.dp)
                .heightIn(min = 48.dp),
        ) {
            Text("›")
        }
    }
}

@Composable
private fun StreamActionButton(
    label: String,
    enabled: Boolean,
    stop: Boolean,
    onClick: () -> Unit,
    modifier: Modifier,
    vertical: Boolean = false,
) {
    Button(
        onClick = onClick,
        enabled = enabled,
        modifier = modifier
            .then(if (vertical) Modifier else Modifier.fillMaxWidth())
            .heightIn(min = 52.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = if (stop) LocalCamRed else LocalCamAction,
            contentColor = Color.White,
            disabledContainerColor = LocalCamAction.copy(alpha = 0.35f),
        ),
        shape = MaterialTheme.shapes.medium,
    ) {
        Icon(
            painter = painterResource(if (stop) R.drawable.ic_stop else R.drawable.ic_play),
            contentDescription = null,
        )
        if (!vertical) {
            androidx.compose.foundation.layout.Spacer(Modifier.width(10.dp))
            Text(label, style = MaterialTheme.typography.titleMedium)
        }
    }
}

@Composable
private fun SecondaryAction(
    text: String,
    modifier: Modifier,
    onClick: () -> Unit,
) {
    OutlinedButton(
        onClick = onClick,
        modifier = modifier.heightIn(min = 48.dp),
        colors = ButtonDefaults.outlinedButtonColors(
            contentColor = MaterialTheme.colorScheme.onSurface,
        ),
        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 8.dp),
    ) {
        Text(text, maxLines = 1, style = MaterialTheme.typography.labelLarge)
    }
}
