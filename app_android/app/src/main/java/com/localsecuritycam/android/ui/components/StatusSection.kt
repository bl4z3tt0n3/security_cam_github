package com.localsecuritycam.android.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import com.localsecuritycam.android.R
import com.localsecuritycam.android.presentation.HeaderUiMode
import com.localsecuritycam.android.presentation.StreamingScreenState
import com.localsecuritycam.android.presentation.StreamingVisualState
import com.localsecuritycam.android.ui.theme.LocalCamGreen
import com.localsecuritycam.android.ui.theme.LocalCamMuted
import com.localsecuritycam.android.ui.theme.LocalCamRed

@Composable
internal fun StatusSection(
    state: StreamingScreenState?,
    cameraLabel: String,
    modifier: Modifier = Modifier,
) {
    val visualState = state?.visualState
    val header = state?.headerLabel ?: "LAN"
    val statusColor = when (visualState) {
        StreamingVisualState.ERROR -> LocalCamRed
        StreamingVisualState.STOPPED,
        StreamingVisualState.WAITING_FOR_NETWORK,
        null,
        -> LocalCamGreen
        else -> MaterialTheme.colorScheme.onSurface
    }
    Row(
        modifier = modifier
            .fillMaxWidth()
            .background(Color.Black.copy(alpha = 0.42f))
            .padding(horizontal = 20.dp, vertical = 14.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.End,
    ) {
        Text(
            text = "LocalCam",
            modifier = Modifier.weight(1f).testTag("localcam_title"),
            color = MaterialTheme.colorScheme.onSurface,
            style = MaterialTheme.typography.headlineSmall,
            maxLines = 1,
        )
        Surface(
            modifier = Modifier
                .testTag("header_status_chip")
                .semantics { contentDescription = "Stato stream $header" },
            color = Color.Black.copy(alpha = 0.35f),
            shape = MaterialTheme.shapes.small,
        ) {
            Text(
                text = "● $header",
                modifier = Modifier.padding(horizontal = 14.dp, vertical = 12.dp),
                color = statusColor,
                style = MaterialTheme.typography.labelLarge,
            )
        }
        if (state?.headerUiMode?.showsCameraSelector != false) {
            Spacer(Modifier.width(8.dp))
            Surface(
                modifier = Modifier
                    .testTag("camera_selector")
                    .semantics { contentDescription = "Obiettivo selezionato" },
                color = Color.Black.copy(alpha = 0.35f),
                shape = MaterialTheme.shapes.small,
            ) {
                Text(
                    text = cameraLabel,
                    modifier = Modifier.padding(horizontal = 14.dp, vertical = 12.dp),
                    color = MaterialTheme.colorScheme.onSurface,
                    style = MaterialTheme.typography.labelLarge,
                )
            }
        }
    }
}

@Composable
internal fun PreviewBadge(
    state: StreamingScreenState?,
    modifier: Modifier = Modifier,
) {
    val label = state?.previewLabel ?: "Preview idle"
    val active = state?.serverReady == true || state?.previewState == com.localsecuritycam.android.service.PreviewState.ACTIVE
    Surface(
        modifier = modifier
            .testTag("preview_status_badge")
            .semantics { contentDescription = label },
        color = if (active) Color.Black.copy(alpha = 0.68f) else Color.Black.copy(alpha = 0.52f),
        shape = MaterialTheme.shapes.small,
    ) {
        Text(
            text = label,
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 12.dp),
            color = if (active) LocalCamGreen else LocalCamMuted,
            style = MaterialTheme.typography.labelLarge,
        )
    }
}

