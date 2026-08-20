package com.localsecuritycam.android.ui.screens

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.localsecuritycam.android.R
import com.localsecuritycam.android.ui.DiagnosticsContent
import com.localsecuritycam.android.viewmodel.CameraUiState

@Composable
internal fun DiagnosticsScreen(
    uiState: CameraUiState,
    modifier: Modifier = Modifier,
) {
    if (uiState.diagnostics.isEmpty()) {
        Text(
            text = stringResource(R.string.diagnostics_title),
            modifier = modifier.padding(16.dp),
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    } else {
        DiagnosticsContent(
            sections = uiState.diagnostics,
            modifier = modifier.fillMaxSize(),
        )
    }
}

