package com.localsecuritycam.android.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.localsecuritycam.android.R
import com.localsecuritycam.android.ui.theme.LocalCamGreen
import com.localsecuritycam.android.ui.theme.LocalCamRed

@Composable
internal fun DiagnosticsContent(
    sections: List<DiagnosticsSectionUiState>,
    modifier: Modifier = Modifier,
) {
    LazyColumn(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(10.dp),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(vertical = 4.dp),
    ) {
        items(sections, key = { section -> section.title }) { section ->
            DiagnosticsSectionCard(section)
        }
    }
}

@Composable
private fun DiagnosticsSectionCard(section: DiagnosticsSectionUiState) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant,
        ),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(
                    painter = painterResource(sectionIcon(section.icon)),
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                )
                Text(
                    text = section.title,
                    modifier = Modifier.padding(start = 12.dp),
                    color = MaterialTheme.colorScheme.onSurface,
                    style = MaterialTheme.typography.titleMedium,
                )
            }
            Column(
                modifier = Modifier.padding(top = 12.dp),
                verticalArrangement = Arrangement.spacedBy(0.dp),
            ) {
                section.rows.forEachIndexed { index, row ->
                    if (index > 0) HorizontalDivider(color = Color.White.copy(alpha = 0.13f))
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 12.dp, vertical = if (row.allowsMultilineValue) 10.dp else 9.dp),
                        verticalAlignment = if (row.allowsMultilineValue) Alignment.Top else Alignment.CenterVertically,
                    ) {
                        Text(
                            text = row.label,
                            modifier = Modifier.weight(0.92f),
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            style = MaterialTheme.typography.bodyMedium,
                        )
                        Text(
                            text = row.value,
                            modifier = Modifier.weight(1.18f),
                            color = valueColor(row.tone),
                            style = MaterialTheme.typography.bodyMedium,
                            textAlign = TextAlign.End,
                        )
                    }
                }
            }
        }
    }
}

private fun sectionIcon(icon: DiagnosticsSectionIcon): Int = when (icon) {
    DiagnosticsSectionIcon.STREAM_CAMERA -> R.drawable.ic_diagnostics_stream_camera
    DiagnosticsSectionIcon.CONNECTION -> R.drawable.ic_diagnostics_connection
    DiagnosticsSectionIcon.ORIENTATION -> R.drawable.ic_diagnostics_orientation
    DiagnosticsSectionIcon.PERFORMANCE -> R.drawable.ic_diagnostics_performance
}

@Composable
private fun valueColor(tone: DiagnosticsValueTone): Color = when (tone) {
    DiagnosticsValueTone.NEUTRAL -> MaterialTheme.colorScheme.onSurface
    DiagnosticsValueTone.SUCCESS -> LocalCamGreen
    DiagnosticsValueTone.ERROR -> LocalCamRed
}
