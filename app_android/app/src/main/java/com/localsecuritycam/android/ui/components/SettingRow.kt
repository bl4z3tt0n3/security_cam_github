package com.localsecuritycam.android.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import com.localsecuritycam.android.ui.theme.LocalCamBeige
import com.localsecuritycam.android.ui.theme.LocalCamCreamLight
import com.localsecuritycam.android.ui.theme.LocalCamMuted
import com.localsecuritycam.android.ui.theme.LocalCamSubtle

@Composable
internal fun SettingRow(
    label: String,
    modifier: Modifier = Modifier,
    content: @Composable () -> Unit,
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .heightIn(min = 56.dp)
            .padding(horizontal = 16.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(
            text = label,
            modifier = Modifier.weight(0.75f),
            color = LocalCamBeige,
            style = androidx.compose.material3.MaterialTheme.typography.bodyLarge,
        )
        Column(
            modifier = Modifier.weight(1.25f),
            horizontalAlignment = Alignment.End,
        ) {
            content()
        }
    }
}

@Composable
internal fun SectionTitle(text: String, modifier: Modifier = Modifier) {
    Text(
        text = text,
        modifier = modifier
            .fillMaxWidth()
            .padding(top = 18.dp, bottom = 4.dp),
        color = LocalCamBeige,
        style = androidx.compose.material3.MaterialTheme.typography.titleMedium,
    )
}

@Composable
internal fun ChoiceField(
    label: String,
    value: String,
    options: List<String>,
    enabled: Boolean = true,
    modifier: Modifier = Modifier,
    onSelected: (Int) -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }
    Column(horizontalAlignment = Alignment.End) {
        OutlinedButton(
            onClick = { expanded = true },
            enabled = enabled && options.isNotEmpty(),
            modifier = modifier
                .fillMaxWidth()
                .semantics { contentDescription = label },
            shape = RoundedCornerShape(10.dp),
        ) {
            Text(
                text = value,
                color = if (enabled) LocalCamCreamLight else LocalCamMuted,
                maxLines = 1,
            )
        }
        DropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false },
        ) {
            options.forEachIndexed { index, option ->
                DropdownMenuItem(
                    text = { Text(option) },
                    onClick = {
                        expanded = false
                        onSelected(index)
                    },
                )
            }
        }
    }
}

@Composable
internal fun ToggleRow(
    label: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 56.dp)
            .padding(horizontal = 16.dp, vertical = 4.dp)
            .semantics { contentDescription = label },
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(label, modifier = Modifier.weight(1f), color = LocalCamBeige)
        Switch(checked = checked, onCheckedChange = onCheckedChange)
    }
}

@Composable
internal fun FormSurface(content: @Composable () -> Unit) {
    androidx.compose.material3.Surface(
        modifier = Modifier.fillMaxWidth(),
        color = LocalCamSubtle,
        shape = RoundedCornerShape(12.dp),
        tonalElevation = 1.dp,
        content = {
            Column(
                modifier = Modifier.fillMaxWidth(),
                content = { content() },
            )
        },
    )
}

@Composable
internal fun SectionDivider() {
    HorizontalDivider(color = LocalCamMuted.copy(alpha = 0.22f))
}
