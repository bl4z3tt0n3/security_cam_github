package com.localsecuritycam.android.ui.theme

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.unit.dp

private val LocalCamDarkColorScheme = darkColorScheme(
    primary = LocalCamAction,
    onPrimary = LocalCamCreamLight,
    primaryContainer = LocalCamActionDeep,
    onPrimaryContainer = LocalCamCreamLight,
    secondary = LocalCamBeige,
    onSecondary = LocalCamShell,
    secondaryContainer = LocalCamSubtle,
    onSecondaryContainer = LocalCamCream,
    tertiary = LocalCamGreen,
    onTertiary = LocalCamShell,
    background = LocalCamShell,
    onBackground = LocalCamCreamLight,
    surface = LocalCamSurface,
    onSurface = LocalCamCreamLight,
    surfaceVariant = LocalCamSubtle,
    onSurfaceVariant = LocalCamBeige,
    outline = LocalCamBorder,
    error = LocalCamRed,
    onError = LocalCamCreamLight,
)

private val LocalCamShapes = Shapes(
    extraSmall = RoundedCornerShape(6.dp),
    small = RoundedCornerShape(10.dp),
    medium = RoundedCornerShape(14.dp),
    large = RoundedCornerShape(20.dp),
)

@Composable
internal fun LocalCamTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = LocalCamDarkColorScheme,
        typography = LocalCamTypography,
        shapes = LocalCamShapes,
        content = content,
    )
}

