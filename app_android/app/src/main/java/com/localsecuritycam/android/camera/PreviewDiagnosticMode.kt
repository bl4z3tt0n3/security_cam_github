package com.localsecuritycam.android.camera

/**
 * Debug-only render modes used to isolate the SurfaceView/OES/transform path.
 * NORMAL is the production path and must remain behaviorally unchanged.
 */
enum class PreviewDiagnosticMode(val wireValue: String) {
    NORMAL("normal"),
    PATTERN("pattern"),
    OES_IDENTITY("oes_identity"),
    OES_ROTATION("oes_rotation"),
    FULL("full"),
    ;

    companion object {
        fun fromWireValue(value: String?): PreviewDiagnosticMode =
            entries.firstOrNull { it.wireValue == value } ?: NORMAL
    }
}

const val PREVIEW_DIAGNOSTIC_EXTRA = "preview_diagnostic"
