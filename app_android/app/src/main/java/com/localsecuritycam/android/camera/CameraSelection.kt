package com.localsecuritycam.android.camera

import com.localsecuritycam.android.settings.CameraLens

/** Selects the first camera whose reported facing matches the requested lens. */
internal fun selectCameraId(
    cameraIds: List<String>,
    lensForCamera: (String) -> CameraLens?,
    requestedLens: CameraLens,
): String? = cameraIds.firstOrNull { cameraId -> lensForCamera(cameraId) == requestedLens }
