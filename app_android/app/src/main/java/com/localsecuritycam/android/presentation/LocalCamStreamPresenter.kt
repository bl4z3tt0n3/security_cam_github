package com.localsecuritycam.android.presentation

import com.localsecuritycam.android.service.ServiceSnapshot

/**
 * Presentation-only controller. The service remains the sole authority for the
 * streaming lifecycle; this class only maps snapshots and forwards user intent.
 */
class LocalCamStreamPresenter(
    private val onStartRequested: () -> Unit,
    private val onStopRequested: () -> Unit,
    private val onStateChanged: (StreamingScreenState) -> Unit,
) {
    private var current: StreamingScreenState? = null

    fun onSnapshot(snapshot: ServiceSnapshot) {
        StreamingScreenStateMapper.map(snapshot).also {
            current = it
            onStateChanged(it)
        }
    }

    fun onStreamActionClicked() {
        when (current?.action) {
            StreamAction.START -> onStartRequested()
            StreamAction.STOP -> onStopRequested()
            StreamAction.NONE -> Unit
            null -> onStartRequested()
        }
    }
}
