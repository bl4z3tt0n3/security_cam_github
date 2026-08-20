package com.localsecuritycam.android.encoder

/** Keeps output-buffer release guaranteed while preserving the primary error. */
internal fun <T> withH264OutputBuffer(
    process: () -> T,
    release: () -> Unit,
    onReleaseError: (Throwable) -> Unit,
): T {
    var processFailure: Throwable? = null
    try {
        return process()
    } catch (error: Throwable) {
        processFailure = error
        throw error
    } finally {
        try {
            release()
        } catch (error: Throwable) {
            onReleaseError(error)
            if (processFailure == null) throw error
        }
    }
}
