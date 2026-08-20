package com.localsecuritycam.android.camera

import android.view.Surface

data class PreviewSurfaceAttachment(
    val surface: Surface,
    val width: Int,
    val height: Int,
)

internal enum class PreviewSurfaceUpdateKind {
    DEFERRED,
    APPLY,
    IGNORED,
}

internal data class PreviewSurfaceUpdate<T>(
    val kind: PreviewSurfaceUpdateKind,
    val surface: T? = null,
    val generation: Long = 0L,
    val sequence: Long = 0L,
)

/**
 * Keeps preview-surface changes ordered across renderer initialization and stop.
 * It contains no Android or EGL code so the lifecycle race is JVM-testable.
 */
internal class PreviewSurfaceBinding<T> {
    private enum class State {
        NOT_READY,
        READY,
        STOPPED,
    }

    private var state = State.NOT_READY
    private var hasPendingSurface = false
    private var pendingSurface: T? = null
    private var generation = 0L
    private var sequence = 0L
    private var pendingSequence = 0L

    @Synchronized
    fun begin() {
        generation++
        if (state == State.STOPPED) {
            hasPendingSurface = false
            pendingSurface = null
        }
        pendingSequence = 0L
        state = State.NOT_READY
    }

    @Synchronized
    fun request(surface: T?): PreviewSurfaceUpdate<T> {
        if (state == State.STOPPED) return PreviewSurfaceUpdate(PreviewSurfaceUpdateKind.IGNORED)
        pendingSequence = ++sequence
        pendingSurface = surface
        hasPendingSurface = true
        return if (state == State.READY) {
            PreviewSurfaceUpdate(
                kind = PreviewSurfaceUpdateKind.APPLY,
                surface = surface,
                generation = generation,
                sequence = pendingSequence,
            )
        } else {
            PreviewSurfaceUpdate(
                kind = PreviewSurfaceUpdateKind.DEFERRED,
                generation = generation,
                sequence = pendingSequence,
            )
        }
    }

    @Synchronized
    fun markReady(): PreviewSurfaceUpdate<T> {
        if (state == State.STOPPED) return PreviewSurfaceUpdate(PreviewSurfaceUpdateKind.IGNORED)
        state = State.READY
        return if (hasPendingSurface) {
            PreviewSurfaceUpdate(
                kind = PreviewSurfaceUpdateKind.APPLY,
                surface = pendingSurface,
                generation = generation,
                sequence = pendingSequence,
            )
        } else {
            PreviewSurfaceUpdate(
                kind = PreviewSurfaceUpdateKind.DEFERRED,
                generation = generation,
                sequence = pendingSequence,
            )
        }
    }

    /** Latest requested target, used to recover an EGL target without waiting
     * for a second SurfaceHolder callback. */
    @Synchronized
    fun currentSurface(): T? = if (state == State.READY && hasPendingSurface) pendingSurface else null

    @Synchronized
    fun isCurrent(update: PreviewSurfaceUpdate<T>): Boolean =
        state != State.STOPPED &&
            update.generation == generation &&
            update.sequence != 0L &&
            update.sequence == pendingSequence

    @Synchronized
    fun stop() {
        state = State.STOPPED
        hasPendingSurface = false
        pendingSurface = null
        pendingSequence = 0L
    }
}
