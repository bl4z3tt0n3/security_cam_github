package com.localsecuritycam.android.diagnostics

class StreamStateMachine(initial: StreamState = StreamState.STOPPED) {
    private var current = initial

    @get:Synchronized
    val state: StreamState
        get() = current

    @Synchronized
    fun transitionTo(next: StreamState) {
        if (next == current) return
        require(next in allowed[current].orEmpty()) {
            "invalid stream state transition ${current.name} -> ${next.name}"
        }
        current = next
    }

    private companion object {
        val allowed = mapOf(
            StreamState.STOPPED to setOf(StreamState.STARTING),
            StreamState.WAITING_NETWORK to setOf(
                StreamState.STARTING,
                StreamState.STOPPED,
                StreamState.STOPPING,
                StreamState.ERROR,
            ),
            StreamState.STARTING to setOf(
                StreamState.WAITING_NETWORK,
                StreamState.STREAMING,
                StreamState.STOPPING,
                StreamState.ERROR,
            ),
            StreamState.STREAMING to setOf(StreamState.STOPPING, StreamState.ERROR),
            StreamState.STOPPING to setOf(
                StreamState.STOPPED,
                StreamState.WAITING_NETWORK,
                StreamState.ERROR,
            ),
            StreamState.ERROR to setOf(StreamState.STARTING, StreamState.STOPPED),
        )
    }
}
