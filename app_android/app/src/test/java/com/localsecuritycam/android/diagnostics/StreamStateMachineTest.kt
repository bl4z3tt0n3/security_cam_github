package com.localsecuritycam.android.diagnostics

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Collections
import java.util.concurrent.CountDownLatch

class StreamStateMachineTest {
    @Test
    fun startsStopped() {
        assertEquals(StreamState.STOPPED, StreamStateMachine().state)
    }

    @Test
    fun followsStartSuccessAndStopLifecycle() {
        val machine = StreamStateMachine()

        machine.transitionTo(StreamState.STARTING)
        machine.transitionTo(StreamState.STREAMING)
        machine.transitionTo(StreamState.STOPPING)
        machine.transitionTo(StreamState.STOPPED)

        assertEquals(StreamState.STOPPED, machine.state)
    }

    @Test
    fun followsStartFailureAndRetry() {
        val machine = StreamStateMachine()

        machine.transitionTo(StreamState.STARTING)
        machine.transitionTo(StreamState.ERROR)
        machine.transitionTo(StreamState.STARTING)

        assertEquals(StreamState.STARTING, machine.state)
    }

    @Test
    fun supportsWaitingForWifiAndResuming() {
        val machine = StreamStateMachine()

        machine.transitionTo(StreamState.STARTING)
        machine.transitionTo(StreamState.WAITING_NETWORK)
        machine.transitionTo(StreamState.STARTING)
        machine.transitionTo(StreamState.STREAMING)

        assertEquals(StreamState.STREAMING, machine.state)
    }

    @Test
    fun failedTransitionDoesNotChangeState() {
        val machine = StreamStateMachine()

        assertThrows(IllegalArgumentException::class.java) {
            machine.transitionTo(StreamState.STREAMING)
        }

        assertEquals(StreamState.STOPPED, machine.state)
    }

    @Test
    fun rejectsInvalidShortcuts() {
        val machine = StreamStateMachine(StreamState.STREAMING)

        assertThrows(IllegalArgumentException::class.java) {
            machine.transitionTo(StreamState.STOPPED)
        }
        assertEquals(StreamState.STREAMING, machine.state)

        machine.transitionTo(StreamState.STOPPING)
        assertThrows(IllegalArgumentException::class.java) {
            machine.transitionTo(StreamState.STREAMING)
        }
        assertEquals(StreamState.STOPPING, machine.state)
    }

    @Test
    fun duplicateStartAndStopAreIdempotent() {
        val machine = StreamStateMachine()

        machine.transitionTo(StreamState.STARTING)
        machine.transitionTo(StreamState.STARTING)
        machine.transitionTo(StreamState.STREAMING)
        machine.transitionTo(StreamState.STOPPING)
        machine.transitionTo(StreamState.STOPPING)
        machine.transitionTo(StreamState.STOPPED)
        machine.transitionTo(StreamState.STOPPED)

        assertEquals(StreamState.STOPPED, machine.state)
    }

    @Test
    fun concurrentValidCyclesKeepStateConsistent() {
        val machine = StreamStateMachine()
        val failures = Collections.synchronizedList(mutableListOf<Throwable>())
        val ready = CountDownLatch(1)
        val writer = Thread {
            try {
                ready.await()
                repeat(250) {
                    machine.transitionTo(StreamState.STARTING)
                    machine.transitionTo(StreamState.STREAMING)
                    machine.transitionTo(StreamState.STOPPING)
                    machine.transitionTo(StreamState.STOPPED)
                }
            } catch (error: Throwable) {
                failures += error
            }
        }
        val reader = Thread {
            try {
                ready.await()
                repeat(5_000) {
                    assertTrue(machine.state in StreamState.entries)
                }
            } catch (error: Throwable) {
                failures += error
            }
        }

        writer.start()
        reader.start()
        ready.countDown()
        writer.join(5_000)
        reader.join(5_000)

        assertTrue(failures.isEmpty())
        assertEquals(StreamState.STOPPED, machine.state)
    }
}
