package com.localsecuritycam.android.streaming

import com.localsecuritycam.android.diagnostics.CleanupReport
import com.localsecuritycam.android.diagnostics.StreamMetrics
import java.util.concurrent.CopyOnWriteArraySet
import java.util.concurrent.TimeUnit
import java.util.concurrent.locks.ReentrantLock

interface AccessUnitSink {
    fun enqueue(unit: EncodedAccessUnit): Boolean
    fun closeSink(): CleanupReport
}

class StreamBroadcaster(private val metrics: StreamMetrics) {
    private val sinks = CopyOnWriteArraySet<AccessUnitSink>()
    private val lock = ReentrantLock()
    private val parameterSetsReady = lock.newCondition()
    private var parameterSets: H264ParameterSets? = null

    fun reset() {
        lock.lock()
        try {
            parameterSets = null
            parameterSetsReady.signalAll()
        } finally {
            lock.unlock()
        }
    }

    fun setParameterSets(value: H264ParameterSets) {
        lock.lock()
        try {
            parameterSets = H264ParameterSets(value.sps.copyOf(), value.pps.copyOf())
            parameterSetsReady.signalAll()
        } finally {
            lock.unlock()
        }
    }

    fun awaitParameterSets(timeoutMs: Long): H264ParameterSets? {
        require(timeoutMs >= 0) { "parameter-set timeout cannot be negative" }
        var remainingNs = TimeUnit.MILLISECONDS.toNanos(timeoutMs)
        lock.lock()
        try {
            while (parameterSets == null) {
                if (remainingNs <= 0) break
                remainingNs = parameterSetsReady.awaitNanos(remainingNs)
            }
            return parameterSets?.let { H264ParameterSets(it.sps.copyOf(), it.pps.copyOf()) }
        } finally {
            lock.unlock()
        }
    }

    fun addSink(sink: AccessUnitSink) {
        sinks += sink
    }

    fun removeSink(sink: AccessUnitSink) {
        sinks -= sink
    }

    fun publish(unit: EncodedAccessUnit) {
        val prepared = if (unit.isKeyFrame) addMissingParameterSets(unit) else unit
        sinks.forEach { sink ->
            if (!sink.enqueue(prepared)) metrics.recordDroppedFrame()
        }
    }

    private fun addMissingParameterSets(unit: EncodedAccessUnit): EncodedAccessUnit {
        val hasSps = unit.nals.any { H264NalParser.nalType(it) == 7 }
        val hasPps = unit.nals.any { H264NalParser.nalType(it) == 8 }
        if (hasSps && hasPps) return unit

        val sets = run {
            lock.lock()
            try {
                parameterSets
            } finally {
                lock.unlock()
            }
        } ?: return unit
        val prefix = buildList {
            if (!hasSps) add(sets.sps.copyOf())
            if (!hasPps) add(sets.pps.copyOf())
        }
        return unit.copy(nals = prefix + unit.nals)
    }
}
