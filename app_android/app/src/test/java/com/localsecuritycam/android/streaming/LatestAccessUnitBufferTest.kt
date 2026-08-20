package com.localsecuritycam.android.streaming

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class LatestAccessUnitBufferTest {
    @Test
    fun capacityPressureDiscardsOldUnitsAndRetainsTheNewestIdr() {
        val buffer = LatestAccessUnitBuffer(capacity = 2, maxQueueBytes = 1_024, maxSingleUnitBytes = 1_024)

        assertTrue(buffer.offer(unit(1, keyFrame = true)).accepted)
        assertTrue(buffer.offer(unit(2, keyFrame = false)).accepted)

        val replacement = buffer.offer(unit(3, keyFrame = true))

        assertTrue(replacement.accepted)
        assertEquals(2, replacement.droppedUnits)
        assertEquals(3L, buffer.take().ptsUs)
    }

    @Test
    fun keyframeLossDropsDependentFramesUntilAnotherIdrArrives() {
        val buffer = LatestAccessUnitBuffer(capacity = 4, maxQueueBytes = 7, maxSingleUnitBytes = 64)

        assertTrue(buffer.offer(unit(1, keyFrame = true, payloadSize = 4)).accepted)
        assertTrue(buffer.offer(unit(2, keyFrame = false, payloadSize = 3)).accepted)

        val dependent = buffer.offer(unit(3, keyFrame = false, payloadSize = 4))
        assertTrue(dependent.accepted)
        assertEquals(3, dependent.droppedUnits)

        assertTrue(buffer.offer(unit(4, keyFrame = true, payloadSize = 4)).accepted)
        assertEquals(4L, buffer.take().ptsUs)
    }

    @Test
    fun rejectsOneUnitThatExceedsTheByteBound() {
        val buffer = LatestAccessUnitBuffer(capacity = 2, maxQueueBytes = 8, maxSingleUnitBytes = 64)

        val outcome = buffer.offer(unit(1, keyFrame = true, payloadSize = 9))

        assertFalse(outcome.accepted)
        assertEquals(0, outcome.droppedUnits)
    }

    private fun unit(
        timeUs: Int,
        keyFrame: Boolean,
        payloadSize: Int = 2,
    ): EncodedAccessUnit {
        val nal = ByteArray(payloadSize) { index ->
            when (index) {
                0 -> if (keyFrame) 0x65.toByte() else 0x41.toByte()
                else -> timeUs.toByte()
            }
        }
        return EncodedAccessUnit(listOf(nal), timeUs.toLong(), keyFrame)
    }
}
