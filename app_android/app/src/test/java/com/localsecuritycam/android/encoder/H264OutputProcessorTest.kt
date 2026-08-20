package com.localsecuritycam.android.encoder

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class H264OutputProcessorTest {
    @Test
    fun codecConfigOutputPublishesSpsAndPpsButNotAFrame() {
        val result = H264OutputProcessor.process(
            H264OutputSample(
                bytes = annexB(0x67, 1, 0x68, 2),
                ptsUs = 10,
                codecConfig = true,
                keyFrame = false,
            ),
        )

        assertNotNull(result.parameterSets)
        assertEquals(2, result.parameterSets?.sps?.size)
        assertEquals(2, result.parameterSets?.pps?.size)
        assertNull(result.accessUnit)
    }

    @Test
    fun realOutputCreatesOneAccessUnitAndDetectsIdr() {
        val result = H264OutputProcessor.process(
            H264OutputSample(
                bytes = annexB(0x65, 9, 0x41, 8),
                ptsUs = 20,
                codecConfig = false,
                keyFrame = false,
            ),
        )

        assertNull(result.parameterSets)
        assertNotNull(result.accessUnit)
        assertTrue(result.accessUnit!!.isKeyFrame)
        assertEquals(4, result.accessUnit!!.byteCount)
        assertEquals(20, result.accessUnit!!.ptsUs)
    }

    @Test
    fun emptyOutputDoesNotCreateAFrame() {
        val result = H264OutputProcessor.process(
            H264OutputSample(ByteArray(0), 0, codecConfig = false, keyFrame = false),
        )

        assertNull(result.parameterSets)
        assertNull(result.accessUnit)
    }

    @Test
    fun formatCodecSpecificDataProducesSpsAndPps() {
        val result = H264OutputProcessor.parameterSetsFromFormat(
            sps = byteArrayOf(0x67, 1),
            pps = byteArrayOf(0x68, 2),
        )

        assertNotNull(result)
        assertEquals(7, result?.sps?.first()?.toInt()?.and(0x1f))
        assertEquals(8, result?.pps?.first()?.toInt()?.and(0x1f))
    }

    private fun annexB(vararg values: Int): ByteArray = buildList {
        var index = 0
        while (index < values.size) {
            add(0)
            add(0)
            add(1)
            add(values[index++])
            if (index < values.size) add(values[index++])
        }
    }.map { it.toByte() }.toByteArray()
}
