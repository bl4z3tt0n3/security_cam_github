package com.localsecuritycam.android.streaming

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class H264NalParserTest {
    @Test
    fun splitsAnnexBAccessUnit() {
        val value = byteArrayOf(0, 0, 0, 1, 0x67, 1, 2, 0, 0, 1, 0x68, 3)
        val nals = H264NalParser.split(value)
        assertEquals(2, nals.size)
        assertEquals(7, H264NalParser.nalType(nals[0]))
        assertEquals(8, H264NalParser.nalType(nals[1]))
        assertNotNull(H264NalParser.parameterSets(nals))
    }

    @Test
    fun splitsFourByteLengthPrefixedAccessUnit() {
        val value = byteArrayOf(0, 0, 0, 2, 0x65, 9, 0, 0, 0, 2, 0x41, 8)
        val nals = H264NalParser.split(value)
        assertEquals(2, nals.size)
        assertEquals(5, H264NalParser.nalType(nals[0]))
        assertTrue(nals[1].contentEquals(byteArrayOf(0x41, 8)))
    }

    @Test
    fun acceptsTheShortestAnnexBStartCode() {
        val nals = H264NalParser.split(byteArrayOf(0, 0, 1, 0x65, 7))
        assertEquals(1, nals.size)
        assertEquals(5, H264NalParser.nalType(nals.single()))
    }

    @Test
    fun rejectsOverflowingLengthPrefixedNalUnit() {
        val malformed = byteArrayOf(0x7f, 0xff.toByte(), 0xff.toByte(), 0xff.toByte(), 0x65)
        assertTrue(H264NalParser.split(malformed).single().contentEquals(malformed))
    }
}
