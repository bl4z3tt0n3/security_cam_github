package com.localsecuritycam.android.streaming

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RtpPacketizerTest {
    @Test
    fun packetizesSingleNalWithMarkerAndTimestamp() {
        val packetizer = RtpPacketizer(mtu = 1200, sequenceStart = 100)
        val packet = packetizer.packetize(EncodedAccessUnit(listOf(byteArrayOf(0x65, 1, 2)), 33_333, true)).single()
        assertEquals(100, packet.sequenceNumber)
        assertEquals(2_999L, packet.timestamp)
        assertTrue(packet.marker)
        assertEquals(0x80.toByte(), packet.bytes[0])
        assertEquals(96 or 0x80, packet.bytes[1].toInt() and 0xff)
    }

    @Test
    fun fragmentsLargeNalUsingFuAAndSetsStartEnd() {
        val nal = ByteArray(3_000) { index -> if (index == 0) 0x65 else (index and 0xff).toByte() }
        val packets = RtpPacketizer(mtu = 500, sequenceStart = 1)
            .packetize(EncodedAccessUnit(listOf(nal), 1_000, true))
        assertTrue(packets.size > 1)
        val firstPayload = packets.first().bytes.copyOfRange(12, packets.first().bytes.size)
        val lastPayload = packets.last().bytes.copyOfRange(12, packets.last().bytes.size)
        assertEquals(28, firstPayload[0].toInt() and 0x1f)
        assertTrue(firstPayload[1].toInt() and 0x80 != 0)
        assertTrue(lastPayload[1].toInt() and 0x40 != 0)
        assertTrue(packets.last().marker)
    }

    @Test
    fun sequenceNumbersWrapAtUnsignedSixteenBits() {
        val packetizer = RtpPacketizer(sequenceStart = 65_535)
        val units = listOf(
            packetizer.packetize(EncodedAccessUnit(listOf(byteArrayOf(0x41)), 0, false)).single(),
            packetizer.packetize(EncodedAccessUnit(listOf(byteArrayOf(0x41)), 1, false)).single(),
        )
        assertEquals(65_535, units[0].sequenceNumber)
        assertEquals(0, units[1].sequenceNumber)
    }
}
