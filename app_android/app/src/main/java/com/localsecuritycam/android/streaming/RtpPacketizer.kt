package com.localsecuritycam.android.streaming

import java.security.SecureRandom

data class RtpPacket(
    val bytes: ByteArray,
    val sequenceNumber: Int,
    val timestamp: Long,
    val marker: Boolean,
)

class RtpPacketizer(
    private val mtu: Int = 1200,
    private val payloadType: Int = 96,
    sequenceStart: Int = SecureRandom().nextInt(65_536),
    private val ssrc: Long = SecureRandom().nextInt().toLong() and 0xffffffffL,
) {
    init {
        require(mtu >= 64) { "RTP MTU is too small" }
        require(payloadType in 0..127) { "RTP payload type must be 0..127" }
    }

    private var sequence = sequenceStart and 0xffff

    fun nextSequenceNumber(): Int = sequence

    fun packetize(unit: EncodedAccessUnit): List<RtpPacket> {
        val timestamp = (unit.ptsUs * 90L / 1_000L).coerceAtLeast(0L) and 0xffffffffL
        val output = mutableListOf<RtpPacket>()
        val nals = unit.nals.filter { it.isNotEmpty() }
        nals.forEachIndexed { nalIndex, nal ->
            val isLastNal = nalIndex == nals.lastIndex
            if (nal.size <= mtu - RTP_HEADER_SIZE) {
                output += makePacket(nal, timestamp, marker = isLastNal)
            } else {
                val maxChunk = mtu - RTP_HEADER_SIZE - FU_A_HEADER_SIZE
                var offset = 1
                var first = true
                while (offset < nal.size) {
                    val count = minOf(maxChunk, nal.size - offset)
                    val lastFragment = offset + count >= nal.size
                    val payload = ByteArray(FU_A_HEADER_SIZE + count)
                    payload[0] = ((nal[0].toInt() and 0xe0) or 28).toByte()
                    payload[1] = ((if (first) 0x80 else 0) or
                        (if (lastFragment) 0x40 else 0) or
                        (nal[0].toInt() and 0x1f)).toByte()
                    nal.copyInto(payload, FU_A_HEADER_SIZE, offset, offset + count)
                    output += makePacket(payload, timestamp, marker = isLastNal && lastFragment)
                    first = false
                    offset += count
                }
            }
        }
        return output
    }

    private fun makePacket(payload: ByteArray, timestamp: Long, marker: Boolean): RtpPacket {
        val bytes = ByteArray(RTP_HEADER_SIZE + payload.size)
        bytes[0] = 0x80.toByte()
        bytes[1] = ((if (marker) 0x80 else 0) or payloadType).toByte()
        bytes[2] = (sequence ushr 8).toByte()
        bytes[3] = sequence.toByte()
        bytes[4] = (timestamp ushr 24).toByte()
        bytes[5] = (timestamp ushr 16).toByte()
        bytes[6] = (timestamp ushr 8).toByte()
        bytes[7] = timestamp.toByte()
        bytes[8] = (ssrc ushr 24).toByte()
        bytes[9] = (ssrc ushr 16).toByte()
        bytes[10] = (ssrc ushr 8).toByte()
        bytes[11] = ssrc.toByte()
        payload.copyInto(bytes, RTP_HEADER_SIZE)
        val result = RtpPacket(bytes, sequence, timestamp, marker)
        sequence = (sequence + 1) and 0xffff
        return result
    }

    private companion object {
        const val RTP_HEADER_SIZE = 12
        const val FU_A_HEADER_SIZE = 2
    }
}
