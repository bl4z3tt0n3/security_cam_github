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
        val nals = unit.nals.filter { it.isNotEmpty() }
        val output = ArrayList<RtpPacket>(nals.size)
        nals.forEachIndexed { nalIndex, nal ->
            val isLastNal = nalIndex == nals.lastIndex
            if (nal.size <= mtu - RTP_HEADER_SIZE) {
                output += makeSingleNalPacket(nal, timestamp, marker = isLastNal)
            } else {
                val maxChunk = mtu - RTP_HEADER_SIZE - FU_A_HEADER_SIZE
                var offset = 1
                var first = true
                while (offset < nal.size) {
                    val count = minOf(maxChunk, nal.size - offset)
                    val lastFragment = offset + count >= nal.size
                    output += makeFuAPacket(
                        nal = nal,
                        sourceOffset = offset,
                        count = count,
                        first = first,
                        last = lastFragment,
                        timestamp = timestamp,
                        marker = isLastNal && lastFragment,
                    )
                    first = false
                    offset += count
                }
            }
        }
        return output
    }

    private fun makeSingleNalPacket(payload: ByteArray, timestamp: Long, marker: Boolean): RtpPacket {
        val bytes = ByteArray(RTP_HEADER_SIZE + payload.size)
        val packetSequence = writeRtpHeader(bytes, timestamp, marker)
        payload.copyInto(bytes, RTP_HEADER_SIZE)
        return RtpPacket(bytes, packetSequence, timestamp, marker)
    }

    private fun makeFuAPacket(
        nal: ByteArray,
        sourceOffset: Int,
        count: Int,
        first: Boolean,
        last: Boolean,
        timestamp: Long,
        marker: Boolean,
    ): RtpPacket {
        // Build directly into the final RTP packet. The previous implementation
        // allocated a temporary FU-A payload and then copied it again.
        val bytes = ByteArray(RTP_HEADER_SIZE + FU_A_HEADER_SIZE + count)
        val packetSequence = writeRtpHeader(bytes, timestamp, marker)
        bytes[RTP_HEADER_SIZE] = ((nal[0].toInt() and 0xe0) or 28).toByte()
        bytes[RTP_HEADER_SIZE + 1] = (
            (if (first) 0x80 else 0) or
                (if (last) 0x40 else 0) or
                (nal[0].toInt() and 0x1f)
            ).toByte()
        nal.copyInto(
            bytes,
            RTP_HEADER_SIZE + FU_A_HEADER_SIZE,
            sourceOffset,
            sourceOffset + count,
        )
        return RtpPacket(bytes, packetSequence, timestamp, marker)
    }

    private fun writeRtpHeader(bytes: ByteArray, timestamp: Long, marker: Boolean): Int {
        val packetSequence = sequence
        bytes[0] = 0x80.toByte()
        bytes[1] = ((if (marker) 0x80 else 0) or payloadType).toByte()
        bytes[2] = (packetSequence ushr 8).toByte()
        bytes[3] = packetSequence.toByte()
        bytes[4] = (timestamp ushr 24).toByte()
        bytes[5] = (timestamp ushr 16).toByte()
        bytes[6] = (timestamp ushr 8).toByte()
        bytes[7] = timestamp.toByte()
        bytes[8] = (ssrc ushr 24).toByte()
        bytes[9] = (ssrc ushr 16).toByte()
        bytes[10] = (ssrc ushr 8).toByte()
        bytes[11] = ssrc.toByte()
        sequence = (sequence + 1) and 0xffff
        return packetSequence
    }

    private companion object {
        const val RTP_HEADER_SIZE = 12
        const val FU_A_HEADER_SIZE = 2
    }
}
