package com.localsecuritycam.android.streaming

data class H264ParameterSets(
    val sps: ByteArray,
    val pps: ByteArray,
)

object H264NalParser {
    fun split(data: ByteArray): List<ByteArray> {
        if (data.isEmpty()) return emptyList()
        val firstStart = findStartCode(data, 0)
        if (firstStart >= 0) return splitAnnexB(data, firstStart)
        val lengthPrefixed = splitLengthPrefixed(data)
        return lengthPrefixed ?: listOf(data.copyOf())
    }

    fun parameterSets(nals: List<ByteArray>): H264ParameterSets? {
        val sps = nals.firstOrNull { nalType(it) == 7 }
        val pps = nals.firstOrNull { nalType(it) == 8 }
        return if (sps != null && pps != null) H264ParameterSets(sps.copyOf(), pps.copyOf()) else null
    }

    fun nalType(nal: ByteArray): Int = nal.firstOrNull()?.toInt()?.and(0x1f) ?: -1

    private fun splitAnnexB(data: ByteArray, firstStart: Int): List<ByteArray> {
        val result = mutableListOf<ByteArray>()
        var start = firstStart
        while (start >= 0) {
            val prefixLength = if (start + 2 < data.size && data[start] == 0.toByte() && data[start + 1] == 0.toByte() && data[start + 2] == 1.toByte()) 3 else 4
            val nalStart = start + prefixLength
            val next = findStartCode(data, nalStart)
            val end = if (next >= 0) next else data.size
            if (nalStart < end) result += data.copyOfRange(nalStart, end)
            start = next
        }
        return result
    }

    private fun splitLengthPrefixed(data: ByteArray): List<ByteArray>? {
        val result = mutableListOf<ByteArray>()
        var offset = 0
        while (offset + 4 <= data.size) {
            val length = ((data[offset].toInt() and 0xff) shl 24) or
                ((data[offset + 1].toInt() and 0xff) shl 16) or
                ((data[offset + 2].toInt() and 0xff) shl 8) or
                (data[offset + 3].toInt() and 0xff)
            offset += 4
            if (length <= 0 || length > data.size - offset) return null
            result += data.copyOfRange(offset, offset + length)
            offset += length
        }
        return if (offset == data.size && result.isNotEmpty()) result else null
    }

    private fun findStartCode(data: ByteArray, from: Int): Int {
        var index = from.coerceAtLeast(0)
        while (index + 2 < data.size) {
            if (data[index] == 0.toByte() && data[index + 1] == 0.toByte() &&
                data[index + 2] == 1.toByte()) return index
            if (index + 3 < data.size && data[index] == 0.toByte() && data[index + 1] == 0.toByte() &&
                data[index + 2] == 0.toByte() && data[index + 3] == 1.toByte()) return index
            index++
        }
        return -1
    }
}
