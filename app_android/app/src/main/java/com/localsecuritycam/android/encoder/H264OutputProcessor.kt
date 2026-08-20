package com.localsecuritycam.android.encoder

import com.localsecuritycam.android.streaming.EncodedAccessUnit
import com.localsecuritycam.android.streaming.H264NalParser
import com.localsecuritycam.android.streaming.H264ParameterSets

internal data class H264OutputSample(
    val bytes: ByteArray,
    val ptsUs: Long,
    val codecConfig: Boolean,
    val keyFrame: Boolean,
)

internal data class H264OutputResult(
    val parameterSets: H264ParameterSets?,
    val accessUnit: EncodedAccessUnit?,
)

/** Parses MediaCodec output without depending on Android MediaCodec classes. */
internal object H264OutputProcessor {
    fun process(sample: H264OutputSample): H264OutputResult {
        val nals = H264NalParser.split(sample.bytes)
        val parameterSets = H264NalParser.parameterSets(nals)
        val accessUnit = if (!sample.codecConfig && nals.isNotEmpty()) {
            val isKeyFrame = sample.keyFrame || nals.any { H264NalParser.nalType(it) == 5 }
            EncodedAccessUnit(nals, sample.ptsUs, isKeyFrame)
        } else {
            null
        }
        return H264OutputResult(parameterSets, accessUnit)
    }

    fun parameterSetsFromFormat(sps: ByteArray?, pps: ByteArray?): H264ParameterSets? {
        val spsNal = parameterSetFromBytes(sps, 7) ?: return null
        val ppsNal = parameterSetFromBytes(pps, 8) ?: return null
        return H264ParameterSets(spsNal, ppsNal)
    }

    private fun parameterSetFromBytes(bytes: ByteArray?, expectedType: Int): ByteArray? {
        val source = bytes ?: return null
        if (source.isEmpty()) return null
        val nals = H264NalParser.split(source)
        return nals.firstOrNull { H264NalParser.nalType(it) == expectedType }?.copyOf()
    }
}
