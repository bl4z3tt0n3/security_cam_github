package com.localsecuritycam.android.streaming

data class EncodedAccessUnit(
    val nals: List<ByteArray>,
    val ptsUs: Long,
    val isKeyFrame: Boolean,
) {
    val byteCount: Int get() = nals.sumOf { it.size }
}
