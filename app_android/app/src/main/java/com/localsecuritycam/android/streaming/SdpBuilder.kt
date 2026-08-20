package com.localsecuritycam.android.streaming

import com.localsecuritycam.android.settings.StreamSettings
import java.util.Base64

object SdpBuilder {
    fun build(settings: StreamSettings, parameterSets: H264ParameterSets?): String {
        val profile = parameterSets?.sps?.let { profileLevelId(it) } ?: "42e01f"
        val sprop = parameterSets?.let {
            ";sprop-parameter-sets=${Base64.getEncoder().encodeToString(it.sps)},${Base64.getEncoder().encodeToString(it.pps)}"
        }.orEmpty()
        return buildString {
            append("v=0\r\n")
            append("o=- 0 0 IN IP4 0.0.0.0\r\n")
            append("s=Huawei LAN Camera\r\n")
            append("t=0 0\r\n")
            append("a=control:*")
            append("\r\n")
            append("m=video 0 RTP/AVP 96\r\n")
            append("c=IN IP4 0.0.0.0\r\n")
            append("a=rtpmap:96 H264/90000\r\n")
            append("a=fmtp:96 packetization-mode=1;profile-level-id=")
            append(profile).append(sprop).append("\r\n")
            append("a=framerate:").append(settings.fps).append("\r\n")
            append("a=framesize:96 ").append(settings.resolution.width).append('-').append(settings.resolution.height).append("\r\n")
            append("a=control:trackID=0\r\n")
        }
    }

    private fun profileLevelId(sps: ByteArray): String {
        val bytes = if (sps.firstOrNull()?.toInt()?.and(0xff) == 0) sps.dropWhile { it == 0.toByte() }.toByteArray() else sps
        return if (bytes.size >= 4) {
            "%02x%02x%02x".format(bytes[1].toInt() and 0xff, bytes[2].toInt() and 0xff, bytes[3].toInt() and 0xff)
        } else {
            "42e01f"
        }
    }
}
