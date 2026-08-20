package com.localsecuritycam.android.streaming

import com.localsecuritycam.android.settings.StreamSettings
import com.localsecuritycam.android.settings.Resolution
import org.junit.Assert.assertTrue
import org.junit.Test

class SdpBuilderTest {
    @Test
    fun buildsH264SdpWithSpropParameterSets() {
        val sdp = SdpBuilder.build(
            StreamSettings(),
            H264ParameterSets(byteArrayOf(0x67, 0x42, 0x00, 0x1f), byteArrayOf(0x68, 0xce.toByte())),
        )
        assertTrue(sdp.contains("m=video 0 RTP/AVP 96"))
        assertTrue(sdp.contains("a=rtpmap:96 H264/90000"))
        assertTrue(sdp.contains("packetization-mode=1"))
        assertTrue(sdp.contains("sprop-parameter-sets="))
        assertTrue(sdp.contains("a=control:trackID=0"))
    }

    @Test
    fun advertisesPortraitFramesizeWhenTheEncoderIsPortrait() {
        val sdp = SdpBuilder.build(
            StreamSettings(
                resolution = Resolution(720, 1280),
                authEnabled = false,
            ),
            parameterSets = null,
        )

        assertTrue(sdp.contains("a=framesize:96 720-1280"))
    }
}
