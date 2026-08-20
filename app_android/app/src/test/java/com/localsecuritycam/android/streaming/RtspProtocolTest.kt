package com.localsecuritycam.android.streaming

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.charset.StandardCharsets
import java.util.Base64

class RtspProtocolTest {
    @Test
    fun parsesRequestAndHeadersCaseInsensitively() {
        val request = RtspProtocol.parse(
            "DESCRIBE rtsp://192.168.1.20:8554/stream RTSP/1.0\r\n" +
                "cSeQ: 7\r\nAccept: application/sdp\r\n\r\n",
        )
        assertEquals("DESCRIBE", request.method)
        assertEquals("7", request.header("CSeq"))
        assertEquals("application/sdp", request.header("accept"))
        assertTrue(RtspProtocol.pathMatches(request.uri, "/stream"))
    }

    @Test
    fun rejectsDuplicateHeadersThatCouldMakeFramingAmbiguous() {
        org.junit.Assert.assertThrows(IllegalArgumentException::class.java) {
            RtspProtocol.parse(
                "OPTIONS rtsp://camera/stream RTSP/1.0\r\n" +
                    "Content-Length: 0\r\ncontent-length: 1\r\n\r\n",
            )
        }
    }

    @Test
    fun parsesOnlyInterleavedTcpTransport() {
        val transport = RtspProtocol.parseTransport("RTP/AVP/TCP;unicast;interleaved=4-5")
        assertEquals(4, transport?.interleavedRtpChannel)
        assertEquals(5, transport?.interleavedRtcpChannel)
        assertEquals(null, RtspProtocol.parseTransport("RTP/AVP;unicast;client_port=5000-5001"))
        assertEquals(null, RtspProtocol.parseTransport("RTP/AVP/TCP;interleaved=256-257"))
        assertEquals(null, RtspProtocol.parseTransport("RTP/AVP/TCP;interleaved=4-4"))
    }

    @Test
    fun createsResponseWithCseqAndBodyLength() {
        val response = RtspProtocol.response(200, "OK", "11", mapOf("Content-Type" to "text/plain"), "ok")
        assertTrue(response.startsWith("RTSP/1.0 200 OK\r\nCSeq: 11\r\n"))
        assertTrue(response.contains("Content-Length: 2\r\n"))
        assertTrue(response.endsWith("\r\nok"))
    }

    @Test
    fun validatesBasicAuthenticationWithoutLoggingSecrets() {
        val credentials = RtspCredentials(true, "admin", "very-secret")
        val header = "Basic " + Base64.getEncoder().encodeToString("admin:very-secret".toByteArray(StandardCharsets.UTF_8))
        assertTrue(RtspAuth.isAuthorized(header, credentials))
        assertFalse(RtspAuth.isAuthorized(null, credentials))
        assertFalse(RtspAuth.isAuthorized("Basic invalid", credentials))
        assertTrue(RtspAuth.isAuthorized(null, credentials.copy(enabled = false)))
    }

    @Test
    fun rejectsTrackFromAnotherPath() {
        assertTrue(RtspProtocol.pathMatches("rtsp://camera/stream/trackID=0", "/stream"))
        assertFalse(RtspProtocol.pathMatches("rtsp://camera/other/trackID=0", "/stream"))
    }

    @Test
    fun stripsCredentialsFromResponseUri() {
        assertEquals(
            "rtsp://camera/stream",
            RtspProtocol.withoutCredentials("rtsp://admin:wire-secret@camera/stream?password=query-secret"),
        )
        assertEquals(
            "rtsp://camera/stream",
            RtspProtocol.withoutCredentials("rtsp://admin:pa%2Fss@camera/stream"),
        )
    }
}
