package com.localsecuritycam.android.streaming

import com.localsecuritycam.android.diagnostics.StreamMetrics
import com.localsecuritycam.android.settings.StreamSettings
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.assertThrows
import org.junit.Test
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.IOException
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.nio.charset.StandardCharsets
import java.util.Base64

class RtspServerIntegrationTest {
    @Test
    fun completesTcpHandshakeAndReceivesInterleavedRtp() {
        val port = freePort()
        val settings = StreamSettings(port = port, authEnabled = false)
        val metrics = StreamMetrics()
        val broadcaster = StreamBroadcaster(metrics)
        val server = RtspServer(settings, metrics, broadcaster, { null }) { error -> throw AssertionError(error) }
        try {
            server.start(InetAddress.getLoopbackAddress())
            broadcaster.setParameterSets(H264ParameterSets(byteArrayOf(0x67, 0x42, 0x00, 0x1f), byteArrayOf(0x68, 0xce.toByte())))
            Socket("127.0.0.1", port).use { socket ->
                socket.soTimeout = 3_000
                val input = BufferedInputStream(socket.getInputStream())
                val output = BufferedOutputStream(socket.getOutputStream())
                write(output, "OPTIONS rtsp://127.0.0.1:$port/stream RTSP/1.0\r\nCSeq: 1\r\n\r\n")
                assertTrue(readResponse(input).startsWith("RTSP/1.0 200"))
                write(output, "DESCRIBE rtsp://127.0.0.1:$port/stream RTSP/1.0\r\nCSeq: 2\r\n\r\n")
                assertTrue(readResponse(input).contains("a=rtpmap:96 H264/90000"))
                write(output, "SETUP rtsp://127.0.0.1:$port/stream/trackID=0 RTSP/1.0\r\nCSeq: 3\r\nTransport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n\r\n")
                val setupResponse = readResponse(input)
                assertTrue(setupResponse.startsWith("RTSP/1.0 200"))
                val session = Regex("(?im)^Session:\\s*([^;\\r\\n]+)").find(setupResponse)!!.groupValues[1]
                write(output, "PLAY rtsp://127.0.0.1:$port/stream RTSP/1.0\r\nCSeq: 4\r\nSession: $session\r\n\r\n")
                assertTrue(readResponse(input).startsWith("RTSP/1.0 200"))
                broadcaster.publish(
                    EncodedAccessUnit(
                        listOf(byteArrayOf(0x67, 0x42, 0, 0x1f), byteArrayOf(0x68, 0), byteArrayOf(0x65, 1, 2)),
                        0,
                        true,
                    ),
                )
                val marker = input.read()
                assertEquals('$'.code, marker)
                assertEquals(0, input.read())
                val packetLength = (input.read() shl 8) or input.read()
                assertTrue(packetLength >= 14)
                val packet = ByteArray(packetLength)
                readFully(input, packet)
                assertEquals(0x80, packet[0].toInt() and 0xff)
                assertEquals(0x67, packet[12].toInt() and 0xff)
            }
        } finally {
            server.stop()
        }
    }

    @Test
    fun reportsPortConflictWithoutLeavingServerRunning() {
        val socket = ServerSocket(0, 1, InetAddress.getLoopbackAddress())
        val settings = StreamSettings(port = socket.localPort, authEnabled = false)
        val metrics = StreamMetrics()
        val server = RtspServer(settings, metrics, StreamBroadcaster(metrics), { null }) { }
        try {
            var failed = false
            try {
                server.start(InetAddress.getLoopbackAddress())
            } catch (_: Exception) {
                failed = true
            }
            assertTrue(failed)
        } finally {
            socket.close()
        }
        server.start(InetAddress.getLoopbackAddress())
        server.stop()
    }

    @Test
    fun teardownClosesOnlyItsSessionAndKeepsTheListenerAvailable() {
        val port = freePort()
        val settings = StreamSettings(port = port, authEnabled = false)
        val metrics = StreamMetrics()
        val server = RtspServer(settings, metrics, StreamBroadcaster(metrics), { null }) { error -> throw AssertionError(error) }
        try {
            server.start(InetAddress.getLoopbackAddress())
            Socket("127.0.0.1", port).use { socket ->
                socket.soTimeout = 3_000
                val input = BufferedInputStream(socket.getInputStream())
                val output = BufferedOutputStream(socket.getOutputStream())
                write(output, "SETUP rtsp://127.0.0.1:$port/stream/trackID=0 RTSP/1.0\r\nCSeq: 1\r\nTransport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n\r\n")
                val setupResponse = readResponse(input)
                assertTrue(setupResponse.startsWith("RTSP/1.0 200"))
                val session = Regex("(?im)^Session:\\s*([^;\\r\\n]+)").find(setupResponse)!!.groupValues[1]
                write(output, "PLAY rtsp://127.0.0.1:$port/stream RTSP/1.0\r\nCSeq: 2\r\nSession: $session\r\n\r\n")
                assertTrue(readResponse(input).startsWith("RTSP/1.0 200"))
                write(output, "TEARDOWN rtsp://127.0.0.1:$port/stream RTSP/1.0\r\nCSeq: 3\r\nSession: $session\r\n\r\n")
                assertTrue(readResponse(input).startsWith("RTSP/1.0 200"))
            }
            Socket("127.0.0.1", port).use { socket ->
                socket.soTimeout = 3_000
                val input = BufferedInputStream(socket.getInputStream())
                val output = BufferedOutputStream(socket.getOutputStream())
                write(output, "OPTIONS rtsp://127.0.0.1:$port/stream RTSP/1.0\r\nCSeq: 4\r\n\r\n")
                assertTrue(readResponse(input).startsWith("RTSP/1.0 200"))
            }
        } finally {
            server.stop()
        }
    }

    @Test
    fun acceptsAnUnauthenticatedNonLoopbackListenerWhenExplicitlyConfigured() {
        val port = freePort()
        val settings = StreamSettings(port = port, authEnabled = false)
        val metrics = StreamMetrics()
        val server = RtspServer(settings, metrics, StreamBroadcaster(metrics), { null }) { }
        try {
            server.start(InetAddress.getByName("0.0.0.0"))
            Socket("127.0.0.1", port).use { socket ->
                socket.soTimeout = 3_000
                val input = BufferedInputStream(socket.getInputStream())
                val output = BufferedOutputStream(socket.getOutputStream())
                write(output, "OPTIONS rtsp://127.0.0.1:$port/stream RTSP/1.0\r\nCSeq: 1\r\n\r\n")
                assertTrue(readResponse(input).startsWith("RTSP/1.0 200"))
            }
        } finally {
            server.stop()
        }
    }

    @Test
    fun requiresBasicAuthAndSurvivesMalformedPeerInput() {
        val port = freePort()
        val settings = StreamSettings(port = port, authEnabled = true, username = "huawei")
        val metrics = StreamMetrics()
        val server = RtspServer(
            settings,
            metrics,
            StreamBroadcaster(metrics),
            { RtspCredentials(enabled = true, username = "huawei", password = "local-secret") },
        ) { error -> throw AssertionError(error) }
        try {
            server.start(InetAddress.getLoopbackAddress())
            Socket("127.0.0.1", port).use { malformed ->
                malformed.getOutputStream().write("NOT-AN-RTSP-REQUEST\r\n\r\n".toByteArray(StandardCharsets.UTF_8))
                malformed.getOutputStream().flush()
            }
            Socket("127.0.0.1", port).use { socket ->
                socket.soTimeout = 3_000
                val input = BufferedInputStream(socket.getInputStream())
                val output = BufferedOutputStream(socket.getOutputStream())
                write(output, "OPTIONS rtsp://127.0.0.1:$port/stream RTSP/1.0\r\nCSeq: 1\r\n\r\n")
                val unauthorized = readResponse(input)
                assertTrue(unauthorized.startsWith("RTSP/1.0 401"))
                assertTrue(unauthorized.contains("WWW-Authenticate: Basic"))

                write(
                    output,
                    "OPTIONS rtsp://127.0.0.1:$port/stream RTSP/1.0\r\n" +
                        "CSeq: 2\r\nAuthorization: Basic invalid-base64\r\n\r\n",
                )
                assertTrue(readResponse(input).startsWith("RTSP/1.0 401"))

                val authorization = Base64.getEncoder().encodeToString(
                    "huawei:local-secret".toByteArray(StandardCharsets.UTF_8),
                )
                write(
                    output,
                    "OPTIONS rtsp://127.0.0.1:$port/stream RTSP/1.0\r\n" +
                        "CSeq: 3\r\nAuthorization: Basic $authorization\r\n\r\n",
                )
                assertTrue(readResponse(input).startsWith("RTSP/1.0 200"))
            }
        } finally {
            server.stop()
        }
    }

    @Test
    fun clientDisconnectDoesNotStopServer() {
        val port = freePort()
        val settings = StreamSettings(port = port, authEnabled = false)
        val metrics = StreamMetrics()
        val server = RtspServer(settings, metrics, StreamBroadcaster(metrics), { null }) { error -> throw AssertionError(error) }
        try {
            server.start(InetAddress.getLoopbackAddress())
            Socket("127.0.0.1", port).use { socket ->
                val input = BufferedInputStream(socket.getInputStream())
                val output = BufferedOutputStream(socket.getOutputStream())
                write(output, "OPTIONS rtsp://127.0.0.1:$port/stream RTSP/1.0\r\nCSeq: 1\r\n\r\n")
                assertTrue(readResponse(input).startsWith("RTSP/1.0 200"))
            }
            assertTrue(awaitConnectedClients(metrics, 0))
            var connected = false
            repeat(10) {
                if (connected) return@repeat
                runCatching {
                    Socket("127.0.0.1", port).use { socket ->
                        val input = BufferedInputStream(socket.getInputStream())
                        val output = BufferedOutputStream(socket.getOutputStream())
                        write(output, "OPTIONS rtsp://127.0.0.1:$port/stream RTSP/1.0\r\nCSeq: 2\r\n\r\n")
                        connected = readResponse(input).startsWith("RTSP/1.0 200")
                    }
                }
                if (!connected) Thread.sleep(25)
            }
            assertTrue(connected)
        } finally {
            server.stop()
        }
    }

    @Test
    fun servesConfiguredLivePathWithoutChangingDefaults() {
        assertEquals(8554, StreamSettings().port)
        assertEquals("/stream", StreamSettings().normalizedPath)

        val port = freePort()
        val settings = StreamSettings(port = port, streamPath = "live", authEnabled = false)
        val metrics = StreamMetrics()
        val broadcaster = StreamBroadcaster(metrics)
        val server = RtspServer(settings, metrics, broadcaster, { null }) { error -> throw AssertionError(error) }
        try {
            server.start(InetAddress.getLoopbackAddress())
            broadcaster.setParameterSets(
                H264ParameterSets(byteArrayOf(0x67, 0x42, 0x00, 0x1f), byteArrayOf(0x68, 0xce.toByte())),
            )
            Socket("127.0.0.1", port).use { socket ->
                socket.soTimeout = 3_000
                val input = BufferedInputStream(socket.getInputStream())
                val output = BufferedOutputStream(socket.getOutputStream())
                write(output, "DESCRIBE rtsp://127.0.0.1:$port/live RTSP/1.0\r\nCSeq: 1\r\n\r\n")
                assertTrue(readResponse(input).startsWith("RTSP/1.0 200"))
                write(output, "DESCRIBE rtsp://127.0.0.1:$port/stream RTSP/1.0\r\nCSeq: 2\r\n\r\n")
                assertTrue(readResponse(input).startsWith("RTSP/1.0 404"))
            }
        } finally {
            server.stop()
        }
    }

    @Test
    fun stopReleasesPortAndAllowsRestartOnSameServer() {
        val port = freePort()
        val settings = StreamSettings(port = port, authEnabled = false)
        val metrics = StreamMetrics()
        val server = RtspServer(settings, metrics, StreamBroadcaster(metrics), { null }) { error -> throw AssertionError(error) }
        server.start(InetAddress.getLoopbackAddress())
        assertTrue(server.stop().isSuccessful)
        assertThrows(IOException::class.java) {
            Socket("127.0.0.1", port).use { }
        }

        try {
            server.start(InetAddress.getLoopbackAddress())
            Socket("127.0.0.1", port).use { socket ->
                val input = BufferedInputStream(socket.getInputStream())
                val output = BufferedOutputStream(socket.getOutputStream())
                write(output, "OPTIONS rtsp://127.0.0.1:$port/stream RTSP/1.0\r\nCSeq: 1\r\n\r\n")
                assertTrue(readResponse(input).startsWith("RTSP/1.0 200"))
            }
        } finally {
            server.stop()
        }
    }

    @Test
    fun stopIsIdempotentAndDoesNotRepeatCleanup() {
        val port = freePort()
        val settings = StreamSettings(port = port, authEnabled = false)
        val metrics = StreamMetrics()
        val server = RtspServer(settings, metrics, StreamBroadcaster(metrics), { null }) { error -> throw AssertionError(error) }
        try {
            server.start(InetAddress.getLoopbackAddress())
            assertTrue(server.stop().isSuccessful)
            assertTrue(server.stop().isSuccessful)
        } finally {
            server.stop()
        }
    }

    @Test
    fun acceptsTwoIndependentRtspClients() {
        val port = freePort()
        val settings = StreamSettings(port = port, authEnabled = false)
        val metrics = StreamMetrics()
        val server = RtspServer(settings, metrics, StreamBroadcaster(metrics), { null }) { error -> throw AssertionError(error) }
        try {
            server.start(InetAddress.getLoopbackAddress())
            Socket("127.0.0.1", port).use { first ->
                Socket("127.0.0.1", port).use { second ->
                    val firstInput = BufferedInputStream(first.getInputStream())
                    val firstOutput = BufferedOutputStream(first.getOutputStream())
                    val secondInput = BufferedInputStream(second.getInputStream())
                    val secondOutput = BufferedOutputStream(second.getOutputStream())
                    write(firstOutput, "OPTIONS rtsp://127.0.0.1:$port/stream RTSP/1.0\r\nCSeq: 1\r\n\r\n")
                    write(secondOutput, "OPTIONS rtsp://127.0.0.1:$port/stream RTSP/1.0\r\nCSeq: 2\r\n\r\n")
                    assertTrue(readResponse(firstInput).startsWith("RTSP/1.0 200"))
                    assertTrue(readResponse(secondInput).startsWith("RTSP/1.0 200"))
                    var connectedClients = metrics.snapshot().connectedClients
                    repeat(20) {
                        if (connectedClients >= 2) return@repeat
                        Thread.sleep(10)
                        connectedClients = metrics.snapshot().connectedClients
                    }
                    assertEquals(2, connectedClients)
                }
            }
        } finally {
            server.stop()
        }
    }

    private fun freePort(): Int = ServerSocket(0).use { it.localPort }

    private fun awaitConnectedClients(metrics: StreamMetrics, expected: Int): Boolean {
        repeat(30) {
            if (metrics.snapshot().connectedClients == expected) return true
            Thread.sleep(10)
        }
        return metrics.snapshot().connectedClients == expected
    }

    private fun write(output: BufferedOutputStream, request: String) {
        output.write(request.toByteArray())
        output.flush()
    }

    private fun readResponse(input: BufferedInputStream): String {
        val buffer = StringBuilder()
        while (!buffer.toString().endsWith("\r\n\r\n")) {
            val value = input.read()
            if (value < 0) error("socket closed while reading RTSP response")
            buffer.append(value.toChar())
        }
        val headers = buffer.toString()
        val length = Regex("(?im)^Content-Length:\\s*(\\d+)").find(headers)?.groupValues?.get(1)?.toIntOrNull() ?: 0
        repeat(length) {
            val value = input.read()
            if (value < 0) error("socket closed while reading RTSP response body")
            buffer.append(value.toChar())
        }
        return buffer.toString()
    }

    private fun readFully(input: BufferedInputStream, target: ByteArray) {
        var offset = 0
        while (offset < target.size) {
            val read = input.read(target, offset, target.size - offset)
            if (read < 0) error("socket closed while reading RTP packet")
            offset += read
        }
    }
}
