package com.localsecuritycam.android.streaming

import com.localsecuritycam.android.diagnostics.CleanupCollector
import com.localsecuritycam.android.diagnostics.CleanupReport
import com.localsecuritycam.android.diagnostics.StreamErrorFormatter
import com.localsecuritycam.android.diagnostics.StreamErrorKind
import com.localsecuritycam.android.diagnostics.StreamErrorLogger
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.ByteArrayOutputStream
import java.io.EOFException
import java.io.IOException
import java.net.Socket
import java.net.SocketTimeoutException
import java.nio.charset.StandardCharsets
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.atomic.AtomicBoolean

internal class RtspSession(
    private val socket: Socket,
    private val server: RtspServer,
) : AccessUnitSink, Runnable {
    private val closed = AtomicBoolean(false)
    private val frameQueue = LatestAccessUnitBuffer()
    private val outputLock = Any()
    private val packetizer = RtpPacketizer()
    private var input: BufferedInputStream? = null
    private var output: BufferedOutputStream? = null
    private var writerThread: Thread? = null
    @Volatile
    private var playing = false
    private var setupDone = false
    private var authenticationFailures = 0
    @Volatile
    private var writerWriting = false
    @Volatile
    private var lastWriteNs = System.nanoTime()
    private var rtpChannel = 0
    private var rtcpChannel = 1
    private val sessionId = java.lang.Long.toHexString(System.nanoTime())

    override fun run() {
        try {
            socket.tcpNoDelay = true
            socket.keepAlive = true
            // A short read tick lets the absolute request deadline work while
            // established PLAY sessions can still wait for keep-alive traffic.
            socket.soTimeout = READ_TICK_TIMEOUT_MS
            input = BufferedInputStream(socket.getInputStream())
            output = BufferedOutputStream(socket.getOutputStream())
            while (!closed.get()) {
                val request = try {
                    readRequest(input!!)
                } catch (_: SocketTimeoutException) {
                    if (playing) {
                        checkWriterHealth()
                        continue
                    } else break
                } ?: break
                val shouldClose = handle(request)
                if (shouldClose) break
                checkWriterHealth()
            }
        } catch (error: Exception) {
            if (!closed.get()) {
                server.reportSessionError(
                    StreamErrorFormatter.fromThrowable(StreamErrorKind.SOCKET, error).detail,
                )
            }
        } finally {
            closeSink()
        }
    }

    override fun enqueue(unit: EncodedAccessUnit): Boolean {
        if (!playing || closed.get()) return true
        val outcome = frameQueue.offer(unit)
        repeat(outcome.droppedUnits) { server.metrics.recordDroppedFrame() }
        return outcome.accepted
    }

    override fun closeSink(): CleanupReport {
        if (!closed.compareAndSet(false, true)) return CleanupReport()
        val cleanup = CleanupCollector()
        playing = false
        server.broadcaster.removeSink(this)
        frameQueue.clear()
        cleanup.runUnit("RTSP writer thread interrupt") { writerThread?.interrupt() }
        cleanup.runUnit("RTSP client socket") { socket.close() }
        server.sessionClosed(this)
        return cleanup.report()
    }

    private fun handle(request: RtspRequest): Boolean {
        val cSeq = request.header("CSeq")
        val credentials = server.credentials()
        if (server.settings.authEnabled && credentials?.enabled != true) {
            write(RtspProtocol.response(503, "Service Unavailable", cSeq))
            return true
        }
        if (!RtspAuth.isAuthorized(request.header("Authorization"), credentials)) {
            write(RtspProtocol.response(401, "Unauthorized", cSeq, mapOf("WWW-Authenticate" to RtspAuth.challenge())))
            authenticationFailures++
            if (authenticationFailures >= MAX_AUTH_FAILURES) return true
            Thread.sleep(AUTH_RETRY_DELAY_MS)
            return false
        }
        authenticationFailures = 0
        val requestSession = request.header("Session")?.substringBefore(';')?.trim()
        if (requestSession != null && requestSession != sessionId && request.method !in setOf("OPTIONS", "DESCRIBE")) {
            write(RtspProtocol.response(454, "Session Not Found", cSeq))
            return false
        }

        return when (request.method) {
            "OPTIONS" -> {
                write(RtspProtocol.response(200, "OK", cSeq, mapOf("Public" to "OPTIONS, DESCRIBE, SETUP, PLAY, GET_PARAMETER, TEARDOWN")))
                false
            }
            "DESCRIBE" -> describe(request, cSeq)
            "SETUP" -> setup(request, cSeq)
            "PLAY" -> play(request, cSeq)
            "GET_PARAMETER" -> {
                write(RtspProtocol.response(200, "OK", cSeq, mapOf("Session" to sessionId)))
                false
            }
            "TEARDOWN" -> {
                write(RtspProtocol.response(200, "OK", cSeq, mapOf("Session" to sessionId)))
                true
            }
            else -> {
                write(RtspProtocol.response(501, "Not Implemented", cSeq, mapOf("Session" to sessionId)))
                false
            }
        }
    }

    private fun describe(request: RtspRequest, cSeq: String?): Boolean {
        if (!RtspProtocol.pathMatches(request.uri, server.settings.normalizedPath)) {
            write(RtspProtocol.response(404, "Not Found", cSeq))
            return false
        }
        val parameters = server.broadcaster.awaitParameterSets(1_500)
        val body = SdpBuilder.build(server.settings, parameters)
        val responseUri = RtspProtocol.withoutCredentials(request.uri)
        write(
            RtspProtocol.response(
                200,
                "OK",
                cSeq,
                mapOf(
                    "Content-Type" to "application/sdp",
                    "Content-Base" to responseUri.trimEnd('/') + "/",
                ),
                body,
            ),
        )
        return false
    }

    private fun setup(request: RtspRequest, cSeq: String?): Boolean {
        if (!RtspProtocol.pathMatches(request.uri, server.settings.normalizedPath)) {
            write(RtspProtocol.response(404, "Not Found", cSeq))
            return false
        }
        val transport = RtspProtocol.parseTransport(request.header("Transport"))
        if (transport == null) {
            write(RtspProtocol.response(461, "Unsupported Transport", cSeq))
            return false
        }
        rtpChannel = transport.interleavedRtpChannel
        rtcpChannel = transport.interleavedRtcpChannel
        setupDone = true
        write(
            RtspProtocol.response(
                200,
                "OK",
                cSeq,
                mapOf(
                    "Transport" to "RTP/AVP/TCP;unicast;interleaved=$rtpChannel-$rtcpChannel",
                    "Session" to sessionId,
                ),
            ),
        )
        return false
    }

    private fun play(request: RtspRequest, cSeq: String?): Boolean {
        if (!RtspProtocol.pathMatches(request.uri, server.settings.normalizedPath)) {
            write(RtspProtocol.response(404, "Not Found", cSeq))
            return false
        }
        if (!setupDone) {
            write(RtspProtocol.response(455, "Method Not Valid in This State", cSeq))
            return false
        }
        if (closed.get()) return true
        playing = true
        frameQueue.resetForPlayback()
        server.broadcaster.addSink(this)
        if (closed.get()) {
            server.broadcaster.removeSink(this)
            return true
        }
        val responseUri = RtspProtocol.withoutCredentials(request.uri)
        write(
            RtspProtocol.response(
                200,
                "OK",
                cSeq,
                mapOf(
                    "Session" to sessionId,
                    "Range" to "npt=0.000-",
                    "RTP-Info" to "url=${responseUri.trimEnd('/')}/trackID=0;seq=${packetizer.nextSequenceNumber()};rtptime=0",
                ),
            ),
        )
        startWriter()
        return false
    }

    private fun startWriter() {
        if (writerThread?.isAlive == true) return
        writerThread = Thread {
            try {
                while (!closed.get()) {
                    val unit = frameQueue.take()
                    val packets = packetizer.packetize(unit)
                    writerWriting = true
                    try {
                        synchronized(outputLock) {
                            packets.forEach { packet ->
                                val length = packet.bytes.size
                                output!!.write('$'.code)
                                output!!.write(rtpChannel)
                                output!!.write((length ushr 8) and 0xff)
                                output!!.write(length and 0xff)
                                output!!.write(packet.bytes)
                                server.metrics.recordBytesSent(length.toLong() + 4L)
                            }
                            output!!.flush()
                            lastWriteNs = System.nanoTime()
                        }
                    } finally {
                        writerWriting = false
                    }
                }
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
            } catch (error: IOException) {
                if (!closed.get()) {
                    StreamErrorLogger.error(StreamErrorFormatter.fromThrowable(StreamErrorKind.SOCKET, error))
                }
                closeSink()
            }
        }.also {
            it.name = "rtsp-writer-${socket.inetAddress.hostAddress}"
            it.isDaemon = true
            it.start()
        }
    }

    private fun checkWriterHealth() {
        if (!playing || closed.get()) return
        val writer = writerThread
        if (writer?.isAlive != true || (writerWriting && System.nanoTime() - lastWriteNs > WRITER_STALL_NS)) {
            closeSink()
        }
    }

    private fun write(value: String) {
        synchronized(outputLock) {
            output?.write(value.toByteArray(StandardCharsets.UTF_8))
            output?.flush()
        }
    }

    private fun readRequest(stream: BufferedInputStream): RtspRequest? {
        while (true) {
            val first = stream.read()
            if (first < 0) return null
            if (first == '$'.code) {
                val header = ByteArray(3)
                val deadline = requestDeadline()
                readFully(stream, header, deadline)
                val channel = header[0].toInt() and 0xff
                require(!setupDone || channel == rtpChannel || channel == rtcpChannel) {
                    "unexpected RTSP interleaved channel"
                }
                val size = ((header[1].toInt() and 0xff) shl 8) or (header[2].toInt() and 0xff)
                skipFully(stream, size, deadline)
                continue
            }
            val deadline = requestDeadline()
            val bytes = ByteArrayOutputStream()
            bytes.write(first)
            var matched = 0
            while (bytes.size() < MAX_REQUEST_BYTES) {
                val value = readByteWithDeadline(stream, deadline)
                if (value < 0) return null
                bytes.write(value)
                matched = when {
                    matched == 0 && value == '\r'.code -> 1
                    matched == 1 && value == '\n'.code -> 2
                    matched == 2 && value == '\r'.code -> 3
                    matched == 3 && value == '\n'.code -> 4
                    else -> 0
                }
                if (matched == 4) break
            }
            require(matched == 4) { "RTSP header too large or incomplete" }
            val headText = bytes.toString(StandardCharsets.UTF_8.name())
            val headerRequest = RtspProtocol.parse(headText)
            val contentLength = headerRequest.header("Content-Length")?.let { value ->
                value.trim().toIntOrNull() ?: error("invalid RTSP Content-Length")
            } ?: 0
            require(contentLength in 0..MAX_BODY_BYTES) { "RTSP body too large" }
            if (contentLength > 0) {
                val body = ByteArray(contentLength)
                readFully(stream, body, deadline)
                bytes.write(body)
            }
            return RtspProtocol.parse(bytes.toString(StandardCharsets.UTF_8.name()))
        }
    }

    private fun requestDeadline(): Long = System.nanoTime() + REQUEST_DEADLINE_NS

    private fun readByteWithDeadline(stream: BufferedInputStream, deadlineNs: Long): Int {
        while (true) {
            if (System.nanoTime() >= deadlineNs) throw RequestDeadlineExceeded()
            try {
                return stream.read()
            } catch (_: SocketTimeoutException) {
                // Re-check the absolute deadline instead of allowing a slow-drip
                // peer to extend the request forever.
            }
        }
    }

    private fun readFully(stream: BufferedInputStream, target: ByteArray, deadlineNs: Long) {
        var offset = 0
        while (offset < target.size) {
            val count = try {
                if (System.nanoTime() >= deadlineNs) throw RequestDeadlineExceeded()
                stream.read(target, offset, target.size - offset)
            } catch (_: SocketTimeoutException) {
                continue
            }
            if (count < 0) throw EOFException("RTSP socket closed")
            offset += count
        }
    }

    private fun skipFully(stream: BufferedInputStream, size: Int, deadlineNs: Long) {
        var remaining = size
        while (remaining > 0) {
            val skipped = try {
                if (System.nanoTime() >= deadlineNs) throw RequestDeadlineExceeded()
                stream.skip(remaining.toLong()).toInt()
            } catch (_: SocketTimeoutException) {
                0
            }
            if (skipped <= 0) {
                if (readByteWithDeadline(stream, deadlineNs) < 0) throw EOFException("interleaved RTCP socket closed")
                remaining--
            } else {
                remaining -= skipped
            }
        }
    }

    private companion object {
        const val MAX_REQUEST_BYTES = 64 * 1024
        const val MAX_BODY_BYTES = 16 * 1024
        const val READ_TICK_TIMEOUT_MS = 1_000
        const val REQUEST_DEADLINE_NS = 15_000_000_000L
        const val MAX_AUTH_FAILURES = 3
        const val AUTH_RETRY_DELAY_MS = 250L
        const val WRITER_STALL_NS = 10_000_000_000L
    }

    private class RequestDeadlineExceeded : IOException("RTSP request deadline exceeded")
}

/**
 * Per-client latest-frame buffer. It is intentionally independent from socket
 * writes so a slow RTSP/TCP consumer cannot grow unbounded memory or force the
 * encoder to wait. When a bound is reached, stale units are removed before the
 * incoming unit; a lost key frame discards dependent frames until the next IDR.
 */
internal data class LatestAccessUnitBufferOffer(
    val accepted: Boolean,
    val droppedUnits: Int,
)

internal class LatestAccessUnitBuffer(
    capacity: Int = DEFAULT_QUEUE_CAPACITY,
    private val maxQueueBytes: Long = DEFAULT_MAX_QUEUE_BYTES,
    private val maxSingleUnitBytes: Int = DEFAULT_MAX_SINGLE_UNIT_BYTES,
) {
    private val queue = ArrayBlockingQueue<EncodedAccessUnit>(capacity)
    private val lock = Any()
    private var queuedBytes = 0L
    private var awaitingKeyFrame = true

    init {
        require(capacity > 0) { "queue capacity must be positive" }
        require(maxQueueBytes > 0) { "maximum queue bytes must be positive" }
        require(maxSingleUnitBytes > 0) { "maximum unit bytes must be positive" }
    }

    fun offer(unit: EncodedAccessUnit): LatestAccessUnitBufferOffer = synchronized(lock) {
        if (awaitingKeyFrame && !unit.isKeyFrame) {
            return@synchronized LatestAccessUnitBufferOffer(accepted = true, droppedUnits = 1)
        }
        if (unit.byteCount > maxSingleUnitBytes || unit.byteCount.toLong() > maxQueueBytes) {
            return@synchronized LatestAccessUnitBufferOffer(accepted = false, droppedUnits = 0)
        }

        var droppedUnits = 0
        while (
            (queuedBytes + unit.byteCount.toLong() > maxQueueBytes || queue.remainingCapacity() == 0) &&
            queue.isNotEmpty()
        ) {
            val stale = queue.poll() ?: break
            queuedBytes = (queuedBytes - stale.byteCount).coerceAtLeast(0L)
            droppedUnits++
            if (stale.isKeyFrame) {
                awaitingKeyFrame = true
                droppedUnits += clearLocked()
                break
            }
        }

        if (awaitingKeyFrame && !unit.isKeyFrame) {
            return@synchronized LatestAccessUnitBufferOffer(accepted = true, droppedUnits = droppedUnits + 1)
        }
        if (!queue.offer(unit)) {
            return@synchronized LatestAccessUnitBufferOffer(accepted = false, droppedUnits = droppedUnits)
        }
        queuedBytes += unit.byteCount.toLong()
        if (unit.isKeyFrame) awaitingKeyFrame = false
        LatestAccessUnitBufferOffer(accepted = true, droppedUnits = droppedUnits)
    }

    fun take(): EncodedAccessUnit {
        val unit = queue.take()
        synchronized(lock) {
            queuedBytes = (queuedBytes - unit.byteCount).coerceAtLeast(0L)
        }
        return unit
    }

    fun resetForPlayback() {
        synchronized(lock) {
            clearLocked()
            awaitingKeyFrame = true
        }
    }

    fun clear() {
        synchronized(lock) { clearLocked() }
    }

    private fun clearLocked(): Int {
        var removed = 0
        while (queue.poll() != null) removed++
        queuedBytes = 0L
        return removed
    }
}

private const val DEFAULT_QUEUE_CAPACITY = 12
private const val DEFAULT_MAX_QUEUE_BYTES = 8 * 1024 * 1024L
private const val DEFAULT_MAX_SINGLE_UNIT_BYTES = 16 * 1024 * 1024
