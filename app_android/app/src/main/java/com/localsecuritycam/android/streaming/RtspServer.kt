package com.localsecuritycam.android.streaming

import com.localsecuritycam.android.diagnostics.CleanupCollector
import com.localsecuritycam.android.diagnostics.CleanupReport
import com.localsecuritycam.android.diagnostics.StreamMetrics
import com.localsecuritycam.android.diagnostics.StreamErrorFormatter
import com.localsecuritycam.android.diagnostics.StreamErrorKind
import com.localsecuritycam.android.diagnostics.StreamErrorLogger
import com.localsecuritycam.android.diagnostics.StreamFailure
import com.localsecuritycam.android.diagnostics.StreamFailureException
import com.localsecuritycam.android.settings.StreamSettings
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class RtspServer(
    val settings: StreamSettings,
    val metrics: StreamMetrics,
    val broadcaster: StreamBroadcaster,
    private val credentialsProvider: () -> RtspCredentials?,
    private val errorCallback: (StreamFailure) -> Unit,
) {
    private val stopped = AtomicBoolean(true)
    private val sessions = ConcurrentHashMap.newKeySet<RtspSession>()
    private var serverSocket: ServerSocket? = null
    private var acceptThread: Thread? = null
    private var clientExecutor: ExecutorService? = null

    @Synchronized
    fun start(bindAddress: InetAddress) {
        validateStartConfiguration(
            settings = settings,
            bindAddress = bindAddress,
            credentials = if (settings.authEnabled) credentialsProvider() else null,
        )?.let { throw StreamFailureException(it) }
        if (!stopped.compareAndSet(true, false)) return
        StreamErrorLogger.info("RTSP server starting")
        StreamErrorLogger.info("RTSP bind address=${bindAddress.hostAddress}")
        StreamErrorLogger.info("RTSP port=${settings.port}")
        StreamErrorLogger.info("RTSP path=${settings.normalizedPath}")
        broadcaster.reset()
        var socket: ServerSocket? = null
        try {
            val openedSocket = ServerSocket()
            socket = openedSocket
            openedSocket.reuseAddress = true
            openedSocket.bind(InetSocketAddress(bindAddress, settings.port), 16)
            serverSocket = openedSocket
            clientExecutor = Executors.newFixedThreadPool(MAX_CLIENTS) { runnable ->
                Thread(runnable, "rtsp-client").apply { isDaemon = true }
            }
            acceptThread = Thread({ acceptLoop(openedSocket) }, "rtsp-accept").also {
                it.isDaemon = true
                it.start()
            }
            StreamErrorLogger.info("RTSP server started")
        } catch (error: Exception) {
            stopped.set(true)
            val cleanup = CleanupCollector()
            val socketToClose = serverSocket ?: socket
            cleanup.runUnit("RTSP server socket") { socketToClose?.close() }
            serverSocket = null
            cleanup.runUnit("RTSP accept thread interrupt") { acceptThread?.interrupt() }
            acceptThread = null
            val executor = clientExecutor
            clientExecutor = null
            shutdownClientExecutor(executor, cleanup)
            val failure = StreamErrorFormatter.withCleanup(
                StreamErrorFormatter.fromRtspThrowable(error),
                cleanup.report(),
            )
            reportError(failure)
            throw StreamFailureException(failure)
        }
    }

    @Synchronized
    fun stop(): CleanupReport {
        if (!stopped.compareAndSet(false, true)) return CleanupReport()
        StreamErrorLogger.info("RTSP server stopping")
        val cleanup = CleanupCollector()
        // The accept/session threads may remove entries while stop is closing
        // them. ConcurrentHashMap.forEach is weakly consistent and avoids the
        // snapshot iterator race that can otherwise throw during cleanup.
        sessions.forEach { session ->
            cleanup.add(session.closeSink())
        }
        sessions.clear()
        cleanup.runUnit("RTSP server socket") { serverSocket?.close() }
        serverSocket = null
        val accept = acceptThread
        acceptThread = null
        cleanup.runUnit("RTSP accept thread interrupt") { accept?.interrupt() }
        if (accept != null && Thread.currentThread() !== accept) {
            cleanup.runUnit("RTSP accept thread join") { accept.join(1_000) }
            if (accept.isAlive) {
                cleanup.add(
                    StreamErrorFormatter.cleanupFailure(
                        "RTSP accept thread join",
                        IllegalStateException("RTSP accept thread did not stop"),
                    ),
                )
            }
        }
        val executor = clientExecutor
        clientExecutor = null
        shutdownClientExecutor(executor, cleanup)
        metrics.setConnectedClients(0)
        val report = cleanup.report()
        StreamErrorLogger.info("RTSP server stopped")
        return report
    }

    fun credentials(): RtspCredentials? = credentialsProvider()

    internal fun sessionClosed(session: RtspSession) {
        if (sessions.remove(session)) {
            metrics.setConnectedClients(sessions.size)
            StreamErrorLogger.info("RTSP client disconnected")
        }
    }

    /** A malformed or disconnected client must not take the camera pipeline down. */
    fun reportSessionError(message: String) {
        val failure = StreamErrorFormatter.fromMessage(StreamErrorKind.SOCKET, message)
        StreamErrorLogger.error(failure)
        metrics.recordError(StreamErrorFormatter.message(failure))
    }

    private fun reportError(failure: StreamFailure) {
        try {
            errorCallback(failure)
        } catch (error: Exception) {
            StreamErrorLogger.observer(error)
        }
    }

    private fun acceptLoop(socket: ServerSocket) {
        while (!stopped.get()) {
            try {
                val client = socket.accept()
                var stopLoop = false
                synchronized(this) {
                    if (stopped.get()) {
                        try {
                            client.close()
                        } catch (error: Exception) {
                            StreamErrorLogger.cleanup(
                                StreamErrorFormatter.cleanupFailure("RTSP client socket", error),
                            )
                        }
                        stopLoop = true
                    } else if (sessions.size >= MAX_CLIENTS) {
                        try {
                            client.close()
                        } catch (error: Exception) {
                            StreamErrorLogger.cleanup(
                                StreamErrorFormatter.cleanupFailure("RTSP client socket", error),
                            )
                        }
                    } else {
                        val session = RtspSession(client, this)
                        sessions += session
                        metrics.setConnectedClients(sessions.size)
                        StreamErrorLogger.info("RTSP client connected")
                        val executor = clientExecutor
                        if (executor == null) {
                            session.closeSink()
                        } else {
                            try {
                                executor.execute(session)
                            } catch (error: RejectedExecutionException) {
                                session.closeSink()
                                StreamErrorLogger.error(
                                    StreamErrorFormatter.fromThrowable(StreamErrorKind.THREAD, error),
                                )
                            }
                        }
                    }
                }
                if (stopLoop) break
            } catch (error: Exception) {
                if (!stopped.get()) reportError(StreamErrorFormatter.fromRtspThrowable(error))
            }
        }
    }

    private fun shutdownClientExecutor(executor: ExecutorService?, cleanup: CleanupCollector) {
        if (executor == null) return
        cleanup.runUnit("RTSP client executor") {
            executor.shutdownNow()
            if (!executor.awaitTermination(CLEANUP_TIMEOUT_MS, TimeUnit.MILLISECONDS)) {
                throw IllegalStateException("RTSP client executor did not stop")
            }
        }
    }

    companion object {
        /** Bounds sockets, session objects and writer threads per camera process. */
        const val MAX_CLIENTS = 8
        private const val CLEANUP_TIMEOUT_MS = 1_000L

        /**
         * Validates checks that must pass before allocating MediaCodec output.
         * RtspServer.start() repeats the check as a defense-in-depth guard.
         */
        fun validateStartConfiguration(
            settings: StreamSettings,
            bindAddress: InetAddress,
            credentials: RtspCredentials?,
        ): StreamFailure? {
            if (settings.authEnabled && credentials?.enabled != true) {
                return StreamErrorFormatter.fromMessage(
                    StreamErrorKind.CONFIGURATION,
                    "configured RTSP credentials are unavailable",
                    retryable = false,
                )
            }
            return null
        }
    }
}
