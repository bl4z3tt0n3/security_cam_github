package com.localsecuritycam.android.diagnostics

import android.util.Log
import java.io.IOException
import java.net.BindException

enum class StreamErrorKind(val defaultRetryable: Boolean) {
    PERMISSION(false),
    CONFIGURATION(false),
    CAMERA(true),
    CAPTURE_SESSION(true),
    SURFACE(true),
    MEDIACODEC(true),
    ENCODER(true),
    RTSP_SERVER(true),
    SOCKET(true),
    PORT(false),
    THREAD(true),
}

data class StreamFailure(
    val kind: StreamErrorKind,
    val detail: String,
    val retryable: Boolean = kind.defaultRetryable,
)

data class CleanupFailure(
    val resource: String,
    val detail: String,
)

data class CleanupReport(
    val failures: List<CleanupFailure> = emptyList(),
) {
    val isSuccessful: Boolean
        get() = failures.isEmpty()

    operator fun plus(other: CleanupReport): CleanupReport =
        CleanupReport(failures + other.failures)
}

class StreamFailureException(
    val failure: StreamFailure,
) : RuntimeException(StreamErrorFormatter.message(failure))

object StreamErrorFormatter {
    fun fromThrowable(
        kind: StreamErrorKind,
        error: Throwable,
        retryable: Boolean = kind.defaultRetryable,
    ): StreamFailure = StreamFailure(
        kind = kind,
        detail = LogSanitizer.error(error),
        retryable = retryable,
    )

    fun fromMessage(
        kind: StreamErrorKind,
        message: String,
        retryable: Boolean = kind.defaultRetryable,
    ): StreamFailure = StreamFailure(
        kind = kind,
        detail = LogSanitizer.text(message).ifBlank { kind.name.lowercase() },
        retryable = retryable,
    )

    fun fromRtspThrowable(error: Throwable): StreamFailure {
        val detail = LogSanitizer.error(error)
        return when {
            error is BindException || detail.contains("address already in use", ignoreCase = true) ||
                detail.contains("eaddrinuse", ignoreCase = true) ->
                fromMessage(StreamErrorKind.PORT, detail, retryable = false)
            error is IllegalThreadStateException || error is SecurityException ->
                fromMessage(StreamErrorKind.THREAD, detail)
            error is IOException -> fromMessage(StreamErrorKind.SOCKET, detail)
            else -> fromMessage(StreamErrorKind.RTSP_SERVER, detail)
        }
    }

    fun message(failure: StreamFailure): String {
        val label = when (failure.kind) {
            StreamErrorKind.PERMISSION -> "Permesso fotocamera negato"
            StreamErrorKind.CONFIGURATION -> "Configurazione stream non valida"
            StreamErrorKind.CAMERA -> "Impossibile aprire la fotocamera"
            StreamErrorKind.CAPTURE_SESSION -> "Impossibile avviare la capture session"
            StreamErrorKind.SURFACE -> "Surface non disponibile"
            StreamErrorKind.MEDIACODEC -> "Errore MediaCodec H.264"
            StreamErrorKind.ENCODER -> "Errore encoder H.264"
            StreamErrorKind.RTSP_SERVER -> "Impossibile avviare il server RTSP"
            StreamErrorKind.SOCKET -> "Errore socket RTSP"
            StreamErrorKind.PORT -> "Porta RTSP non disponibile"
            StreamErrorKind.THREAD -> "Errore thread stream"
        }
        val detail = LogSanitizer.text(failure.detail).trim()
        return if (detail.isBlank()) label else "$label: $detail"
    }

    fun withCleanup(failure: StreamFailure, cleanup: CleanupReport): StreamFailure {
        if (cleanup.isSuccessful) return failure
        val details = buildString {
            append(failure.detail)
            cleanup.failures.forEach { item ->
                append("; cleanup ").append(item.resource).append(": ").append(item.detail)
            }
        }
        return failure.copy(detail = LogSanitizer.text(details))
    }

    fun cleanupFailure(resource: String, error: Throwable): CleanupFailure = CleanupFailure(
        resource = resource,
        detail = LogSanitizer.error(error),
    )
}

object StreamErrorLogger {
    private const val TAG = "LocalSecurityCam.Stream"

    fun info(message: String) {
        Log.i(TAG, LogSanitizer.text(message))
    }

    fun error(failure: StreamFailure) {
        Log.e(TAG, StreamErrorFormatter.message(failure))
    }

    fun cleanup(failure: CleanupFailure) {
        Log.e(TAG, "cleanup ${failure.resource}: ${LogSanitizer.text(failure.detail)}")
    }

    fun observer(error: Throwable) {
        Log.e(TAG, "observer callback failed: ${LogSanitizer.error(error)}")
    }
}

internal class CleanupCollector {
    private val failures = mutableListOf<CleanupFailure>()

    fun add(report: CleanupReport) {
        report.failures.forEach(::add)
    }

    fun add(failure: CleanupFailure) {
        failures += failure
        StreamErrorLogger.cleanup(failure)
    }

    fun run(resource: String, action: () -> CleanupReport) {
        try {
            add(action())
        } catch (error: Exception) {
            add(StreamErrorFormatter.cleanupFailure(resource, error))
        }
    }

    fun runUnit(resource: String, action: () -> Unit) {
        run(resource) {
            action()
            CleanupReport()
        }
    }

    fun report(): CleanupReport = CleanupReport(failures.toList())
}
