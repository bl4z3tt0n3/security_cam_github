package com.localsecuritycam.android.diagnostics

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.IOException
import java.net.BindException

class StreamErrorsTest {
    @Test
    fun formatsEveryCriticalCategoryWithReadableText() {
        val labels = mapOf(
            StreamErrorKind.PERMISSION to "Permesso fotocamera negato",
            StreamErrorKind.CONFIGURATION to "Configurazione stream non valida",
            StreamErrorKind.CAMERA to "Impossibile aprire la fotocamera",
            StreamErrorKind.CAPTURE_SESSION to "Impossibile avviare la capture session",
            StreamErrorKind.SURFACE to "Surface non disponibile",
            StreamErrorKind.MEDIACODEC to "Errore MediaCodec H.264",
            StreamErrorKind.ENCODER to "Errore encoder H.264",
            StreamErrorKind.RTSP_SERVER to "Impossibile avviare il server RTSP",
            StreamErrorKind.SOCKET to "Errore socket RTSP",
            StreamErrorKind.PORT to "Porta RTSP non disponibile",
            StreamErrorKind.THREAD to "Errore thread stream",
        )

        labels.forEach { (kind, label) ->
            val message = StreamErrorFormatter.message(
                StreamErrorFormatter.fromMessage(kind, "causa reale"),
            )
            assertTrue("missing label for $kind", message.startsWith(label))
            assertTrue(message.contains("causa reale"))
        }
    }

    @Test
    fun sanitizesCredentialsAndUrlsInMessages() {
        val failure = StreamErrorFormatter.fromMessage(
            StreamErrorKind.RTSP_SERVER,
            "failed rtsp://admin:secret@camera.local/live?password=query-secret",
        )

        val message = StreamErrorFormatter.message(failure)

        assertFalse(message.contains("secret"))
        assertFalse(message.contains("query-secret"))
        assertTrue(message.contains("rtsp://admin:***@camera.local/live?password=***"))
    }

    @Test
    fun classifiesPortAndSocketFailuresSeparately() {
        val port = StreamErrorFormatter.fromRtspThrowable(BindException("Address already in use"))
        val socket = StreamErrorFormatter.fromRtspThrowable(IOException("connection reset"))

        assertTrue(port.kind == StreamErrorKind.PORT)
        assertFalse(port.retryable)
        assertTrue(socket.kind == StreamErrorKind.SOCKET)
        assertTrue(socket.retryable)
    }

    @Test
    fun appendsCleanupFailuresWithoutReplacingPrimaryCause() {
        val primary = StreamErrorFormatter.fromMessage(
            StreamErrorKind.RTSP_SERVER,
            "porta occupata",
            retryable = false,
        )
        val combined = StreamErrorFormatter.withCleanup(
            primary,
            CleanupReport(listOf(CleanupFailure("encoder", "release failed"))),
        )

        assertTrue(combined.detail.startsWith("porta occupata"))
        assertTrue(combined.detail.contains("cleanup encoder: release failed"))
        assertFalse(combined.retryable)
    }

    @Test
    fun persistentFailuresAreManualRetryOnly() {
        assertFalse(StreamErrorKind.PERMISSION.defaultRetryable)
        assertFalse(StreamErrorKind.CONFIGURATION.defaultRetryable)
        assertFalse(StreamErrorKind.PORT.defaultRetryable)
        assertTrue(StreamErrorKind.CAMERA.defaultRetryable)
        assertTrue(StreamErrorKind.CAPTURE_SESSION.defaultRetryable)
        assertTrue(StreamErrorKind.SURFACE.defaultRetryable)
        assertTrue(StreamErrorKind.MEDIACODEC.defaultRetryable)
        assertTrue(StreamErrorKind.ENCODER.defaultRetryable)
        assertTrue(StreamErrorKind.RTSP_SERVER.defaultRetryable)
        assertTrue(StreamErrorKind.SOCKET.defaultRetryable)
        assertTrue(StreamErrorKind.THREAD.defaultRetryable)
    }
}
