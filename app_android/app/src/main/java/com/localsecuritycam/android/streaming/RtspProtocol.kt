package com.localsecuritycam.android.streaming

import java.net.URI

data class RtspRequest(
    val method: String,
    val uri: String,
    val version: String,
    val headers: Map<String, String>,
    val body: String = "",
) {
    fun header(name: String): String? = headers.entries.firstOrNull { it.key.equals(name, ignoreCase = true) }?.value
}

data class RtspTransport(
    val protocol: String,
    val interleavedRtpChannel: Int,
    val interleavedRtcpChannel: Int,
)

object RtspProtocol {
    fun parse(raw: String): RtspRequest {
        val separator = raw.indexOf("\r\n\r\n")
        val headerBlock = if (separator >= 0) raw.substring(0, separator) else raw
        val body = if (separator >= 0) raw.substring(separator + 4) else ""
        val lines = headerBlock.split("\r\n", "\n")
        require(lines.isNotEmpty()) { "empty RTSP request" }
        val requestLine = lines.first().trim().split(Regex("\\s+"), limit = 3)
        require(requestLine.size == 3) { "invalid RTSP request line" }
        val headers = linkedMapOf<String, String>()
        lines.drop(1).forEach { line ->
            if (line.isBlank()) return@forEach
            val colon = line.indexOf(':')
            require(colon > 0) { "invalid RTSP header" }
            val name = line.substring(0, colon).trim()
            require(name.isNotEmpty()) { "invalid RTSP header name" }
            require(headers.keys.none { it.equals(name, ignoreCase = true) }) { "duplicate RTSP header" }
            headers[name] = line.substring(colon + 1).trim()
        }
        return RtspRequest(
            method = requestLine[0].uppercase(),
            uri = requestLine[1],
            version = requestLine[2],
            headers = headers,
            body = body,
        )
    }

    fun parseTransport(value: String?): RtspTransport? {
        val text = value ?: return null
        val first = text.substringBefore(';').trim().uppercase()
        if (first != "RTP/AVP/TCP") return null
        val interleaved = Regex("(?i)(?:^|;)\\s*interleaved\\s*=\\s*(\\d+)\\s*-\\s*(\\d+)")
            .find(text) ?: return null
        val rtp = interleaved.groupValues[1].toIntOrNull() ?: return null
        val rtcp = interleaved.groupValues[2].toIntOrNull() ?: return null
        if (rtp !in 0..255 || rtcp !in 0..255 || rtp == rtcp) return null
        return RtspTransport(
            protocol = first,
            interleavedRtpChannel = rtp,
            interleavedRtcpChannel = rtcp,
        )
    }

    fun pathFromUri(uri: String): String {
        return runCatching {
            val path = URI(uri).rawPath
            if (path.isNullOrBlank()) "/" else path
        }.getOrElse {
            uri.substringAfter("://", uri).substringAfter('/', "/").substringBefore('?')
                .ifBlank { "/" }
                .let { if (it.startsWith('/')) it else "/$it" }
        }
    }

    fun pathMatches(requestUri: String, configuredPath: String): Boolean {
        val actual = pathFromUri(requestUri).trimEnd('/').ifBlank { "/" }
        val base = configuredPath.trim().let { if (it.startsWith('/')) it else "/$it" }
            .trimEnd('/').ifBlank { "/" }
        return actual == base || actual == "$base/trackID=0"
    }

    fun withoutCredentials(uri: String): String {
        val clean = uri.substringBefore('?').substringBefore('#')
        val schemeEnd = clean.indexOf("://")
        if (schemeEnd < 0) return clean
        val authorityStart = schemeEnd + 3
        val pathStart = clean.indexOf('/', authorityStart).let { if (it < 0) clean.length else it }
        val at = clean.lastIndexOf('@', pathStart - 1)
        return if (at >= authorityStart) clean.removeRange(authorityStart, at + 1) else clean
    }

    fun response(
        statusCode: Int,
        reason: String,
        cSeq: String?,
        headers: Map<String, String> = emptyMap(),
        body: String = "",
    ): String {
        val output = StringBuilder("RTSP/1.0 $statusCode $reason\r\n")
        if (!cSeq.isNullOrBlank()) output.append("CSeq: ").append(cSeq).append("\r\n")
        headers.forEach { (key, value) -> output.append(key).append(": ").append(value).append("\r\n") }
        output.append("Content-Length: ").append(body.toByteArray(Charsets.UTF_8).size).append("\r\n")
        output.append("\r\n").append(body)
        return output.toString()
    }
}
