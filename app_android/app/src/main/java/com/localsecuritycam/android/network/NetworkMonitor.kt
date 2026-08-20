package com.localsecuritycam.android.network

import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.os.Handler
import android.os.Looper
import com.localsecuritycam.android.diagnostics.StreamErrorFormatter
import com.localsecuritycam.android.diagnostics.StreamErrorKind
import com.localsecuritycam.android.diagnostics.StreamErrorLogger

class NetworkMonitor(
    private val provider: NetworkInfoProvider,
    private val onChanged: (WifiNetwork?) -> Unit,
) {
    private val handler = Handler(Looper.getMainLooper())
    private var registered = false
    private val callback = object : ConnectivityManager.NetworkCallback() {
        override fun onAvailable(network: Network) = notifyCurrent()

        override fun onLost(network: Network) = notifyCurrent()

        override fun onLinkPropertiesChanged(network: Network, linkProperties: android.net.LinkProperties) = notifyCurrent()

        private fun notifyCurrent() {
            if (!handler.post { onChanged(provider.currentWifi()) }) {
                StreamErrorLogger.error(
                    StreamErrorFormatter.fromMessage(
                        StreamErrorKind.THREAD,
                        "network state callback dispatch rejected",
                    ),
                )
            }
        }
    }

    fun start(connectivity: ConnectivityManager) {
        if (registered) return
        registered = true
        handler.post { onChanged(provider.currentWifi()) }
        try {
            connectivity.registerNetworkCallback(
                NetworkRequest.Builder()
                    .addTransportType(NetworkCapabilities.TRANSPORT_WIFI)
                    .build(),
                callback,
            )
        } catch (error: Exception) {
            registered = false
            StreamErrorLogger.error(StreamErrorFormatter.fromThrowable(StreamErrorKind.THREAD, error))
            // Fail closed if a vendor blocks callbacks: an initial Wi-Fi snapshot
            // must not keep an obsolete RTSP listener alive indefinitely.
            handler.post { onChanged(null) }
        }
    }

    fun stop(connectivity: ConnectivityManager) {
        if (!registered) return
        registered = false
        try {
            connectivity.unregisterNetworkCallback(callback)
        } catch (error: Exception) {
            StreamErrorLogger.cleanup(
                StreamErrorFormatter.cleanupFailure("network callback", error),
            )
        }
    }
}
