package com.localsecuritycam.android.network

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.LinkProperties
import android.net.wifi.WifiManager
import java.net.Inet4Address

data class WifiNetwork(
    val ipAddress: String,
    val ssid: String?,
)

class NetworkInfoProvider(context: Context) {
    private val connectivity = context.applicationContext.getSystemService(ConnectivityManager::class.java)
    private val wifiManager = context.applicationContext.getSystemService(WifiManager::class.java)

    fun currentWifi(): WifiNetwork? {
        val networks = connectivity.allNetworks
        for (network in networks) {
            val capabilities = connectivity.getNetworkCapabilities(network) ?: continue
            if (!capabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI)) continue
            val ip = ipv4(connectivity.getLinkProperties(network)) ?: continue
            val reportedSsid = runCatching { wifiManager?.connectionInfo?.ssid }.getOrNull()
                ?.trim('"')
                ?.takeUnless { it.isBlank() || it.equals("<unknown ssid>", ignoreCase = true) }
            return WifiNetwork(ipAddress = ip, ssid = reportedSsid)
        }
        return null
    }

    private fun ipv4(properties: LinkProperties?): String? = properties?.linkAddresses
        ?.asSequence()
        ?.map { it.address }
        ?.filterIsInstance<Inet4Address>()
        ?.filter { !it.isLoopbackAddress && !it.isLinkLocalAddress }
        ?.firstOrNull()
        ?.hostAddress
}
