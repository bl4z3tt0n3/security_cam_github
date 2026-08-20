package com.localsecuritycam.android.service

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.net.ConnectivityManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.SystemClock
import android.view.OrientationEventListener
import com.localsecuritycam.android.MainActivity
import com.localsecuritycam.android.R
import com.localsecuritycam.android.camera.CameraOrientationState
import com.localsecuritycam.android.camera.DeviceOrientation
import com.localsecuritycam.android.camera.PhysicalOrientationStabilizer
import com.localsecuritycam.android.camera.PreviewDiagnosticMode
import com.localsecuritycam.android.camera.PreviewSurfaceAttachment
import com.localsecuritycam.android.camera.PREVIEW_DIAGNOSTIC_EXTRA
import com.localsecuritycam.android.diagnostics.CleanupReport
import com.localsecuritycam.android.diagnostics.StreamErrorFormatter
import com.localsecuritycam.android.diagnostics.StreamErrorKind
import com.localsecuritycam.android.diagnostics.StreamErrorLogger
import com.localsecuritycam.android.diagnostics.StreamFailure
import com.localsecuritycam.android.diagnostics.StreamMetrics
import com.localsecuritycam.android.diagnostics.StreamMetricsSnapshot
import com.localsecuritycam.android.diagnostics.StreamState
import com.localsecuritycam.android.diagnostics.StreamStateMachine
import com.localsecuritycam.android.diagnostics.StreamSubsystem
import com.localsecuritycam.android.diagnostics.StreamSubsystemSnapshot
import com.localsecuritycam.android.diagnostics.SubsystemState
import com.localsecuritycam.android.diagnostics.subsystemStatesForCleanupFailure
import com.localsecuritycam.android.network.NetworkInfoProvider
import com.localsecuritycam.android.network.NetworkMonitor
import com.localsecuritycam.android.network.WifiNetwork
import com.localsecuritycam.android.settings.AppSettings
import com.localsecuritycam.android.settings.SettingsRepository
import com.localsecuritycam.android.settings.StreamUrlBuilder
import java.net.InetAddress
import java.util.concurrent.CopyOnWriteArraySet
import java.util.concurrent.atomic.AtomicBoolean

data class ServiceSnapshot(
    val state: StreamState,
    val wifiConnected: Boolean,
    val ssid: String?,
    val localIp: String?,
    val rtspUrl: String?,
    val metrics: StreamMetricsSnapshot,
    val settings: AppSettings,
    val lastError: String?,
    val lastErrorKind: StreamErrorKind? = null,
    val subsystems: StreamSubsystemSnapshot = StreamSubsystemSnapshot(),
    val orientation: CameraOrientationState? = null,
    val previewState: PreviewState = PreviewState.IDLE,
    val previewError: String? = null,
    val previewErrorKind: StreamErrorKind? = null,
)

enum class PreviewState {
    IDLE,
    STARTING,
    ACTIVE,
    ERROR,
}

fun interface ServiceObserver {
    fun onSnapshot(snapshot: ServiceSnapshot)
}

class CameraStreamingService : Service() {
    private val mainHandler = Handler(Looper.getMainLooper())
    private val binder = LocalBinder()
    private val observers = CopyOnWriteArraySet<ServiceObserver>()
    private lateinit var settingsRepository: SettingsRepository
    private lateinit var networkProvider: NetworkInfoProvider
    private lateinit var networkMonitor: NetworkMonitor
    private lateinit var connectivity: ConnectivityManager
    private var settings = AppSettings()
    private var wifiNetwork: WifiNetwork? = null
    private var networkResumeReason: NetworkResumeReason? = null
    private var previewSurface: PreviewSurfaceAttachment? = null
    private var previewBindingGeneration = 0L
    private var activePreviewBindingGeneration = 0L
    private var previewState = PreviewState.IDLE
    private var previewError: String? = null
    private var previewErrorKind: StreamErrorKind? = null
    private var previewDiagnosticMode = PreviewDiagnosticMode.NORMAL
    private var displayRotationDegrees: Int = 0
    private val orientationStabilizer = PhysicalOrientationStabilizer()
    private var orientationMonitor: OrientationEventListener? = null
    private var orientationSensorAvailable = false
    private var physicalOrientationDegrees: Int? = null
    private var orientationState: CameraOrientationState? = null
    private val stateMachine = StreamStateMachine()
    private val state: StreamState
        get() = stateMachine.state
    @Volatile
    private var subsystemStates = StreamSubsystemSnapshot()
    private var lastError: String? = null
    private var lastErrorKind: StreamErrorKind? = null
    private var reconnectCount = 0L
    private var sessionRestartCount = 0L
    private var streamStartRequested = false
    private var retryAttempt = 0
    private var retryRunnable: Runnable? = null
    private val notificationUpdatePending = AtomicBoolean(false)
    private val notificationUpdateRunnable = Runnable {
        notificationUpdatePending.set(false)
        notifyObservers()
    }
    private val lifecycle = StreamLifecycleController(
        factory = StreamPipelineFactory { request, callbacks ->
            StreamingPipeline(
                request = request,
                callbacks = callbacks,
                resources = AndroidStreamingResourceFactory(this@CameraStreamingService),
                dispatch = { action -> mainHandler.post(action) },
                onMetricsChanged = { mainHandler.post { scheduleNotifyObservers() } },
            )
        },
        dispatch = { action -> mainHandler.post(action) },
        onPreviewReady = ::onPreviewReady,
        onStreamReady = ::onPipelineReady,
        onError = ::onPipelineError,
        onPreviewDiagnostic = ::onPreviewDiagnostic,
        onPreviewRecovered = ::onPreviewRecovered,
        onSubsystemStateChanged = ::onSubsystemStateChanged,
        onOrientationChanged = { next ->
            onControl {
                orientationState = next
                scheduleNotifyObservers()
            }
        },
    )

    override fun onCreate() {
        super.onCreate()
        settingsRepository = SettingsRepository(this)
        settings = settingsRepository.load()
        connectivity = getSystemService(ConnectivityManager::class.java)
        networkProvider = NetworkInfoProvider(this)
        networkMonitor = NetworkMonitor(networkProvider) { network -> onNetworkChanged(network) }
        createNotificationChannel()
        networkMonitor.start(connectivity)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0) {
            intent?.getStringExtra(PREVIEW_DIAGNOSTIC_EXTRA)?.let { value ->
                previewDiagnosticMode = PreviewDiagnosticMode.fromWireValue(value)
                StreamErrorLogger.info("PREVIEW_DIAGNOSTIC_MODE ${previewDiagnosticMode.wireValue}")
            }
        }
        val action = intent?.action
        when (action) {
            ACTION_STOP -> stopStreamingInternal()
            ACTION_START -> startStreamingInternal()
            ACTION_START_PREVIEW -> startPreviewInternal()
            else -> if (settings.stream.autoStart) startStreamingInternal() else startPreviewInternal()
        }
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onDestroy() {
        cancelRetry()
        mainHandler.removeCallbacks(notificationUpdateRunnable)
        notificationUpdatePending.set(false)
        networkResumeReason = null
        if (lifecycle.activePipeline != null) {
            if (state == StreamState.STARTING || state == StreamState.STREAMING || state == StreamState.WAITING_NETWORK) {
                transitionTo(StreamState.STOPPING)
            }
            val cleanup = lifecycle.stop()
            if (cleanup.isSuccessful) {
                if (state == StreamState.STOPPING) transitionTo(StreamState.STOPPED)
                previewState = PreviewState.IDLE
            } else {
                if (state != StreamState.ERROR && state != StreamState.STOPPED) transitionTo(StreamState.ERROR)
                StreamErrorLogger.error(cleanupFailure(cleanup))
            }
        }
        networkMonitor.stop(connectivity)
        stopOrientationMonitoring()
        previewSurface = null
        removeForegroundNotification()
        super.onDestroy()
    }

    inner class LocalBinder : android.os.Binder() {
        fun startPreview() = onControl(::startPreviewInternal)
        fun startStreaming() = onControl(::startStreamingInternal)
        fun stopStreaming() = onControl(::stopStreamingInternal)
        fun beginPreviewBinding(): Long {
            previewBindingGeneration++
            val bindingGeneration = previewBindingGeneration
            activePreviewBindingGeneration = bindingGeneration
            previewSurface = null
            onControl {
                if (activePreviewBindingGeneration == bindingGeneration) {
                    lifecycle.activePipeline?.setPreviewSurface(null)
                }
            }
            return bindingGeneration
        }

        fun attachPreview(generation: Long, surface: PreviewSurfaceAttachment) {
            onControl {
                if (generation != activePreviewBindingGeneration || !surface.surface.isValid) {
                    StreamErrorLogger.info("Stale or invalid preview Surface attach ignored generation=$generation")
                    return@onControl
                }
                previewSurface = surface
                StreamErrorLogger.info(
                    "SurfaceView preview attached generation=$generation size=${surface.width}x${surface.height}",
                )
                lifecycle.activePipeline?.setPreviewSurface(surface)
            }
        }

        fun detachPreview(generation: Long) {
            onControl {
                if (generation != activePreviewBindingGeneration) {
                    StreamErrorLogger.info("Stale preview Surface detach ignored generation=$generation")
                    return@onControl
                }
                previewSurface = null
                StreamErrorLogger.info("SurfaceView preview detached generation=$generation")
                lifecycle.activePipeline?.setPreviewSurface(null)
            }
        }

        fun reportPreviewPermissionDenied() = onControl {
            setPreviewError(
                StreamErrorFormatter.fromMessage(
                    StreamErrorKind.PERMISSION,
                    "camera permission is not granted",
                    retryable = false,
                ),
            )
        }
        fun setDisplayRotation(rotationDegrees: Int) = onControl {
            val normalized = ((rotationDegrees % 360) + 360) % 360
            displayRotationDegrees = normalized
            // The Activity is never the orientation authority. It refreshes
            // diagnostics and is only used if the physical sensor is absent.
            lifecycle.activePipeline?.setOrientation(currentDeviceOrientation())
        }
        fun snapshot(): ServiceSnapshot = createSnapshot()
        fun settings(): AppSettings = settings
        fun saveSettings(value: AppSettings) = onControl { saveSettingsInternal(value) }
        fun addObserver(observer: ServiceObserver) {
            observers += observer
            try {
                observer.onSnapshot(createSnapshot())
            } catch (error: Exception) {
                StreamErrorLogger.observer(error)
            }
        }
        fun removeObserver(observer: ServiceObserver) = observers.remove(observer)
    }

    /**
     * This foreground service remains alive when the Activity is hidden, so it
     * is the sole owner of physical orientation. Display rotation only fills
     * the gap on hardware that cannot report orientation samples.
     */
    private fun startOrientationMonitoring() {
        val listener = orientationMonitor ?: object : OrientationEventListener(this) {
            override fun onOrientationChanged(orientation: Int) {
                onControl { handlePhysicalOrientationSample(orientation) }
            }
        }.also { orientationMonitor = it }
        orientationStabilizer.reset()
        physicalOrientationDegrees = null
        orientationSensorAvailable = listener.canDetectOrientation()
        if (orientationSensorAvailable) {
            listener.enable()
            StreamErrorLogger.info("CAMERA_ORIENTATION source=physical_sensor waiting_for_first_quadrant")
        } else {
            listener.disable()
            StreamErrorLogger.info(
                "CAMERA_ORIENTATION source=display_fallback display=$displayRotationDegrees sensor=unavailable",
            )
        }
    }

    private fun stopOrientationMonitoring() {
        orientationMonitor?.disable()
        orientationSensorAvailable = false
        physicalOrientationDegrees = null
        orientationStabilizer.reset()
    }

    private fun handlePhysicalOrientationSample(rawOrientationDegrees: Int) {
        if (!orientationSensorAvailable) return
        val stable = orientationStabilizer.update(rawOrientationDegrees, SystemClock.elapsedRealtime()) ?: return
        if (physicalOrientationDegrees == stable) return
        physicalOrientationDegrees = stable
        val orientation = currentDeviceOrientation()
        StreamErrorLogger.info(
            "CAMERA_ORIENTATION source=physical_sensor physical=$stable " +
                "target_surface_rotation=${orientation.targetSurfaceRotationDegrees} " +
                "display=${orientation.displayRotationDegrees}",
        )
        lifecycle.activePipeline?.setOrientation(orientation)
        scheduleNotifyObservers()
    }

    private fun currentDeviceOrientation(): DeviceOrientation = DeviceOrientation(
        physicalOrientationDegrees = if (orientationSensorAvailable) physicalOrientationDegrees else null,
        displayRotationDegrees = displayRotationDegrees,
    )

    private fun startPreviewInternal() {
        if (previewState == PreviewState.ACTIVE || previewState == PreviewState.STARTING) return
        if (previewState == PreviewState.ERROR && lifecycle.activePipeline != null) {
            lifecycle.activePipeline?.setPreviewSurface(previewSurface)
            return
        }
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            setPreviewError(
                StreamErrorFormatter.fromMessage(
                    StreamErrorKind.PERMISSION,
                    "camera permission is not granted",
                    retryable = false,
                ),
            )
            return
        }
        val foregroundFailure = ensureForegroundFailure()
        if (foregroundFailure != null) {
            setPreviewError(foregroundFailure)
            return
        }
        cancelRetry()
        previewState = PreviewState.STARTING
        previewError = null
        previewErrorKind = null
        prepareSubsystemsForPreview()
        notifyObservers()
        startOrientationMonitoring()
        if (lifecycle.activePipeline == null) {
            lifecycle.start(
                StreamPipelineRequest(
                    settings = settings,
                    preview = previewSurface,
                    displayRotationDegrees = displayRotationDegrees,
                    initialOrientation = currentDeviceOrientation(),
                    initialReconnectCount = reconnectCount,
                    initialSessionRestartCount = sessionRestartCount,
                    previewDiagnosticMode = previewDiagnosticMode,
                ),
            )
        }
    }

    private fun startStreamingInternal() {
        if (state == StreamState.STOPPING) return
        cancelRetry()
        retryAttempt = 0
        streamStartRequested = true
        if (state == StreamState.STOPPED || state == StreamState.ERROR) transitionTo(StreamState.STARTING)
        lastError = null
        lastErrorKind = null
        prepareSubsystemsForStreamStart()
        notifyObservers()
        if (previewState != PreviewState.ACTIVE) {
            startPreviewInternal()
            if (previewState == PreviewState.ERROR && state == StreamState.STARTING && lifecycle.activePipeline == null) {
                transitionTo(StreamState.ERROR)
                notifyObservers()
            }
            return
        }
        // The network callback is asynchronous. Read once here so an explicit
        // Start on an already connected phone does not briefly enter a stale
        // waiting state just because the observer has not fired yet.
        if (wifiNetwork == null) wifiNetwork = networkProvider.currentWifi()
        if (wifiNetwork == null) {
            // Waiting for a network is visible and cancellable, but it is not
            // an implicit auto-start. Only a stream that was already live may
            // resume automatically after a Wi-Fi interruption.
            networkResumeReason = null
            subsystemStates = subsystemStates.copy(
                camera = SubsystemState.RUNNING,
                encoder = SubsystemState.WAITING_NETWORK,
                rtspServer = SubsystemState.WAITING_NETWORK,
            )
            transitionTo(StreamState.WAITING_NETWORK)
            notifyObservers()
            return
        }
        requestStreamingOutput()
    }

    private fun stopStreamingInternal() {
        if (state == StreamState.STOPPING) return
        cancelRetry()
        networkResumeReason = null
        streamStartRequested = false
        val cleanup = if (state != StreamState.STOPPED || lifecycle.activePipeline != null) {
            stopStreamOutputFor(StreamState.STOPPED)
        } else {
            CleanupReport()
        }
        if (!cleanup.isSuccessful) {
            notifyObservers()
            return
        }
        reconnectCount = 0L
        sessionRestartCount = 0L
        retryAttempt = 0
        lastError = null
        lastErrorKind = null
        orientationState = null
        if (previewState == PreviewState.ACTIVE) {
            subsystemStates = subsystemStates.copy(
                camera = SubsystemState.RUNNING,
                encoder = SubsystemState.IDLE,
                rtspServer = SubsystemState.IDLE,
            )
        }
        notifyObservers()
    }

    private fun saveSettingsInternal(value: AppSettings) {
        try {
            settingsRepository.save(value)
            settings = value
            val wasStreaming = streamStartRequested ||
                state == StreamState.STARTING ||
                state == StreamState.STREAMING ||
                state == StreamState.WAITING_NETWORK
            val shouldRestart = state == StreamState.STARTING ||
                state == StreamState.STREAMING ||
                state == StreamState.WAITING_NETWORK ||
                lifecycle.activePipeline != null ||
                retryRunnable != null ||
                networkResumeReason != null
            if (shouldRestart) {
                cancelRetry()
                retryAttempt = 0
                networkResumeReason = null
                streamStartRequested = false
                val cleanup = stopPreviewPipelineFor()
                if (!cleanup.isSuccessful) {
                    notifyObservers()
                    return
                }
                if (wasStreaming) {
                    streamStartRequested = true
                    transitionTo(StreamState.STARTING)
                    prepareSubsystemsForStreamStart()
                }
                startPreviewInternal()
            }
            notifyObservers()
        } catch (error: Exception) {
            val failure = StreamErrorFormatter.fromThrowable(StreamErrorKind.CONFIGURATION, error, retryable = false)
            StreamErrorLogger.error(failure)
            lastError = StreamErrorFormatter.message(failure)
            lastErrorKind = failure.kind
            if (state != StreamState.STOPPED && state != StreamState.STOPPING) {
                transitionTo(StreamState.ERROR)
            }
            notifyObservers()
        }
    }

    private fun onNetworkChanged(network: WifiNetwork?) {
        val previous = wifiNetwork
        wifiNetwork = network
        val lifecycleRequested = streamStartRequested ||
            networkResumeReason != null ||
            state == StreamState.STARTING ||
            state == StreamState.STREAMING ||
            state == StreamState.WAITING_NETWORK
        if (!lifecycleRequested) {
            notifyObservers()
            return
        }
        if (network == null) {
            if (state == StreamState.ERROR) {
                notifyObservers()
                return
            }
            val recoveryReason = NetworkRecoveryPolicy.reasonForNetworkLoss(state)
            if (recoveryReason != null) {
                networkResumeReason = recoveryReason
                reconnectCount++
                sessionRestartCount++
            } else if (state != StreamState.WAITING_NETWORK) {
                networkResumeReason = null
            }
            cancelRetry()
            val cleanup = stopStreamOutputFor(StreamState.WAITING_NETWORK)
            if (cleanup.isSuccessful) lastError = "Wi-Fi network unavailable"
            notifyObservers()
            return
        }
        val networkIdentityChanged = previous != null && (
            previous.ipAddress != network.ipAddress ||
                (previous.ssid != null && network.ssid != null && previous.ssid != network.ssid)
            )
        if (networkIdentityChanged &&
            (state == StreamState.STARTING ||
                state == StreamState.STREAMING ||
                lifecycle.activePipeline != null)
        ) {
            val wasStreaming = state == StreamState.STREAMING
            networkResumeReason = NetworkRecoveryPolicy.reasonForNetworkLoss(state)
            if (networkResumeReason != null) {
                reconnectCount++
                sessionRestartCount++
            }
            val cleanup = stopStreamOutputFor(StreamState.STOPPED)
            if (!cleanup.isSuccessful) {
                notifyObservers()
                return
            }
            if (wasStreaming) {
                networkResumeReason = null
                transitionTo(StreamState.STARTING)
                lastError = null
                lastErrorKind = null
                prepareSubsystemsForStreamStart()
                notifyObservers()
                requestStreamingOutput()
            } else {
                lastError = null
                lastErrorKind = null
            }
            notifyObservers()
            return
        }
        if (NetworkRecoveryPolicy.shouldResume(state, networkResumeReason)) {
            networkResumeReason = null
            transitionTo(StreamState.STARTING)
            lastError = null
            lastErrorKind = null
            prepareSubsystemsForStreamStart()
            notifyObservers()
            requestStreamingOutput()
        } else if (state == StreamState.WAITING_NETWORK) {
            // A manual start made while offline must be explicitly retried once
            // the network is back; it was never an active stream to recover.
            streamStartRequested = false
            subsystemStates = subsystemStates.copy(
                camera = if (previewState == PreviewState.ACTIVE) SubsystemState.RUNNING else SubsystemState.IDLE,
                encoder = SubsystemState.IDLE,
                rtspServer = SubsystemState.IDLE,
            )
            transitionTo(StreamState.STOPPED)
            lastError = null
            lastErrorKind = null
        }
        notifyObservers()
    }

    private fun requestStreamingOutput() {
        val network = wifiNetwork ?: run {
            if (state != StreamState.WAITING_NETWORK) transitionTo(StreamState.WAITING_NETWORK)
            notifyObservers()
            return
        }
        if (state != StreamState.STARTING || lifecycle.activePipeline == null) return
        lifecycle.startStreaming(InetAddress.getByName(network.ipAddress))
    }

    private fun onPreviewReady() {
        if (!mainHandler.post {
                if (lifecycle.activePipeline != null && previewState == PreviewState.STARTING) {
                    previewState = PreviewState.ACTIVE
                    previewError = null
                    previewErrorKind = null
                    subsystemStates = subsystemStates.copy(camera = SubsystemState.RUNNING)
                    StreamErrorLogger.info("PREVIEW_ACTIVE")
                    notifyObservers()
                    if (streamStartRequested && state == StreamState.STARTING) {
                        if (wifiNetwork == null) {
                            transitionTo(StreamState.WAITING_NETWORK)
                            subsystemStates = subsystemStates.copy(
                                encoder = SubsystemState.WAITING_NETWORK,
                                rtspServer = SubsystemState.WAITING_NETWORK,
                            )
                            notifyObservers()
                        } else {
                            requestStreamingOutput()
                        }
                    }
                }
            }) {
            StreamErrorLogger.error(
                StreamFailure(StreamErrorKind.THREAD, "preview ready callback dispatch rejected"),
            )
        }
    }

    private fun onPipelineReady() {
        if (!mainHandler.post {
            if (lifecycle.activePipeline != null && state == StreamState.STARTING) {
                retryAttempt = 0
                networkResumeReason = null
                subsystemStates = subsystemStates.copy(
                    camera = SubsystemState.RUNNING,
                    encoder = SubsystemState.RUNNING,
                    rtspServer = SubsystemState.RUNNING,
                )
                transitionTo(StreamState.STREAMING)
                StreamErrorLogger.info("STREAM_ACTIVE")
                notifyObservers()
            }
        }) {
            StreamErrorLogger.error(
                StreamFailure(StreamErrorKind.THREAD, "pipeline ready callback dispatch rejected"),
            )
        }
    }

    private fun onPipelineError(stage: PipelineStage, failure: StreamFailure) {
        if (!mainHandler.post {
            StreamErrorLogger.error(failure)
            if (stage == PipelineStage.PREVIEW) {
                StreamErrorLogger.info("camera_failed_fatal kind=${failure.kind.name.lowercase()}")
                setPreviewError(failure)
                if (state == StreamState.STARTING || state == StreamState.STREAMING) {
                    transitionTo(StreamState.ERROR)
                }
                notifyObservers()
            } else {
                if (state != StreamState.STARTING && state != StreamState.STREAMING) return@post
                lastError = StreamErrorFormatter.message(failure)
                lastErrorKind = failure.kind
                subsystemStates = subsystemStates.copy(
                    camera = if (previewState == PreviewState.ACTIVE) SubsystemState.RUNNING else SubsystemState.ERROR,
                    encoder = SubsystemState.ERROR,
                    rtspServer = SubsystemState.ERROR,
                )
                transitionTo(StreamState.ERROR)
                notifyObservers()
                if (failure.retryable) scheduleRetry()
                else cancelRetry()
            }
        }) {
            StreamErrorLogger.error(
                StreamFailure(StreamErrorKind.THREAD, "pipeline error callback dispatch rejected"),
            )
        }
    }

    private fun onPreviewDiagnostic(failure: StreamFailure) {
        if (lifecycle.activePipeline == null || previewState == PreviewState.IDLE) return
        previewState = PreviewState.ERROR
        previewError = StreamErrorFormatter.message(failure)
        previewErrorKind = failure.kind
        StreamErrorLogger.error(failure)
        notifyObservers()
    }

    private fun onPreviewRecovered() {
        if (lifecycle.activePipeline == null || previewState == PreviewState.IDLE) return
        previewState = PreviewState.ACTIVE
        previewError = null
        previewErrorKind = null
        subsystemStates = subsystemStates.copy(camera = SubsystemState.RUNNING)
        StreamErrorLogger.info("PREVIEW_ACTIVE recovered=true")
        notifyObservers()
        if (streamStartRequested && state == StreamState.STARTING) {
            if (wifiNetwork == null) {
                transitionTo(StreamState.WAITING_NETWORK)
                subsystemStates = subsystemStates.copy(
                    encoder = SubsystemState.WAITING_NETWORK,
                    rtspServer = SubsystemState.WAITING_NETWORK,
                )
                notifyObservers()
            } else {
                requestStreamingOutput()
            }
        }
    }

    private fun transitionTo(next: StreamState) {
        stateMachine.transitionTo(next)
    }

    private fun stopStreamOutputFor(nextState: StreamState): CleanupReport {
        val needsStopping = state == StreamState.STARTING ||
            state == StreamState.STREAMING ||
            state == StreamState.WAITING_NETWORK
        var cleanup = CleanupReport()
        if (needsStopping) {
            if (state != StreamState.STOPPING) {
                subsystemStates = subsystemStates.copy(
                    camera = if (previewState == PreviewState.ACTIVE) SubsystemState.RUNNING else SubsystemState.STOPPING,
                    encoder = SubsystemState.STOPPING,
                    rtspServer = SubsystemState.STOPPING,
                )
                transitionTo(StreamState.STOPPING)
                notifyObservers()
            }
            cleanup = lifecycle.stopStreaming()
            if (cleanup.isSuccessful) {
                subsystemStates = subsystemStates.copy(
                    camera = if (previewState == PreviewState.ACTIVE) SubsystemState.RUNNING else SubsystemState.IDLE,
                    encoder = if (nextState == StreamState.WAITING_NETWORK) {
                        SubsystemState.WAITING_NETWORK
                    } else {
                        SubsystemState.IDLE
                    },
                    rtspServer = if (nextState == StreamState.WAITING_NETWORK) {
                        SubsystemState.WAITING_NETWORK
                    } else {
                        SubsystemState.IDLE
                    },
                )
                transitionTo(nextState)
            } else {
                subsystemStates = subsystemStatesForCleanupFailure(cleanup)
                lastError = StreamErrorFormatter.message(cleanupFailure(cleanup))
                lastErrorKind = StreamErrorKind.THREAD
                transitionTo(StreamState.ERROR)
                StreamErrorLogger.error(cleanupFailure(cleanup))
                notifyObservers()
            }
        } else if (state == StreamState.ERROR && nextState == StreamState.WAITING_NETWORK) {
            // A failed stream remains an explicit error until the user retries;
            // losing Wi-Fi must not turn that diagnostic state into a fake wait.
        } else if (state != nextState) {
            subsystemStates = subsystemStates.copy(
                camera = if (previewState == PreviewState.ACTIVE) SubsystemState.RUNNING else SubsystemState.IDLE,
                encoder = SubsystemState.IDLE,
                rtspServer = SubsystemState.IDLE,
            )
            transitionTo(nextState)
        }
        return cleanup
    }

    private fun stopPreviewPipelineFor(): CleanupReport {
        if (lifecycle.activePipeline == null) {
            previewState = PreviewState.IDLE
            stopOrientationMonitoring()
            subsystemStates = StreamSubsystemSnapshot()
            if (state != StreamState.STOPPED && state != StreamState.ERROR) transitionTo(StreamState.STOPPED)
            return CleanupReport()
        }
        if (state == StreamState.STARTING || state == StreamState.STREAMING || state == StreamState.WAITING_NETWORK) {
            transitionTo(StreamState.STOPPING)
        }
        val cleanup = lifecycle.stop()
        if (cleanup.isSuccessful) {
            previewState = PreviewState.IDLE
            previewError = null
            previewErrorKind = null
            stopOrientationMonitoring()
            subsystemStates = StreamSubsystemSnapshot()
            if (state == StreamState.STOPPING) transitionTo(StreamState.STOPPED)
        } else {
            lastError = StreamErrorFormatter.message(cleanupFailure(cleanup))
            lastErrorKind = StreamErrorKind.THREAD
            if (state != StreamState.ERROR) transitionTo(StreamState.ERROR)
        }
        return cleanup
    }

    private fun prepareSubsystemsForPreview() {
        subsystemStates = StreamSubsystemSnapshot(camera = SubsystemState.STARTING)
    }

    private fun prepareSubsystemsForStreamStart() {
        subsystemStates = subsystemStates.copy(
            camera = if (previewState == PreviewState.ACTIVE) SubsystemState.RUNNING else SubsystemState.STARTING,
            encoder = SubsystemState.STARTING,
            rtspServer = SubsystemState.IDLE,
        )
    }

    private fun setPreviewError(failure: StreamFailure) {
        previewState = PreviewState.ERROR
        previewError = StreamErrorFormatter.message(failure)
        previewErrorKind = failure.kind
        subsystemStates = StreamSubsystemSnapshot(camera = SubsystemState.ERROR)
        StreamErrorLogger.error(failure)
    }

    private fun onSubsystemStateChanged(subsystem: StreamSubsystem, next: SubsystemState) {
        subsystemStates = subsystemStates.withState(subsystem, next)
        scheduleNotifyObservers()
    }

    private fun cancelRetry() {
        retryRunnable?.let(mainHandler::removeCallbacks)
        retryRunnable = null
    }

    private fun onControl(action: () -> Unit) {
        if (Looper.myLooper() == Looper.getMainLooper()) {
            action()
        } else {
            if (!mainHandler.post(action)) {
                StreamErrorLogger.error(
                    StreamFailure(StreamErrorKind.THREAD, "service control callback dispatch rejected"),
                )
            }
        }
    }

    private fun scheduleRetry() {
        if (
            state != StreamState.ERROR ||
            previewState != PreviewState.ACTIVE ||
            wifiNetwork == null ||
            retryRunnable != null ||
            lifecycle.activePipeline == null
        ) return
        if (retryAttempt >= MAX_RETRY_ATTEMPTS) {
            lastError = buildString {
                append(lastError ?: "Errore stream")
                append("; retry automatico esaurito: premi RIPROVA")
            }
            notifyObservers()
            return
        }
        val attempt = retryAttempt++
        val delayMs = (2_000L shl attempt.coerceAtMost(5)).coerceAtMost(MAX_RETRY_DELAY_MS)
        retryRunnable = Runnable {
            retryRunnable = null
            if (state == StreamState.ERROR && previewState == PreviewState.ACTIVE && lifecycle.activePipeline != null && wifiNetwork != null) {
                transitionTo(StreamState.STARTING)
                streamStartRequested = true
                lastError = null
                lastErrorKind = null
                prepareSubsystemsForStreamStart()
                notifyObservers()
                requestStreamingOutput()
            }
        }.also { mainHandler.postDelayed(it, delayMs) }
    }

    private fun createSnapshot(): ServiceSnapshot {
        val currentMetrics = lifecycle.activePipeline?.metrics?.snapshot()
            ?: StreamMetrics(reconnectCount, sessionRestartCount).snapshot()
        val ip = wifiNetwork?.ipAddress
        return ServiceSnapshot(
            state = state,
            wifiConnected = wifiNetwork != null,
            ssid = wifiNetwork?.ssid,
            localIp = ip,
            rtspUrl = StreamUrlBuilder.sanitizedUrl(settings.stream, ip),
            metrics = currentMetrics,
            // Observers/UI receive a redacted settings snapshot. The plaintext
            // password remains private to the service pipeline and credential store.
            settings = AppSettings(stream = settings.stream),
            // Keep stream and preview failures independent. A preview error is
            // projected through previewError below and must not turn into the
            // stream's lastError, otherwise a healthy stream can be presented
            // as a failed stream (and vice versa).
            lastError = lastError ?: currentMetrics.lastError,
            lastErrorKind = lastErrorKind,
            subsystems = subsystemStates,
            orientation = orientationState,
            previewState = previewState,
            previewError = previewError,
            previewErrorKind = previewErrorKind,
        )
    }

    private fun notifyObservers() {
        val snapshot = createSnapshot()
        observers.forEach { observer ->
            try {
                observer.onSnapshot(snapshot)
            } catch (error: Exception) {
                StreamErrorLogger.observer(error)
            }
        }
        updateNotification(snapshot)
    }

    private fun scheduleNotifyObservers() {
        if (!notificationUpdatePending.compareAndSet(false, true)) return
        mainHandler.postDelayed(notificationUpdateRunnable, 250L)
    }

    private fun ensureForegroundFailure(): StreamFailure? {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            return StreamFailure(
                StreamErrorKind.PERMISSION,
                "camera permission is not granted",
                retryable = false,
            )
        }
        try {
            val notification = buildNotification(createSnapshot())
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q &&
                checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
            ) {
                startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA)
            } else {
                @Suppress("DEPRECATION")
                startForeground(NOTIFICATION_ID, notification)
            }
            StreamErrorLogger.info("FOREGROUND_SERVICE_ACTIVE type=camera")
            return null
        } catch (error: Exception) {
            return StreamErrorFormatter.fromThrowable(
                if (error is SecurityException) StreamErrorKind.PERMISSION else StreamErrorKind.CONFIGURATION,
                error,
                retryable = false,
            )
        }
    }

    private fun cleanupFailure(cleanup: CleanupReport): StreamFailure =
        StreamErrorFormatter.fromMessage(
            StreamErrorKind.THREAD,
            "stream cleanup failed: " + cleanup.failures.joinToString("; ") { "${it.resource}: ${it.detail}" },
            retryable = false,
        )

    private fun updateNotification(snapshot: ServiceSnapshot) {
        if (snapshot.state == StreamState.STOPPED && snapshot.previewState == PreviewState.IDLE) return
        getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, buildNotification(snapshot))
    }

    private fun removeForegroundNotification() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            stopForeground(STOP_FOREGROUND_REMOVE)
        } else {
            @Suppress("DEPRECATION")
            stopForeground(true)
        }
    }

    private fun buildNotification(snapshot: ServiceSnapshot): Notification {
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val text = buildString {
            append(snapshot.state.name)
            append(" • ").append(snapshot.metrics.connectedClients).append(" client")
        }
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, CHANNEL_ID)
                .setSmallIcon(android.R.drawable.ic_menu_camera)
                .setContentTitle(getString(R.string.app_name))
                .setContentText(text)
                .setContentIntent(pendingIntent)
                .setVisibility(Notification.VISIBILITY_PRIVATE)
                .setOngoing(true)
                .build()
        } else {
            @Suppress("DEPRECATION")
            Notification.Builder(this)
                .setSmallIcon(android.R.drawable.ic_menu_camera)
                .setContentTitle(getString(R.string.app_name))
                .setContentText(text)
                .setContentIntent(pendingIntent)
                .setVisibility(Notification.VISIBILITY_PRIVATE)
                .setOngoing(true)
                .build()
        }
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                getString(R.string.notification_channel_name),
                NotificationManager.IMPORTANCE_LOW,
            ).apply { description = getString(R.string.notification_channel_description) },
        )
    }

    companion object {
        const val ACTION_START = "com.localsecuritycam.android.START"
        const val ACTION_START_PREVIEW = "com.localsecuritycam.android.START_PREVIEW"
        const val ACTION_STOP = "com.localsecuritycam.android.STOP"
        private const val MAX_RETRY_ATTEMPTS = 8
        private const val MAX_RETRY_DELAY_MS = 60_000L
        private const val CHANNEL_ID = "camera_streaming"
        private const val NOTIFICATION_ID = 1001
    }
}
