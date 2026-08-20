package com.localsecuritycam.android

import android.Manifest
import android.content.ComponentName
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.view.Surface
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.localsecuritycam.android.camera.CameraCapabilitiesProvider
import com.localsecuritycam.android.camera.PreviewDiagnosticMode
import com.localsecuritycam.android.camera.PreviewPatternRenderer
import com.localsecuritycam.android.camera.PreviewSurfaceAttachment
import com.localsecuritycam.android.camera.PREVIEW_DIAGNOSTIC_EXTRA
import com.localsecuritycam.android.diagnostics.StreamErrorFormatter
import com.localsecuritycam.android.diagnostics.StreamErrorKind
import com.localsecuritycam.android.diagnostics.StreamErrorLogger
import com.localsecuritycam.android.service.CameraStreamingService
import com.localsecuritycam.android.service.ServiceObserver
import com.localsecuritycam.android.service.ServiceSnapshot
import com.localsecuritycam.android.settings.AppSettings
import com.localsecuritycam.android.settings.CameraLens
import com.localsecuritycam.android.settings.SettingsRepository
import com.localsecuritycam.android.ui.LocalCamApp
import com.localsecuritycam.android.viewmodel.CameraDestination
import com.localsecuritycam.android.viewmodel.CameraUiEffect
import com.localsecuritycam.android.viewmodel.CameraViewModel
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    private val viewModel: CameraViewModel by lazy {
        ViewModelProvider(this)[CameraViewModel::class.java]
    }

    private var configuredPassword: String? by mutableStateOf(null)
    private var lastPreviewSurface: Surface? = null
    private var lastPreviewWidth = 0
    private var lastPreviewHeight = 0
    private var previewBindingGeneration: Long? = null
    private var serviceBound = false
    private var binder: CameraStreamingService.LocalBinder? = null
    private var pendingStart = false
    private var pendingSettings: AppSettings? = null
    private var previewDiagnosticMode = PreviewDiagnosticMode.NORMAL
    private val previewPatternRenderer = PreviewPatternRenderer()
    private var capabilityQueryGeneration = 0

    private val observer = ServiceObserver { snapshot ->
        runOnUiThread {
            viewModel.onSnapshot(snapshot)
            updateKeepAwake(snapshot.settings.stream.keepScreenAwake)
        }
    }

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            val connectedBinder = service as? CameraStreamingService.LocalBinder ?: return
            binder = connectedBinder
            serviceBound = true
            connectedBinder.addObserver(observer)
            val settings = connectedBinder.settings()
            configuredPassword = settings.password
            viewModel.onSettings(settings)
            connectedBinder.setDisplayRotation(currentDisplayRotationDegrees())
            previewBindingGeneration = connectedBinder.beginPreviewBinding()
            reattachLastPreview()
            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                connectedBinder.startPreview()
            }
            pendingSettings?.let {
                pendingSettings = null
                connectedBinder.saveSettings(it)
            }
            viewModel.requestCapabilities(settings.stream.lens)
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            serviceBound = false
            binder = null
            previewBindingGeneration = null
        }
    }

    private val requestPermissions = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { granted ->
        if (granted[Manifest.permission.CAMERA] == true) {
            val lens = binder?.settings()?.stream?.lens ?: SettingsRepository(this).load().stream.lens
            viewModel.requestCapabilities(lens)
            ensureService(action = CameraStreamingService.ACTION_START_PREVIEW)
            if (pendingStart) {
                pendingStart = false
                ensureService(action = CameraStreamingService.ACTION_START)
            }
        } else {
            pendingStart = false
            binder?.reportPreviewPermissionDenied()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        previewDiagnosticMode = if (isDebuggableBuild()) {
            PreviewDiagnosticMode.fromWireValue(intent?.getStringExtra(PREVIEW_DIAGNOSTIC_EXTRA))
        } else {
            PreviewDiagnosticMode.NORMAL
        }
        viewModel.restoreUiState(savedInstanceState)
        configureFullscreenPreview()
        setContent {
            val state by viewModel.state.collectAsStateWithLifecycle()
            LocalCamApp(
                uiState = state,
                existingPassword = configuredPassword,
                onSurfaceAvailable = ::onPreviewSurfaceAvailable,
                onSurfaceDestroyed = ::onPreviewSurfaceDestroyed,
                onTogglePanel = viewModel::toggleControlPanel,
                onStreamAction = viewModel::onStreamActionClicked,
                onOpenSetup = { viewModel.openDestination(CameraDestination.SETUP) },
                onOpenDiagnostics = { viewModel.openDestination(CameraDestination.DIAGNOSTICS) },
                onClosePanel = viewModel::closeDestination,
                onLensChanged = viewModel::requestCapabilities,
                onSaveSettings = viewModel::submitSettings,
                onValidationError = viewModel::showSettingsError,
            )
        }
        onBackPressedDispatcher.addCallback(
            this,
            object : OnBackPressedCallback(true) {
                override fun handleOnBackPressed() {
                    if (viewModel.state.value.activeDestination != CameraDestination.PREVIEW) {
                        viewModel.closeDestination()
                    } else {
                        finish()
                    }
                }
            },
        )
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.effects.collect(::handleUiEffect)
            }
        }

        if (previewDiagnosticMode == PreviewDiagnosticMode.PATTERN) {
            StreamErrorLogger.info("PREVIEW_PATTERN_MODE requested")
        } else {
            ensureService()
            requestRequiredPermissions()
            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                ensureService(action = CameraStreamingService.ACTION_START_PREVIEW)
            }
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        viewModel.saveUiState(outState)
        super.onSaveInstanceState(outState)
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) hideSystemBars()
    }

    override fun onResume() {
        super.onResume()
        if (previewDiagnosticMode == PreviewDiagnosticMode.PATTERN) return
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            binder?.startPreview()
        }
        binder?.setDisplayRotation(currentDisplayRotationDegrees())
        reattachLastPreview()
    }

    override fun onPause() {
        if (previewDiagnosticMode == PreviewDiagnosticMode.PATTERN) {
            previewPatternRenderer.stop()
        } else {
            previewBindingGeneration?.let { binder?.detachPreview(it) }
        }
        super.onPause()
    }

    override fun onDestroy() {
        previewPatternRenderer.stop()
        if (serviceBound) {
            binder?.removeObserver(observer)
            unbindService(serviceConnection)
            serviceBound = false
            binder = null
        }
        super.onDestroy()
    }

    private fun onPreviewSurfaceAvailable(surface: Surface, width: Int, height: Int) {
        if (previewDiagnosticMode == PreviewDiagnosticMode.PATTERN) {
            previewPatternRenderer.start(surface, width, height)
        } else {
            attachPreview(surface, width, height)
        }
    }

    private fun onPreviewSurfaceDestroyed() {
        if (previewDiagnosticMode == PreviewDiagnosticMode.PATTERN) {
            previewPatternRenderer.stop()
        } else {
            lastPreviewSurface = null
            lastPreviewWidth = 0
            lastPreviewHeight = 0
            previewBindingGeneration?.let { binder?.detachPreview(it) }
        }
    }

    private fun attachPreview(surface: Surface, width: Int, height: Int) {
        lastPreviewSurface = surface
        lastPreviewWidth = width
        lastPreviewHeight = height
        previewBindingGeneration?.let { generation ->
            binder?.attachPreview(
                generation = generation,
                surface = PreviewSurfaceAttachment(surface, width, height),
            )
        }
        binder?.setDisplayRotation(currentDisplayRotationDegrees())
    }

    private fun reattachLastPreview() {
        val surface = lastPreviewSurface?.takeIf { it.isValid } ?: return
        val generation = previewBindingGeneration ?: return
        binder?.attachPreview(
            generation = generation,
            surface = PreviewSurfaceAttachment(surface, lastPreviewWidth, lastPreviewHeight),
        )
    }

    private fun handleUiEffect(effect: CameraUiEffect) {
        when (effect) {
            CameraUiEffect.StartStream -> startStreaming()
            CameraUiEffect.StopStream -> stopStreaming()
            is CameraUiEffect.SaveSettings -> saveSettings(effect.settings)
            is CameraUiEffect.QueryCapabilities -> queryCapabilities(effect.lens)
        }
    }

    private fun startStreaming() {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            pendingStart = true
            ensureService()
            requestRequiredPermissions()
            return
        }
        pendingStart = false
        ensureService(action = CameraStreamingService.ACTION_START)
    }

    private fun stopStreaming() {
        pendingStart = false
        binder?.stopStreaming() ?: run {
            startService(
                Intent(this, CameraStreamingService::class.java)
                    .setAction(CameraStreamingService.ACTION_STOP),
            )
        }
    }

    private fun saveSettings(value: AppSettings) {
        configuredPassword = if (value.stream.authEnabled) value.password else null
        pendingSettings = value
        ensureService()
        binder?.let {
            pendingSettings = null
            it.saveSettings(value)
        }
    }

    private fun ensureService(action: String? = null) {
        try {
            if (action != null) {
                val intent = Intent(this, CameraStreamingService::class.java).apply {
                    this.action = action
                    if (
                        previewDiagnosticMode != PreviewDiagnosticMode.NORMAL &&
                            previewDiagnosticMode != PreviewDiagnosticMode.PATTERN
                    ) {
                        putExtra(PREVIEW_DIAGNOSTIC_EXTRA, previewDiagnosticMode.wireValue)
                    }
                }
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O &&
                    checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
                ) {
                    ContextCompat.startForegroundService(this, intent)
                } else {
                    startService(intent)
                }
            }
            if (!serviceBound) {
                bindService(
                    Intent(this, CameraStreamingService::class.java),
                    serviceConnection,
                    BIND_AUTO_CREATE,
                )
            }
        } catch (error: Exception) {
            StreamErrorLogger.error(
                StreamErrorFormatter.fromThrowable(
                    StreamErrorKind.CONFIGURATION,
                    error,
                    retryable = false,
                ),
            )
        }
    }

    private fun queryCapabilities(lens: CameraLens) {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return
        val generation = ++capabilityQueryGeneration
        Thread {
            runCatching { CameraCapabilitiesProvider(this).query(lens) }
                .onSuccess { capabilities ->
                    runOnUiThread {
                        if (generation == capabilityQueryGeneration) viewModel.onCapabilities(capabilities)
                    }
                }
                .onFailure {
                    runOnUiThread {
                        if (generation == capabilityQueryGeneration) viewModel.onCapabilitiesUnavailable()
                    }
                }
        }.apply {
            name = "camera-capability-query"
            isDaemon = true
            start()
        }
    }

    private fun requestRequiredPermissions() {
        val missing = buildList {
            if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                add(Manifest.permission.CAMERA)
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
                checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
            ) {
                add(Manifest.permission.POST_NOTIFICATIONS)
            }
        }
        if (missing.isNotEmpty()) requestPermissions.launch(missing.toTypedArray())
    }

    private fun updateKeepAwake(enabled: Boolean) {
        if (enabled) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    private fun configureFullscreenPreview() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        hideSystemBars()
    }

    private fun hideSystemBars() {
        WindowCompat.getInsetsController(window, window.decorView).apply {
            systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            hide(WindowInsetsCompat.Type.systemBars())
        }
    }

    private fun isDebuggableBuild(): Boolean =
        applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0

    private fun currentDisplayRotationDegrees(): Int = when (window.decorView.display?.rotation ?: Surface.ROTATION_0) {
        Surface.ROTATION_90 -> 90
        Surface.ROTATION_180 -> 180
        Surface.ROTATION_270 -> 270
        else -> 0
    }
}
