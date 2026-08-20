package com.localsecuritycam.android

import android.Manifest
import android.app.Activity
import android.app.ActivityManager
import android.content.Intent
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.Bitmap
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.view.PixelCopy
import android.util.Log
import android.view.SurfaceView
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.rule.GrantPermissionRule
import androidx.test.rule.ActivityTestRule
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.runner.lifecycle.ActivityLifecycleMonitorRegistry
import androidx.test.runner.lifecycle.Stage
import com.localsecuritycam.android.service.CameraStreamingService
import com.localsecuritycam.android.settings.SettingsRepository
import com.localsecuritycam.android.diagnostics.PreviewPixelAnalyzer
import com.localsecuritycam.android.diagnostics.PreviewPixelMetrics
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.FileOutputStream

/**
 * Hardware-gated smoke test for the real Camera2 -> SurfaceTexture -> SurfaceView path.
 * The test deliberately records artifacts instead of claiming that an offline JVM test
 * proves EGL, physical rotation, MediaCodec, or RTSP behavior.
 */
@RunWith(AndroidJUnit4::class)
class PreviewInstrumentationTest {
    @get:Rule
    val permissionRule: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @get:Rule
    val activityRule = ActivityTestRule(MainActivity::class.java, true, false)

    @Test
    fun cameraPreviewSurfaceAndDiagnosticsArtifactsAreCaptured() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        assumeTrue(context.packageManager.hasSystemFeature("android.hardware.camera.any"))
        assertEquals(
            PackageManager.PERMISSION_GRANTED,
            context.checkSelfPermission(Manifest.permission.CAMERA),
        )
        seedAuthenticatedSettingsIfRequested(context)

        val activity = activityRule.launchActivity(
            android.content.Intent(context, MainActivity::class.java)
                .setAction(CameraStreamingService.ACTION_START_PREVIEW),
        )
        val preview = activity.findViewById<SurfaceView>(R.id.camera_preview)
        assertNotNull("MainActivity must expose the SurfaceView preview", preview)
        waitForSurface(preview)
        waitForLogLine("preview_first_frame_presented")
        waitForLogLine("FOREGROUND_SERVICE_ACTIVE type=camera")
        waitForForegroundCameraService(context)

        waitForComposeSemantics(activity, "Preview active · stream not started")
        val landscapeFirstFrameBefore = countLogMarker("preview_first_frame_presented")

        instrumentation.runOnMainSync {
            activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
        }
        val landscapeActivity = waitForOrientation(Configuration.ORIENTATION_LANDSCAPE)
        val landscapePreview = landscapeActivity.findViewById<SurfaceView>(R.id.camera_preview)
        waitForSurface(landscapePreview)
        waitForLogMarkerIncrease("preview_first_frame_presented", landscapeFirstFrameBefore)
        waitForComposeSemantics(landscapeActivity, "Preview active · stream not started")
        val landscapeArtifacts = File(context.cacheDir, "preview-instrumentation/landscape").apply { mkdirs() }
        val landscapeFrames = captureSeries(landscapePreview, landscapeArtifacts, "surface", 3)
        assertTrue(
            "landscape preview should contain non-black pixels",
            landscapeFrames.any { it.nonBlackRatio > 0.01 },
        )
        assertTrue(
            "landscape preview should change between frames",
            landscapeFrames.map { it.frameHash }.distinct().size >= 2,
        )
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            instrumentation.uiAutomation.takeScreenshot()?.let { screenshot ->
                writePng(screenshot, File(landscapeArtifacts, "landscape.png"))
            }
        } else {
            val screenshot = captureWindow(landscapeActivity)
            writePng(screenshot, File(landscapeArtifacts, "landscape.png"))
            screenshot.recycle()
        }
        File(landscapeArtifacts, "ui-tree.txt").writeText(dumpAccessibilityTree())
        instrumentation.runOnMainSync {
            landscapeActivity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        }
        val portraitActivity = waitForOrientation(Configuration.ORIENTATION_PORTRAIT)
        waitForSurface(portraitActivity.findViewById(R.id.camera_preview))
        waitForLogLine("preview_first_frame_presented")
        waitForComposeSemantics(portraitActivity, "Preview active · stream not started")

        val artifactDir = File(context.cacheDir, "preview-instrumentation").apply { mkdirs() }
        val uiTree = File(artifactDir, "ui-tree.txt")
        uiTree.writeText(dumpAccessibilityTree())

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val screenshot = instrumentation.uiAutomation.takeScreenshot()
            if (screenshot != null) writePng(screenshot, File(artifactDir, "preview.png"))
        }
        val logcat = captureLogcat()
        File(artifactDir, "logcat.txt").writeText(logcat)
        Log.i(
            TAG,
            "preview instrumentation artifacts=${artifactDir.absolutePath} " +
                "surfaceValid=${preview.holder.surface.isValid}",
        )
    }

    @Test
    fun eglPatternRendersVisiblePixelsOnTheRealSurfaceView() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        context.stopService(Intent(context, CameraStreamingService::class.java))

        val activity = activityRule.launchActivity(
            Intent(context, MainActivity::class.java)
                .putExtra("preview_diagnostic", "pattern"),
        )
        val preview = activity.findViewById<SurfaceView>(R.id.camera_preview)
        waitForSurface(preview)
        waitForLogLine("PREVIEW_PATTERN_RENDERED")

        val artifactDir = File(context.cacheDir, "preview-instrumentation/pattern").apply { mkdirs() }
        val bitmap = capturePixelCopy(preview)
        val metrics = analyzeBitmap(bitmap)
        writePng(bitmap, File(artifactDir, "pattern-pixelcopy.png"))
        File(artifactDir, "pattern-metrics.txt").writeText(PreviewPixelAnalyzer.toLogLine(metrics))
        bitmap.recycle()

        assertEquals(1080, metrics.width)
        assertEquals(2340, metrics.height)
        assertTrue("pattern should contain visible pixels", metrics.nonBlackRatio >= 0.95)
        assertTrue("pattern should be bright enough", metrics.meanLuma >= 32.0)
    }

    @Test
    fun cameraPreviewIdentityModeProducesChangingPixelFrames() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        assumeTrue(context.packageManager.hasSystemFeature("android.hardware.camera.any"))

        val activity = activityRule.launchActivity(
            Intent(context, MainActivity::class.java)
                .setAction(CameraStreamingService.ACTION_START_PREVIEW)
                .putExtra("preview_diagnostic", "oes_identity"),
        )
        val preview = activity.findViewById<SurfaceView>(R.id.camera_preview)
        waitForSurface(preview)
        waitForLogLine("PREVIEW_FIRST_FRAME_RENDERED")

        val artifactDir = File(context.cacheDir, "preview-instrumentation/oes-identity").apply { mkdirs() }
        val metrics = captureSeries(preview, artifactDir, "identity", 3)
        assertTrue("camera identity preview should contain non-black pixels", metrics.any { it.nonBlackRatio > 0.01 })
        assertTrue(
            "camera identity preview should change between frames",
            metrics.map { it.frameHash }.distinct().size >= 2,
        )
    }

    @Test
    fun cameraPreviewRotationModeProducesChangingPixelFrames() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        assumeTrue(context.packageManager.hasSystemFeature("android.hardware.camera.any"))

        val activity = activityRule.launchActivity(
            Intent(context, MainActivity::class.java)
                .setAction(CameraStreamingService.ACTION_START_PREVIEW)
                .putExtra("preview_diagnostic", "oes_rotation"),
        )
        val preview = activity.findViewById<SurfaceView>(R.id.camera_preview)
        waitForSurface(preview)
        waitForLogLine("PREVIEW_FIRST_FRAME_RENDERED")

        val artifactDir = File(context.cacheDir, "preview-instrumentation/oes-rotation").apply { mkdirs() }
        val metrics = captureSeries(preview, artifactDir, "rotation", 3)
        assertTrue("camera rotation preview should contain non-black pixels", metrics.any { it.nonBlackRatio > 0.01 })
        assertTrue(
            "camera rotation preview should change between frames",
            metrics.map { it.frameHash }.distinct().size >= 2,
        )
    }

    @Test
    fun cameraPreviewFullModeProducesChangingPixelFrames() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        assumeTrue(context.packageManager.hasSystemFeature("android.hardware.camera.any"))

        val activity = activityRule.launchActivity(
            Intent(context, MainActivity::class.java)
                .setAction(CameraStreamingService.ACTION_START_PREVIEW)
                .putExtra("preview_diagnostic", "full"),
        )
        val preview = activity.findViewById<SurfaceView>(R.id.camera_preview)
        waitForSurface(preview)
        waitForLogLine("PREVIEW_FIRST_FRAME_RENDERED")

        val artifactDir = File(context.cacheDir, "preview-instrumentation/full").apply { mkdirs() }
        val metrics = captureSeries(preview, artifactDir, "full", 3)
        assertTrue("camera full preview should contain non-black pixels", metrics.any { it.nonBlackRatio > 0.01 })
        assertTrue(
            "camera full preview should change between frames",
            metrics.map { it.frameHash }.distinct().size >= 2,
        )
    }

    @Test
    fun activityBackgroundForegroundReattachesPreviewWithoutNewCameraSession() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        assumeTrue(context.packageManager.hasSystemFeature("android.hardware.camera.any"))

        val activity = activityRule.launchActivity(
            Intent(context, MainActivity::class.java)
                .setAction(CameraStreamingService.ACTION_START_PREVIEW),
        )
        val preview = activity.findViewById<SurfaceView>(R.id.camera_preview)
        waitForSurface(preview)
        waitForLogLine("camera_session_configured")
        waitForLogLine("preview_first_frame_presented")

        val configuredBefore = countLogMarker("camera_session_configured")
        val detachedBefore = countLogMarker("preview_surface_detached")
        val attachedBefore = countLogMarker("preview_surface_attached")
        val firstFrameBefore = countLogMarker("preview_first_frame_presented")

        instrumentation.runOnMainSync {
            check(activity.moveTaskToBack(true)) { "Activity could not be moved to background" }
        }
        waitForLogMarkerIncrease("preview_surface_detached", detachedBefore)

        context.startActivity(
            Intent(context, MainActivity::class.java).addFlags(
                Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_REORDER_TO_FRONT,
            ),
        )
        val resumed = waitForResumedActivity()
        val resumedPreview = resumed.findViewById<SurfaceView>(R.id.camera_preview)
        waitForSurface(resumedPreview)
        waitForLogMarkerIncrease("preview_surface_attached", attachedBefore)
        waitForLogMarkerIncrease("preview_first_frame_presented", firstFrameBefore)

        assertEquals(
            "Activity recreation must not configure a second Camera2 capture session",
            configuredBefore,
            countLogMarker("camera_session_configured"),
        )
        waitForForegroundCameraService(context)
    }

    private fun waitForSurface(preview: SurfaceView) {
        val deadline = SystemClock.uptimeMillis() + 5_000L
        while (!preview.holder.surface.isValid && SystemClock.uptimeMillis() < deadline) {
            SystemClock.sleep(50L)
        }
        assertNotNull("SurfaceView holder must expose a valid Surface", preview.holder.surface.takeIf { it.isValid })
    }

    private fun waitForResumedActivity(): MainActivity {
        val deadline = SystemClock.uptimeMillis() + 7_000L
        while (SystemClock.uptimeMillis() < deadline) {
            resumedActivity()?.let { return it }
            SystemClock.sleep(100L)
        }
        throw AssertionError("MainActivity did not return to RESUMED")
    }

    @Suppress("DEPRECATION")
    private fun waitForForegroundCameraService(context: android.content.Context) {
        val activityManager = context.getSystemService(ActivityManager::class.java)
        val deadline = SystemClock.uptimeMillis() + 5_000L
        var foreground = false
        while (SystemClock.uptimeMillis() < deadline) {
            foreground = activityManager.getRunningServices(100).any { service ->
                service.service.packageName == context.packageName &&
                    service.service.className == CameraStreamingService::class.java.name &&
                    service.foreground
            }
            if (foreground) return
            SystemClock.sleep(100L)
        }
        assertEquals("CameraStreamingService must be foreground", true, foreground)
    }

    private fun waitForOrientation(expected: Int): MainActivity {
        val deadline = SystemClock.uptimeMillis() + 7_000L
        var actual = Configuration.ORIENTATION_UNDEFINED
        var current: MainActivity? = null
        while (SystemClock.uptimeMillis() < deadline) {
            current = resumedActivity()
            actual = current?.resources?.configuration?.orientation
                ?: Configuration.ORIENTATION_UNDEFINED
            if (actual == expected && current != null) return current
            SystemClock.sleep(100L)
        }
        assertEquals("Unexpected Activity orientation", expected, actual)
        return current ?: activityRule.activity
    }

    private fun resumedActivity(): MainActivity? {
        var resumed: Activity? = null
        InstrumentationRegistry.getInstrumentation().runOnMainSync {
            resumed = ActivityLifecycleMonitorRegistry.getInstance()
                .getActivitiesInStage(Stage.RESUMED)
                .firstOrNull()
        }
        return resumed as? MainActivity
    }

    private fun waitForComposeSemantics(activity: Activity, expected: String) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val deadline = SystemClock.uptimeMillis() + 5_000L
        var found = false
        while (SystemClock.uptimeMillis() < deadline) {
            instrumentation.runOnMainSync {
                found = accessibilityTreeContains(
                    instrumentation.uiAutomation.rootInActiveWindow,
                    expected,
                )
            }
            if (found) return
            SystemClock.sleep(100L)
        }
        assertTrue("Compose semantics did not expose preview badge: $expected", found)
    }

    private fun accessibilityTreeContains(
        node: android.view.accessibility.AccessibilityNodeInfo?,
        expected: String,
    ): Boolean {
        if (node == null) return false
        val matches = node.text?.toString() == expected ||
            node.contentDescription?.toString() == expected
        if (matches) return true
        for (index in 0 until node.childCount) {
            val child = node.getChild(index) ?: continue
            try {
                if (accessibilityTreeContains(child, expected)) return true
            } finally {
                child.recycle()
            }
        }
        return false
    }

    private fun waitForLogLine(marker: String) {
        val deadline = SystemClock.uptimeMillis() + 10_000L
        while (SystemClock.uptimeMillis() < deadline) {
            if (captureLogcat().contains(marker)) return
            SystemClock.sleep(100L)
        }
        throw AssertionError("Timed out waiting for log marker: $marker")
    }

    private fun waitForLogMarkerIncrease(marker: String, previousCount: Int) {
        val deadline = SystemClock.uptimeMillis() + 10_000L
        while (SystemClock.uptimeMillis() < deadline) {
            if (countLogMarker(marker) > previousCount) return
            SystemClock.sleep(100L)
        }
        throw AssertionError("Timed out waiting for a new log marker: $marker")
    }

    private fun countLogMarker(marker: String): Int =
        captureLogcat().lineSequence().count { it.contains(marker) }

    private fun dumpAccessibilityTree(): String {
        val automation = InstrumentationRegistry.getInstrumentation().uiAutomation
        val root = automation.rootInActiveWindow ?: return "<no active accessibility root>"
        return buildString {
            appendNode(root, 0)
        }
    }

    private fun StringBuilder.appendNode(node: android.view.accessibility.AccessibilityNodeInfo, depth: Int) {
        repeat(depth) { append("  ") }
        append(node.className ?: "?")
        node.viewIdResourceName?.let { append(" id=").append(it) }
        node.text?.let { append(" text=").append(it) }
        append('\n')
        for (index in 0 until node.childCount) {
            node.getChild(index)?.let { child ->
                appendNode(child, depth + 1)
                child.recycle()
            }
        }
    }

    private fun writePng(bitmap: Bitmap, file: File) {
        FileOutputStream(file).use { output ->
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, output)
        }
    }

    private fun capturePixelCopy(preview: SurfaceView): Bitmap {
        val bitmap = Bitmap.createBitmap(
            preview.width.coerceAtLeast(1),
            preview.height.coerceAtLeast(1),
            Bitmap.Config.ARGB_8888,
        )
        val thread = HandlerThread("preview-pixelcopy").apply { start() }
        val handler = Handler(thread.looper)
        val result = IntArray(1) { Int.MIN_VALUE }
        val completed = java.util.concurrent.CountDownLatch(1)
        val requested = runCatching {
            PixelCopy.request(
                preview,
                bitmap,
                { copyResult ->
                    result[0] = copyResult
                    completed.countDown()
                },
                handler,
            )
        }.getOrElse { error ->
            thread.quitSafely()
            bitmap.recycle()
            throw AssertionError("PixelCopy request failed", error)
        }
        assertTrue("PixelCopy request was not accepted", requested == Unit)
        assertTrue("PixelCopy timed out", completed.await(3, java.util.concurrent.TimeUnit.SECONDS))
        thread.quitSafely()
        assertEquals("PixelCopy failed", PixelCopy.SUCCESS, result[0])
        return bitmap
    }

    private fun captureWindow(activity: Activity): Bitmap {
        val decor = activity.window.decorView
        val bitmap = Bitmap.createBitmap(
            decor.width.coerceAtLeast(1),
            decor.height.coerceAtLeast(1),
            Bitmap.Config.ARGB_8888,
        )
        val thread = HandlerThread("window-pixelcopy").apply { start() }
        val handler = Handler(thread.looper)
        val result = IntArray(1) { Int.MIN_VALUE }
        val completed = java.util.concurrent.CountDownLatch(1)
        PixelCopy.request(
            activity.window,
            bitmap,
            { copyResult ->
                result[0] = copyResult
                completed.countDown()
            },
            handler,
        )
        assertTrue("Window PixelCopy timed out", completed.await(3, java.util.concurrent.TimeUnit.SECONDS))
        thread.quitSafely()
        assertEquals("Window PixelCopy failed", PixelCopy.SUCCESS, result[0])
        return bitmap
    }

    private fun analyzeBitmap(bitmap: Bitmap, previousHash: Long? = null): PreviewPixelMetrics {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        return PreviewPixelAnalyzer.analyze(bitmap.width, bitmap.height, pixels, previousHash)
    }

    private fun captureSeries(
        preview: SurfaceView,
        artifactDir: File,
        prefix: String,
        count: Int,
    ): List<PreviewPixelMetrics> {
        var previousHash: Long? = null
        return (0 until count).map { index ->
            SystemClock.sleep(250L)
            val bitmap = capturePixelCopy(preview)
            val metrics = analyzeBitmap(bitmap, previousHash)
            previousHash = metrics.frameHash
            writePng(bitmap, File(artifactDir, "$prefix-$index.png"))
            File(artifactDir, "$prefix-$index.txt").writeText(PreviewPixelAnalyzer.toLogLine(metrics))
            bitmap.recycle()
            metrics
        }
    }

    private fun captureLogcat(): String = runCatching {
        Runtime.getRuntime()
            .exec(arrayOf("logcat", "-d", "-t", "250"))
            .inputStream
            .bufferedReader()
            .use { it.readText() }
    }.getOrElse { error -> "logcat capture failed: ${error.message}" }

    /** Optional manual-QA hook; the normal instrumentation run never mutates stream settings. */
    private fun seedAuthenticatedSettingsIfRequested(context: android.content.Context) {
        if (InstrumentationRegistry.getArguments().getString("seed_auth") != "true") return
        val repository = SettingsRepository(context)
        val current = repository.load()
        repository.save(
            current.copy(
                stream = current.stream.copy(authEnabled = true, username = "camera"),
                password = "local-pass-123",
            ),
        )
        Log.i(TAG, "seeded authenticated settings for manual emulator QA")
    }

    private companion object {
        const val TAG = "PreviewInstrumentationTest"
    }
}
