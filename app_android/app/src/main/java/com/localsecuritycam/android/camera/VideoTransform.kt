package com.localsecuritycam.android.camera

import com.localsecuritycam.android.settings.CameraLens
import com.localsecuritycam.android.settings.Resolution
import com.localsecuritycam.android.settings.VideoAspectRatio
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.roundToInt

/**
 * The latest device facts received by the foreground service. A physical
 * orientation wins over the Activity display rotation; the latter exists only
 * as a diagnostic and as the fallback on devices without a detectable sensor.
 */
data class DeviceOrientation(
    val physicalOrientationDegrees: Int? = null,
    val displayRotationDegrees: Int = 0,
) {
    init {
        require(physicalOrientationDegrees == null || physicalOrientationDegrees in QUADRANTS) {
            "physical orientation must be one of $QUADRANTS"
        }
    }

    val source: OrientationSource
        get() = if (physicalOrientationDegrees == null) OrientationSource.DISPLAY_FALLBACK else OrientationSource.PHYSICAL_SENSOR

    val targetSurfaceRotationDegrees: Int
        get() = physicalOrientationDegrees?.let(OrientationResolver::targetSurfaceRotationForPhysical)
            ?: normalizeDegrees(displayRotationDegrees)

    private companion object {
        val QUADRANTS = setOf(0, 90, 180, 270)
    }
}

enum class OrientationSource {
    PHYSICAL_SENSOR,
    DISPLAY_FALLBACK,
}

/**
 * Single orientation contract shared by Camera2, EGL preview, MediaCodec and
 * diagnostics. Preview and stream mirror flags intentionally remain distinct,
 * even though the local-camera policy currently keeps both false.
 */
data class CameraOrientationState(
    val sensorOrientationDegrees: Int,
    val lensFacing: CameraLens,
    val physicalOrientationDegrees: Int?,
    val displayRotationDegrees: Int,
    val targetSurfaceRotationDegrees: Int,
    val requestedRotationDegrees: Int,
    val mirrorPreview: Boolean,
    val mirrorStream: Boolean,
    val bufferResolution: Resolution,
    val outputResolution: Resolution,
    val source: OrientationSource,
    val outputRotationDegrees: Int = requestedRotationDegrees,
    val outputAspectRatio: VideoAspectRatio? = null,
) {
    /** Camera2 relative rotation, converted to clockwise pixel rotation. */
    val pixelClockwiseRotationDegrees: Int
        get() = OrientationResolver.pixelClockwiseRotationDegrees(
            cameraRelativeRotationDegrees = requestedRotationDegrees,
            lensFacing = lensFacing,
        )

    /**
     * Camera-only rotation expected by [android.opengl.Matrix.rotateM] when
     * the SurfaceTexture producer has not already rotated the buffer.
     */
    val glTextureRotationDegrees: Int
        get() = OrientationResolver.glTextureRotationDegrees(pixelClockwiseRotationDegrees)

    /** Final clockwise rotation after the optional 16:9/9:16 output choice. */
    val outputPixelClockwiseRotationDegrees: Int
        get() = normalizeDegrees(outputRotationDegrees)

    val outputGlTextureRotationDegrees: Int
        get() = OrientationResolver.glTextureRotationDegrees(outputPixelClockwiseRotationDegrees)
}

/**
 * Canonical video contract. It describes the pixels produced by the camera
 * and the pixels requested from the encoder; it deliberately contains no
 * widget/Surface dimensions. Viewport fitting is represented separately by
 * [ViewportFit].
 */
data class VideoGeometry(
    val sensorOrientationDegrees: Int,
    val physicalOrientationDegrees: Int?,
    val displayRotationDegrees: Int,
    val targetRotationDegrees: Int,
    val lensFacing: CameraLens,
    val sourceResolution: Resolution,
    val encodedResolution: Resolution,
    val pixelRotationDegrees: Int,
    val mirrorPreview: Boolean,
    val mirrorStream: Boolean,
) {
    val sourceAspectRatio: Double
        get() = sourceResolution.width.toDouble() / sourceResolution.height.toDouble()

    val encodedAspectRatio: Double
        get() = encodedResolution.width.toDouble() / encodedResolution.height.toDouble()
}

/** Single owner of Camera2 orientation math. */
object OrientationResolver {
    /** Maps physical clockwise quadrants to Android Surface rotations. */
    fun targetSurfaceRotationForPhysical(physicalOrientationDegrees: Int): Int = when (
        normalizeDegrees(physicalOrientationDegrees)
    ) {
        0 -> 0
        90 -> 270
        180 -> 180
        270 -> 90
        else -> error("physical orientation must be a right-angle quadrant")
    }

    /**
     * Camera2's official relative-rotation formula. Back cameras use -1 and
     * front cameras +1 for the Surface term.
     */
    fun pixelRotationDegrees(
        sensorOrientationDegrees: Int,
        targetSurfaceRotationDegrees: Int,
        lensFacing: CameraLens,
    ): Int {
        val sign = if (lensFacing == CameraLens.BACK) -1 else 1
        return normalizeDegrees(
            normalizeDegrees(sensorOrientationDegrees) -
                normalizeDegrees(targetSurfaceRotationDegrees) * sign,
        )
    }

    /**
     * Converts Camera2's relative rotation into clockwise pixel rotation.
     * Camera2 reports the front-camera result in the mirrored sensor space,
     * while the local recording contract must remain an ordinary, non-mirrored
     * image for both preview and stream.
     */
    fun pixelClockwiseRotationDegrees(
        cameraRelativeRotationDegrees: Int,
        lensFacing: CameraLens,
    ): Int = if (lensFacing == CameraLens.BACK) {
        normalizeDegrees(cameraRelativeRotationDegrees)
    } else {
        normalizeDegrees(-cameraRelativeRotationDegrees)
    }

    /**
     * OpenGL's positive Z rotation is opposite to clockwise pixel rotation in
     * the texture-coordinate space used by Matrix.rotateM.
     */
    fun glTextureRotationDegrees(pixelClockwiseRotationDegrees: Int): Int =
        normalizeDegrees(-pixelClockwiseRotationDegrees)

    fun resolve(
        sensorOrientationDegrees: Int,
        lensFacing: CameraLens,
        deviceOrientation: DeviceOrientation,
        bufferResolution: Resolution,
        outputAspectRatio: VideoAspectRatio? = null,
    ): CameraOrientationState {
        val rotation = pixelRotationDegrees(
            sensorOrientationDegrees = sensorOrientationDegrees,
            targetSurfaceRotationDegrees = deviceOrientation.targetSurfaceRotationDegrees,
            lensFacing = lensFacing,
        )
        val cameraPixelRotation = pixelClockwiseRotationDegrees(rotation, lensFacing)
        val outputRotation = outputAspectRatio?.let {
            outputRotationForAspect(cameraPixelRotation, it)
        } ?: cameraPixelRotation
        val quarterTurn = outputRotation == 90 || outputRotation == 270
        val output = if (quarterTurn) {
            Resolution(bufferResolution.height, bufferResolution.width)
        } else {
            bufferResolution
        }
        return CameraOrientationState(
            sensorOrientationDegrees = normalizeDegrees(sensorOrientationDegrees),
            lensFacing = lensFacing,
            physicalOrientationDegrees = deviceOrientation.physicalOrientationDegrees,
            displayRotationDegrees = normalizeDegrees(deviceOrientation.displayRotationDegrees),
            targetSurfaceRotationDegrees = deviceOrientation.targetSurfaceRotationDegrees,
            requestedRotationDegrees = rotation,
            // Recorded local-camera output must never become a selfie mirror.
            mirrorPreview = false,
            mirrorStream = false,
            bufferResolution = bufferResolution,
            outputResolution = output,
            source = deviceOrientation.source,
            outputRotationDegrees = outputRotation,
            outputAspectRatio = outputAspectRatio,
        )
    }

    /**
     * Keeps the physical camera direction while changing only the output
     * orientation requested by the user. The smallest right-angle correction
     * changes landscape to portrait or vice versa without a crop/stretch.
     */
    fun outputRotationForAspect(
        cameraPixelRotationDegrees: Int,
        aspectRatio: VideoAspectRatio,
    ): Int {
        val cameraIsPortrait = normalizeDegrees(cameraPixelRotationDegrees) == 90 ||
            normalizeDegrees(cameraPixelRotationDegrees) == 270
        if (cameraIsPortrait == aspectRatio.isPortrait) return normalizeDegrees(cameraPixelRotationDegrees)
        return when (normalizeDegrees(cameraPixelRotationDegrees)) {
            0 -> 90
            90 -> 0
            180 -> 270
            270 -> 180
            else -> error("camera pixel rotation must be a right-angle quadrant")
        }
    }
}

/**
 * Android-free hysteresis/debounce for OrientationEventListener samples.
 * The first valid quadrant applies immediately; later changes must move more
 * than [thresholdDegrees] from the stable quadrant and remain stable for
 * [stableForMs]. Negative values model ORIENTATION_UNKNOWN and retain state.
 */
class PhysicalOrientationStabilizer(
    private val thresholdDegrees: Int = 55,
    private val stableForMs: Long = 350L,
) {
    private var stableQuadrant: Int? = null
    private var pendingQuadrant: Int? = null
    private var pendingSinceMs: Long = 0L

    val latestStableOrientationDegrees: Int?
        get() = stableQuadrant

    fun update(rawOrientationDegrees: Int, elapsedRealtimeMs: Long): Int? {
        if (rawOrientationDegrees !in 0..359) return null
        val candidate = nearestQuadrant(rawOrientationDegrees)
        val current = stableQuadrant
        if (current == null) {
            stableQuadrant = candidate
            pendingQuadrant = null
            return candidate
        }
        if (candidate == current || angularDistance(rawOrientationDegrees, current) <= thresholdDegrees) {
            pendingQuadrant = null
            return null
        }
        if (pendingQuadrant != candidate) {
            pendingQuadrant = candidate
            pendingSinceMs = elapsedRealtimeMs
            return null
        }
        if (elapsedRealtimeMs - pendingSinceMs < stableForMs) return null
        stableQuadrant = candidate
        pendingQuadrant = null
        return candidate
    }

    fun reset() {
        stableQuadrant = null
        pendingQuadrant = null
        pendingSinceMs = 0L
    }

    private fun nearestQuadrant(rawOrientationDegrees: Int): Int =
        normalizeDegrees(((rawOrientationDegrees + 45) / 90) * 90)

    private fun angularDistance(first: Int, second: Int): Int {
        val difference = kotlin.math.abs(normalizeDegrees(first) - normalizeDegrees(second))
        return minOf(difference, 360 - difference)
    }
}

/**
 * The rotation passed to the GL texture-coordinate matrix.
 *
 * SurfaceTexture owns the producer crop and vertical-axis conversion. The
 * remaining rotation must therefore be converted explicitly from Camera2's
 * relative rotation to clockwise pixels and then to Matrix.rotateM's direction.
 */
internal fun textureCoordinateRotationDegrees(
    cameraRelativeRotationDegrees: Int,
    lensFacing: CameraLens,
    surfaceTexturePixelClockwiseRotationDegrees: Int = 0,
    outputPixelClockwiseRotationDegrees: Int? = null,
): Int = OrientationResolver.glTextureRotationDegrees(
    normalizeDegrees(
        (outputPixelClockwiseRotationDegrees ?: OrientationResolver.pixelClockwiseRotationDegrees(
            cameraRelativeRotationDegrees = cameraRelativeRotationDegrees,
            lensFacing = lensFacing,
        )) - surfaceTexturePixelClockwiseRotationDegrees,
    ),
)

/**
 * Returns the clockwise right-angle rotation already encoded by a
 * SurfaceTexture matrix. The matrix's first column is the transformed U axis;
 * crop/scaling do not change its angle. SurfaceTexture's mandatory V flip is
 * intentionally left in the matrix and is not applied a second time here.
 */
internal fun surfaceTexturePixelClockwiseRotationDegrees(matrix: FloatArray): Int {
    require(matrix.size >= 16) { "SurfaceTexture matrix must contain 16 values" }
    val uX = matrix[0]
    val uY = matrix[1]
    if (abs(uX) < 0.0001f && abs(uY) < 0.0001f) return 0
    val glAngleDegrees = Math.toDegrees(atan2(uY.toDouble(), uX.toDouble()))
    val snappedGlAngleDegrees = (glAngleDegrees / 90.0).roundToInt() * 90
    return normalizeDegrees(-snappedGlAngleDegrees)
}

/**
 * Deterministic geometry contract shared by Camera2, the GL renderer, and
 * the two render destinations. Orientation and FIT_CENTER stay separate even
 * though the renderer applies them in one draw pass.
 */
data class VideoTransform(
    val orientation: CameraOrientationState,
    val sourceWidth: Int,
    val sourceHeight: Int,
    val targetWidth: Int,
    val targetHeight: Int,
    val logicalWidth: Int,
    val logicalHeight: Int,
    val sourceAspectRatio: Double,
    val destinationAspectRatio: Double,
    val scaleX: Float,
    val scaleY: Float,
    val uniformScale: Float,
) {
    /** The single orientation/encoded-size authority used by both targets. */
    val geometry: VideoGeometry
        get() = VideoGeometry(
            sensorOrientationDegrees = orientation.sensorOrientationDegrees,
            physicalOrientationDegrees = orientation.physicalOrientationDegrees,
            displayRotationDegrees = orientation.displayRotationDegrees,
            targetRotationDegrees = orientation.targetSurfaceRotationDegrees,
            lensFacing = orientation.lensFacing,
            sourceResolution = orientation.bufferResolution,
            encodedResolution = orientation.outputResolution,
            pixelRotationDegrees = outputRotationDegrees,
            mirrorPreview = mirrorPreview,
            mirrorStream = mirrorStream,
        )

    /** FIT_CENTER only; it never changes [geometry.encodedResolution]. */
    val viewportFit: ViewportFit
        get() = ViewportFit(
            scaleX = scaleX,
            scaleY = scaleY,
            uniformScale = uniformScale,
            contentWidth = logicalWidth.toFloat() * uniformScale,
            contentHeight = logicalHeight.toFloat() * uniformScale,
        )

    val rotationDegrees: Int
        get() = orientation.requestedRotationDegrees

    val pixelClockwiseRotationDegrees: Int
        get() = orientation.pixelClockwiseRotationDegrees

    val glTextureRotationDegrees: Int
        get() = orientation.glTextureRotationDegrees

    val outputRotationDegrees: Int
        get() = orientation.outputRotationDegrees

    val outputPixelClockwiseRotationDegrees: Int
        get() = orientation.outputPixelClockwiseRotationDegrees

    val outputGlTextureRotationDegrees: Int
        get() = orientation.outputGlTextureRotationDegrees

    val mirrorPreview: Boolean
        get() = orientation.mirrorPreview

    val mirrorStream: Boolean
        get() = orientation.mirrorStream

    /** Compatibility alias for stream-side callers; use explicit fields in rendering code. */
    val mirror: Boolean
        get() = mirrorStream

    /** Destination dimensions for this specific render viewport. */
    val outputResolution: Resolution
        get() = Resolution(targetWidth, targetHeight)

    val logicalAspectRatio: Double
        get() = logicalWidth.toDouble() / logicalHeight.toDouble()

    fun forTarget(width: Int, height: Int): VideoTransform =
        computeVideoTransform(
            orientation = orientation,
            sourceWidth = sourceWidth,
            sourceHeight = sourceHeight,
            targetWidth = width,
            targetHeight = height,
        )
}

/** FIT_CENTER geometry with a single pixel-space scale and no stretching. */
data class ViewportFit(
    val scaleX: Float,
    val scaleY: Float,
    val uniformScale: Float,
    val contentWidth: Float,
    val contentHeight: Float,
)

/** Compatibility name for existing callers; new code should use [ViewportFit]. */
typealias FitCenterScale = ViewportFit

fun computeVideoTransform(
    sensorOrientation: Int,
    displayRotation: Int,
    lensFacing: CameraLens,
    sourceWidth: Int,
    sourceHeight: Int,
    targetWidth: Int,
    targetHeight: Int,
    outputAspectRatio: VideoAspectRatio? = null,
): VideoTransform = computeVideoTransform(
    sensorOrientation = sensorOrientation,
    deviceOrientation = DeviceOrientation(displayRotationDegrees = displayRotation),
        lensFacing = lensFacing,
        sourceWidth = sourceWidth,
        sourceHeight = sourceHeight,
        targetWidth = targetWidth,
        targetHeight = targetHeight,
        outputAspectRatio = outputAspectRatio,
    )

fun computeVideoTransform(
    sensorOrientation: Int,
    deviceOrientation: DeviceOrientation,
    lensFacing: CameraLens,
    sourceWidth: Int,
    sourceHeight: Int,
    targetWidth: Int,
    targetHeight: Int,
    outputAspectRatio: VideoAspectRatio? = null,
): VideoTransform = computeVideoTransform(
    orientation = OrientationResolver.resolve(
        sensorOrientationDegrees = sensorOrientation,
        lensFacing = lensFacing,
        deviceOrientation = deviceOrientation,
        bufferResolution = Resolution(sourceWidth, sourceHeight),
        outputAspectRatio = outputAspectRatio,
    ),
    sourceWidth = sourceWidth,
    sourceHeight = sourceHeight,
    targetWidth = targetWidth,
    targetHeight = targetHeight,
)

/** Builds the encoder/SDP transform from the actual oriented video dimensions. */
fun computeVideoOutputTransform(
    sensorOrientation: Int,
    displayRotation: Int,
    lensFacing: CameraLens,
    sourceWidth: Int,
    sourceHeight: Int,
    outputAspectRatio: VideoAspectRatio? = null,
): VideoTransform = computeVideoOutputTransform(
    sensorOrientation = sensorOrientation,
    deviceOrientation = DeviceOrientation(displayRotationDegrees = displayRotation),
    lensFacing = lensFacing,
        sourceWidth = sourceWidth,
        sourceHeight = sourceHeight,
        outputAspectRatio = outputAspectRatio,
    )

fun computeVideoOutputTransform(
    sensorOrientation: Int,
    deviceOrientation: DeviceOrientation,
    lensFacing: CameraLens,
    sourceWidth: Int,
    sourceHeight: Int,
    outputAspectRatio: VideoAspectRatio? = null,
): VideoTransform {
    val orientation = OrientationResolver.resolve(
        sensorOrientationDegrees = sensorOrientation,
        lensFacing = lensFacing,
        deviceOrientation = deviceOrientation,
        bufferResolution = Resolution(sourceWidth, sourceHeight),
        outputAspectRatio = outputAspectRatio,
    )
    return computeVideoTransform(
        orientation = orientation,
        sourceWidth = sourceWidth,
        sourceHeight = sourceHeight,
        targetWidth = orientation.outputResolution.width,
        targetHeight = orientation.outputResolution.height,
    )
}

internal fun computeVideoTransform(
    orientation: CameraOrientationState,
    sourceWidth: Int,
    sourceHeight: Int,
    targetWidth: Int,
    targetHeight: Int,
): VideoTransform {
    require(sourceWidth > 0 && sourceHeight > 0) { "source size must be positive" }
    require(targetWidth > 0 && targetHeight > 0) { "target size must be positive" }
    val quarterTurn = orientation.outputRotationDegrees == 90 || orientation.outputRotationDegrees == 270
    val logicalWidth = if (quarterTurn) sourceHeight else sourceWidth
    val logicalHeight = if (quarterTurn) sourceWidth else sourceHeight
    val fit = fitCenterScale(logicalWidth, logicalHeight, targetWidth, targetHeight)
    return VideoTransform(
        orientation = orientation,
        sourceWidth = sourceWidth,
        sourceHeight = sourceHeight,
        targetWidth = targetWidth,
        targetHeight = targetHeight,
        logicalWidth = logicalWidth,
        logicalHeight = logicalHeight,
        sourceAspectRatio = logicalWidth.toDouble() / logicalHeight.toDouble(),
        destinationAspectRatio = targetWidth.toDouble() / targetHeight.toDouble(),
        scaleX = fit.scaleX,
        scaleY = fit.scaleY,
        uniformScale = fit.uniformScale,
    )
}

fun fitCenterScale(
    sourceWidth: Int,
    sourceHeight: Int,
    targetWidth: Int,
    targetHeight: Int,
): FitCenterScale {
    require(sourceWidth > 0 && sourceHeight > 0) { "source size must be positive" }
    require(targetWidth > 0 && targetHeight > 0) { "target size must be positive" }
    val uniformScale = minOf(
        targetWidth.toDouble() / sourceWidth.toDouble(),
        targetHeight.toDouble() / sourceHeight.toDouble(),
    )
    val contentWidth = sourceWidth.toDouble() * uniformScale
    val contentHeight = sourceHeight.toDouble() * uniformScale
    return FitCenterScale(
        scaleX = (contentWidth / targetWidth.toDouble()).toFloat(),
        scaleY = (contentHeight / targetHeight.toDouble()).toFloat(),
        uniformScale = uniformScale.toFloat(),
        contentWidth = contentWidth.toFloat(),
        contentHeight = contentHeight.toFloat(),
    )
}

internal fun normalizeDegrees(value: Int): Int = ((value % 360) + 360) % 360
