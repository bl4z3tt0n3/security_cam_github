package com.localsecuritycam.android.camera

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.params.StreamConfigurationMap
import android.media.MediaCodecInfo
import android.media.MediaCodecList
import android.util.Range
import android.util.Size
import com.localsecuritycam.android.settings.CameraLens
import com.localsecuritycam.android.settings.Resolution
import com.localsecuritycam.android.settings.StreamSettings

data class CameraCapabilities(
    val cameraId: String,
    val lens: CameraLens,
    val resolutions: List<Resolution>,
    val fpsValues: List<Int>,
    val fpsByResolution: Map<Resolution, List<Int>>,
    val minBitrate: Int,
    val maxBitrate: Int,
    val encoderValidated: Boolean = true,
)

class CameraCapabilitiesProvider(private val context: Context) {
    private val cameraManager = context.getSystemService(CameraManager::class.java)

    fun query(lens: CameraLens, requireEncoder: Boolean = true): CameraCapabilities {
        check(context.checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            "camera permission is not granted"
        }
        val cameraId = selectCameraId(
            cameraIds = cameraManager.cameraIdList.toList(),
            lensForCamera = { id ->
                when (cameraManager.getCameraCharacteristics(id).get(CameraCharacteristics.LENS_FACING)) {
                    CameraCharacteristics.LENS_FACING_BACK -> CameraLens.BACK
                    CameraCharacteristics.LENS_FACING_FRONT -> CameraLens.FRONT
                    else -> null
                }
            },
            requestedLens = lens,
        ) ?: throw IllegalStateException("no ${lens.name.lowercase()} camera found")
        val characteristics = cameraManager.getCameraCharacteristics(cameraId)
        val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            ?: throw IllegalStateException("camera has no stream configuration map")
        val ranges: Array<Range<Int>> =
            characteristics.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)
                ?: emptyArray()
        val fpsValues = supportedFps(ranges)
        check(fpsValues.isNotEmpty()) { "camera has no usable FPS capability" }
        val encoderCapabilities = if (requireEncoder) findAvcEncoderCapabilities() else null
        check(!requireEncoder || encoderCapabilities != null) { "no AVC encoder available" }
        val compatible = outputResolutions(map)
            .map { resolution ->
                val compatibleFps = fpsValues.filter { fps ->
                    (if (encoderCapabilities == null) {
                        true
                    } else {
                        supportsBothOutputGeometries(
                            resolution = resolution,
                            fps = fps,
                            encoderSupports = { width, height, rate ->
                                encoderCapabilities.areSizeAndRateSupported(width, height, rate)
                            },
                        )
                    }) && cameraSupportsFps(map, resolution, fps)
                }
                resolution to compatibleFps
            }
            .filter { (_, compatibleFps) -> compatibleFps.isNotEmpty() }
            .distinct()
            .sortedWith(compareBy<Pair<Resolution, List<Int>>> { it.first.width * it.first.height }.thenBy { it.first.width })
        check(compatible.isNotEmpty()) { "no compatible camera/AVC resolution and FPS combination" }
        val resolutions = compatible.map { it.first }
        val fpsByResolution = compatible.toMap()
        val bitrateRange = encoderCapabilities?.bitrateRange
        return CameraCapabilities(
            cameraId = cameraId,
            lens = lens,
            resolutions = resolutions,
            fpsValues = fpsByResolution.values.flatten().distinct().sorted(),
            fpsByResolution = fpsByResolution,
            minBitrate = bitrateRange?.lower ?: 0,
            maxBitrate = bitrateRange?.upper ?: Int.MAX_VALUE,
            encoderValidated = encoderCapabilities != null,
        )
    }

    companion object {
        fun isSupported(capabilities: CameraCapabilities, resolution: Resolution, fps: Int): Boolean =
            fps in (capabilities.fpsByResolution[resolution] ?: emptyList())

        fun validationErrors(capabilities: CameraCapabilities, settings: StreamSettings): List<String> = buildList {
            if (!isSupported(capabilities, settings.resolution, settings.fps)) {
                add(
                    "selected resolution/FPS is not supported by Camera2 and AVC " +
                        "for both landscape and portrait output geometries",
                )
            }
            if (capabilities.encoderValidated && settings.bitrate !in capabilities.minBitrate..capabilities.maxBitrate) {
                add(
                    "bitrate must be between ${capabilities.minBitrate} and " +
                        "${capabilities.maxBitrate} bps for the detected AVC encoder",
                )
            }
        }

        /**
         * A landscape Camera2 buffer can become portrait after the GL pixel
         * transform. AVC must support both dimensions at the selected FPS;
         * otherwise startup is rejected instead of emitting a false SDP.
         */
        internal fun supportsBothOutputGeometries(
            resolution: Resolution,
            fps: Int,
            encoderSupports: (width: Int, height: Int, fps: Double) -> Boolean,
        ): Boolean = encoderSupports(resolution.width, resolution.height, fps.toDouble()) &&
            encoderSupports(resolution.height, resolution.width, fps.toDouble())
    }

    private fun outputResolutions(map: StreamConfigurationMap): List<Resolution> =
        map.getOutputSizes(SurfaceTexture::class.java).orEmpty()
            .map { Resolution(it.width, it.height) }
            .distinct()
            .filter { it.width <= 3840 && it.height <= 2160 }

    private fun cameraSupportsFps(map: StreamConfigurationMap, resolution: Resolution, fps: Int): Boolean {
        val minFrameDurationNs = runCatching {
            map.getOutputMinFrameDuration(SurfaceTexture::class.java, Size(resolution.width, resolution.height))
        }.getOrDefault(0L)
        return minFrameDurationNs <= 0L || minFrameDurationNs <= 1_000_000_000L / fps
    }

    private fun supportedFps(ranges: Array<Range<Int>>): List<Int> {
        val preferred = listOf(15, 20, 24, 25, 30, 60)
        val valid = preferred.filter { fps -> ranges.any { fps in it.lower..it.upper } }
        return (valid.ifEmpty { ranges.flatMap { listOf(it.lower, it.upper) } })
            .filter { it in 1..60 }
            .distinct()
            .sorted()
    }

    private fun findAvcEncoderCapabilities(): MediaCodecInfo.VideoCapabilities? {
        val codecs = MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos
        return codecs.asSequence()
            .filter { info ->
                info.isEncoder && info.supportedTypes.any { it.equals("video/avc", ignoreCase = true) }
            }
            .mapNotNull { encoder ->
                val type = encoder.supportedTypes.first { it.equals("video/avc", ignoreCase = true) }
                val codecCapabilities = runCatching { encoder.getCapabilitiesForType(type) }.getOrNull() ?: return@mapNotNull null
                if (MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface !in codecCapabilities.colorFormats) return@mapNotNull null
                codecCapabilities.videoCapabilities
            }
            .firstOrNull()
    }

}
