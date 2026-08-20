package com.localsecuritycam.android.camera

import com.localsecuritycam.android.settings.CameraLens
import com.localsecuritycam.android.settings.VideoAspectRatio
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class VideoTransformTest {
    @Test
    fun physicalQuadrantsMapToTheRequiredSurfaceRotations() {
        assertEquals(
            listOf(0, 270, 180, 90),
            listOf(0, 90, 180, 270).map(OrientationResolver::targetSurfaceRotationForPhysical),
        )
    }

    @Test
    fun sensor90BackCameraProducesTheFourRequiredH264Geometries() {
        val transforms = listOf(0, 90, 180, 270).map { physical ->
            computeVideoOutputTransform(
                sensorOrientation = 90,
                deviceOrientation = DeviceOrientation(physicalOrientationDegrees = physical, displayRotationDegrees = 180),
                lensFacing = CameraLens.BACK,
                sourceWidth = 1280,
                sourceHeight = 720,
            )
        }

        assertEquals(listOf(90, 0, 270, 180), transforms.map { it.rotationDegrees })
        assertEquals(
            listOf("720x1280", "1280x720", "720x1280", "1280x720"),
            transforms.map { it.outputResolution.toString() },
        )
        assertTrue(transforms.all { !it.mirrorPreview && !it.mirrorStream })
        assertTrue(transforms.all { it.orientation.source == OrientationSource.PHYSICAL_SENSOR })
    }

    @Test
    fun sensor270KeepsPortraitAndLandscapeOutputClassesCorrect() {
        val transforms = listOf(0, 90, 180, 270).map { physical ->
            computeVideoOutputTransform(
                sensorOrientation = 270,
                deviceOrientation = DeviceOrientation(physicalOrientationDegrees = physical),
                lensFacing = CameraLens.BACK,
                sourceWidth = 1280,
                sourceHeight = 720,
            )
        }

        assertEquals(listOf(270, 180, 90, 0), transforms.map { it.rotationDegrees })
        assertEquals(
            listOf("720x1280", "1280x720", "720x1280", "1280x720"),
            transforms.map { it.outputResolution.toString() },
        )
    }

    @Test
    fun displayIsOnlyTheFallbackWhenPhysicalOrientationIsUnavailable() {
        val transform = computeVideoOutputTransform(
            sensorOrientation = 90,
            deviceOrientation = DeviceOrientation(displayRotationDegrees = 90),
            lensFacing = CameraLens.BACK,
            sourceWidth = 1280,
            sourceHeight = 720,
        )

        assertEquals(180, transform.rotationDegrees)
        assertEquals(OrientationSource.DISPLAY_FALLBACK, transform.orientation.source)
        assertEquals("1280x720", transform.outputResolution.toString())
    }

    @Test
    fun frontPolicyKeepsPreviewAndStreamMirrorFlagsSeparateAndFalse() {
        val transform = computeVideoOutputTransform(
            sensorOrientation = 90,
            deviceOrientation = DeviceOrientation(physicalOrientationDegrees = 90),
            lensFacing = CameraLens.FRONT,
            sourceWidth = 1280,
            sourceHeight = 720,
        )

        assertEquals(180, transform.rotationDegrees)
        assertEquals(180, transform.pixelClockwiseRotationDegrees)
        assertEquals(180, transform.glTextureRotationDegrees)
        assertFalse(transform.mirrorPreview)
        assertFalse(transform.mirrorStream)
    }

    @Test
    fun physicalStabilizerAppliesFirstThenRequiresThresholdAndDebounce() {
        val stabilizer = PhysicalOrientationStabilizer()

        assertEquals(0, stabilizer.update(4, 0))
        assertNull(stabilizer.update(45, 100))
        assertNull(stabilizer.update(70, 200))
        assertNull(stabilizer.update(80, 549))
        assertEquals(90, stabilizer.update(82, 550))
        assertEquals(90, stabilizer.latestStableOrientationDegrees)
    }

    @Test
    fun unknownSamplesKeepTheLatestStablePhysicalOrientation() {
        val stabilizer = PhysicalOrientationStabilizer()
        assertEquals(180, stabilizer.update(181, 0))
        assertNull(stabilizer.update(-1, 100))
        assertEquals(180, stabilizer.latestStableOrientationDegrees)
    }

    @Test
    fun cameraRelativeRotationConvertsToClockwisePixelsAndGlDirection() {
        assertEquals(90, OrientationResolver.pixelClockwiseRotationDegrees(90, CameraLens.BACK))
        assertEquals(270, OrientationResolver.pixelClockwiseRotationDegrees(270, CameraLens.BACK))
        assertEquals(270, OrientationResolver.pixelClockwiseRotationDegrees(90, CameraLens.FRONT))
        assertEquals(90, OrientationResolver.pixelClockwiseRotationDegrees(270, CameraLens.FRONT))

        assertEquals(270, textureCoordinateRotationDegrees(90, CameraLens.BACK))
        assertEquals(90, textureCoordinateRotationDegrees(270, CameraLens.BACK))
        assertEquals(90, textureCoordinateRotationDegrees(90, CameraLens.FRONT))
        assertEquals(270, textureCoordinateRotationDegrees(270, CameraLens.FRONT))
        assertEquals(0, textureCoordinateRotationDegrees(360, CameraLens.BACK))

        assertEquals(0, textureCoordinateRotationDegrees(90, CameraLens.BACK, 90))
        assertEquals(270, textureCoordinateRotationDegrees(90, CameraLens.BACK, 0))
    }

    @Test
    fun surfaceTextureRotationIsReadFromTheTransformedUAxis() {
        val standardVerticalFlip = floatArrayOf(
            1f, 0f, 0f, 0f,
            0f, -1f, 0f, 0f,
            0f, 0f, 1f, 0f,
            0f, 1f, 0f, 1f,
        )
        val huaweiProducerRotation = floatArrayOf(
            0f, -1f, 0f, 0f,
            -1f, 0f, 0f, 0f,
            0f, 0f, 1f, 0f,
            1f, 1f, 0f, 1f,
        )

        assertEquals(0, surfaceTexturePixelClockwiseRotationDegrees(standardVerticalFlip))
        assertEquals(90, surfaceTexturePixelClockwiseRotationDegrees(huaweiProducerRotation))
    }

    @Test
    fun allSensorLensAndDisplayQuadrantsKeepOneCanonicalOrientationContract() {
        for (sensor in listOf(90, 270)) {
            for (lens in CameraLens.values()) {
                for (display in listOf(0, 90, 180, 270)) {
                    val transform = computeVideoOutputTransform(
                        sensorOrientation = sensor,
                        deviceOrientation = DeviceOrientation(displayRotationDegrees = display),
                        lensFacing = lens,
                        sourceWidth = 1280,
                        sourceHeight = 720,
                    )
                    val cameraRelative = OrientationResolver.pixelRotationDegrees(sensor, display, lens)
                    val expectedResolution = if (cameraRelative == 90 || cameraRelative == 270) {
                        "720x1280"
                    } else {
                        "1280x720"
                    }

                    assertEquals(cameraRelative, transform.rotationDegrees)
                    assertEquals(
                        OrientationResolver.pixelClockwiseRotationDegrees(cameraRelative, lens),
                        transform.pixelClockwiseRotationDegrees,
                    )
                    assertEquals(
                        OrientationResolver.glTextureRotationDegrees(transform.pixelClockwiseRotationDegrees),
                        transform.glTextureRotationDegrees,
                    )
                    assertEquals(expectedResolution, transform.outputResolution.toString())
                    assertFalse(transform.mirrorPreview)
                    assertFalse(transform.mirrorStream)
                }
            }
        }
    }

    @Test
    fun fitCenterPreservesAspectWithoutNonUniformScale() {
        val portrait = fitCenterScale(720, 1280, 1920, 1080)
        val landscape = fitCenterScale(1280, 720, 700, 500)

        assertEquals(0.31640625f, portrait.scaleX, 0.0001f)
        assertEquals(1f, portrait.scaleY, 0.0001f)
        assertEquals(1f, landscape.scaleX, 0.0001f)
        assertEquals(0.7875f, landscape.scaleY, 0.0001f)
        assertEquals(720.0 / 1280.0, (portrait.contentWidth / portrait.contentHeight).toDouble(), 0.0001)
        assertEquals(1280.0 / 720.0, (landscape.contentWidth / landscape.contentHeight).toDouble(), 0.0001)
    }

    @Test
    fun previewViewportChangesOnlyFitCenterAndNotOutputGeometry() {
        val output = computeVideoOutputTransform(
            sensorOrientation = 90,
            deviceOrientation = DeviceOrientation(physicalOrientationDegrees = 0),
            lensFacing = CameraLens.BACK,
            sourceWidth = 1280,
            sourceHeight = 720,
        )
        val preview = output.forTarget(1000, 600)

        assertEquals("720x1280", output.outputResolution.toString())
        assertEquals(90, preview.rotationDegrees)
        assertEquals(output.orientation, preview.orientation)
        assertTrue(preview.scaleX > 0f && preview.scaleY > 0f)
    }

    @Test
    fun canonicalGeometryKeepsEncodedDimensionsIndependentFromPreviewViewport() {
        val output = computeVideoOutputTransform(
            sensorOrientation = 90,
            deviceOrientation = DeviceOrientation(physicalOrientationDegrees = 0),
            lensFacing = CameraLens.BACK,
            sourceWidth = 1280,
            sourceHeight = 720,
        )
        val preview = output.forTarget(500, 700)

        assertEquals("1280x720", output.geometry.sourceResolution.toString())
        assertEquals("720x1280", output.geometry.encodedResolution.toString())
        assertEquals(90, output.geometry.pixelRotationDegrees)
        assertEquals(720.0 / 1280.0, output.geometry.encodedAspectRatio, 0.0001)
        assertEquals(output.geometry, preview.geometry)
        assertTrue(preview.viewportFit.uniformScale > 0f)
    }

    @Test
    fun previewAndEncoderShareTheSameOrientationAndMirrorContract() {
        val encoder = computeVideoOutputTransform(
            sensorOrientation = 90,
            deviceOrientation = DeviceOrientation(physicalOrientationDegrees = 0),
            lensFacing = CameraLens.BACK,
            sourceWidth = 1280,
            sourceHeight = 720,
        )
        val preview = encoder.forTarget(1080, 2340)

        assertEquals(encoder.orientation, preview.orientation)
        assertEquals(encoder.rotationDegrees, preview.rotationDegrees)
        assertEquals(encoder.pixelClockwiseRotationDegrees, preview.pixelClockwiseRotationDegrees)
        assertEquals(encoder.glTextureRotationDegrees, preview.glTextureRotationDegrees)
        assertEquals(encoder.mirrorPreview, preview.mirrorPreview)
        assertEquals(encoder.mirrorStream, preview.mirrorStream)
        assertEquals(encoder.geometry, preview.geometry)
    }

    @Test
    fun selectedAspectRatioChangesOnlyTheFinalOutputOrientation() {
        val landscapeOutput = computeVideoOutputTransform(
            sensorOrientation = 90,
            deviceOrientation = DeviceOrientation(physicalOrientationDegrees = 0),
            lensFacing = CameraLens.BACK,
            sourceWidth = 1280,
            sourceHeight = 720,
            outputAspectRatio = VideoAspectRatio.LANDSCAPE_16_9,
        )
        val portraitOutput = computeVideoOutputTransform(
            sensorOrientation = 90,
            deviceOrientation = DeviceOrientation(physicalOrientationDegrees = 0),
            lensFacing = CameraLens.BACK,
            sourceWidth = 1280,
            sourceHeight = 720,
            outputAspectRatio = VideoAspectRatio.PORTRAIT_9_16,
        )

        assertEquals(90, landscapeOutput.rotationDegrees)
        assertEquals(90, portraitOutput.rotationDegrees)
        assertEquals(0, landscapeOutput.outputRotationDegrees)
        assertEquals(90, portraitOutput.outputRotationDegrees)
        assertEquals("1280x720", landscapeOutput.outputResolution.toString())
        assertEquals("720x1280", portraitOutput.outputResolution.toString())
        assertEquals(VideoAspectRatio.LANDSCAPE_16_9, landscapeOutput.orientation.outputAspectRatio)
        assertEquals(VideoAspectRatio.PORTRAIT_9_16, portraitOutput.orientation.outputAspectRatio)
        assertEquals(90, landscapeOutput.pixelClockwiseRotationDegrees)
        assertEquals(0, landscapeOutput.outputPixelClockwiseRotationDegrees)
        assertEquals(90, portraitOutput.outputPixelClockwiseRotationDegrees)
    }

    @Test
    fun aspectCorrectionChangesParityWithoutStretchingOrMirroring() {
        val portrait = OrientationResolver.outputRotationForAspect(
            cameraPixelRotationDegrees = 0,
            aspectRatio = VideoAspectRatio.PORTRAIT_9_16,
        )
        val landscape = OrientationResolver.outputRotationForAspect(
            cameraPixelRotationDegrees = 90,
            aspectRatio = VideoAspectRatio.LANDSCAPE_16_9,
        )

        assertEquals(90, portrait)
        assertEquals(0, landscape)
    }

    @Test
    fun officialCamera2FormulaUsesOppositeSurfaceSignsForBackAndFront() {
        assertEquals(0, OrientationResolver.pixelRotationDegrees(90, 270, CameraLens.BACK))
        assertEquals(180, OrientationResolver.pixelRotationDegrees(90, 270, CameraLens.FRONT))
    }

    @Test
    fun physicalOrientationWinsEvenWhenDisplayReportsAnotherQuadrant() {
        val first = computeVideoOutputTransform(
            90,
            DeviceOrientation(physicalOrientationDegrees = 90, displayRotationDegrees = 0),
            CameraLens.BACK,
            1280,
            720,
        )
        val second = computeVideoOutputTransform(
            90,
            DeviceOrientation(physicalOrientationDegrees = 90, displayRotationDegrees = 270),
            CameraLens.BACK,
            1280,
            720,
        )

        assertEquals(first.rotationDegrees, second.rotationDegrees)
        assertEquals(first.outputResolution, second.outputResolution)
        assertEquals(0, second.rotationDegrees)
    }

    @Test
    fun hysteresisRequiresMoreThanFiftyFiveDegreesBeforeChangingQuadrant() {
        val stabilizer = PhysicalOrientationStabilizer()
        assertEquals(0, stabilizer.update(0, 0))
        assertNull(stabilizer.update(55, 100))
        assertNull(stabilizer.update(56, 200))
        assertEquals(90, stabilizer.update(80, 550))
    }

    @Test
    fun resetMakesTheNextValidSampleImmediateAgain() {
        val stabilizer = PhysicalOrientationStabilizer()
        assertEquals(0, stabilizer.update(0, 0))
        stabilizer.reset()

        assertEquals(270, stabilizer.update(271, 1))
    }

    @Test
    fun orientedEncoderViewportNeedsNoLetterboxWhenItMatchesTheOutput() {
        val transform = computeVideoOutputTransform(
            sensorOrientation = 90,
            deviceOrientation = DeviceOrientation(physicalOrientationDegrees = 0),
            lensFacing = CameraLens.BACK,
            sourceWidth = 1280,
            sourceHeight = 720,
        )

        assertEquals(1f, transform.scaleX, 0.0001f)
        assertEquals(1f, transform.scaleY, 0.0001f)
        assertEquals(transform.logicalWidth, transform.targetWidth)
        assertEquals(transform.logicalHeight, transform.targetHeight)
    }
}
