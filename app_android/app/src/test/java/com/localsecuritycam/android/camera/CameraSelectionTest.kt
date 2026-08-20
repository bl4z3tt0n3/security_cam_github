package com.localsecuritycam.android.camera

import com.localsecuritycam.android.settings.CameraLens
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class CameraSelectionTest {
    private val cameras = listOf("0", "1")
    private val lensByCamera = mapOf(
        "0" to CameraLens.BACK,
        "1" to CameraLens.FRONT,
    )

    @Test
    fun selectsBackCameraFromReportedFacing() {
        assertEquals("0", selectCameraId(cameras, lensByCamera::get, CameraLens.BACK))
    }

    @Test
    fun selectsFrontCameraFromReportedFacing() {
        assertEquals("1", selectCameraId(cameras, lensByCamera::get, CameraLens.FRONT))
    }

    @Test
    fun returnsNullWhenNoCameraMatchesTheRequestedLens() {
        assertNull(selectCameraId(listOf("0"), { CameraLens.FRONT }, CameraLens.BACK))
    }
}
