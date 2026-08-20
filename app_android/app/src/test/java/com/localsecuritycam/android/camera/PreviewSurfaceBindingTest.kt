package com.localsecuritycam.android.camera

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PreviewSurfaceBindingTest {
    @Test
    fun previewReceivedBeforeRendererReadyIsAppliedWithTheLatestValue() {
        val binding = PreviewSurfaceBinding<String>()

        assertEquals(PreviewSurfaceUpdateKind.DEFERRED, binding.request("before").kind)
        assertEquals(PreviewSurfaceUpdateKind.DEFERRED, binding.request("during").kind)

        val ready = binding.markReady()
        assertEquals(PreviewSurfaceUpdateKind.APPLY, ready.kind)
        assertEquals("during", ready.surface)
    }

    @Test
    fun previewAttachAndDetachAfterRendererReadyDoNotAffectCameraSession() {
        val binding = PreviewSurfaceBinding<String>()
        binding.request("initial")
        assertEquals(PreviewSurfaceUpdateKind.APPLY, binding.markReady().kind)

        val attach = binding.request("replacement")
        assertEquals(PreviewSurfaceUpdateKind.APPLY, attach.kind)
        assertEquals("replacement", attach.surface)

        val detach = binding.request(null)
        assertEquals(PreviewSurfaceUpdateKind.APPLY, detach.kind)
        assertEquals(null, detach.surface)
    }

    @Test
    fun latePreviewCallbackIsIgnoredAfterStop() {
        val binding = PreviewSurfaceBinding<String>()
        binding.request("initial")
        binding.markReady()
        binding.stop()

        val update = binding.request("late")

        assertEquals(PreviewSurfaceUpdateKind.IGNORED, update.kind)
    }

    @Test
    fun bindingCanBeStartedAgainAfterStopWithAFreshGeneration() {
        val binding = PreviewSurfaceBinding<String>()
        binding.request("old")
        binding.markReady()
        binding.stop()
        binding.begin()

        assertEquals(PreviewSurfaceUpdateKind.DEFERRED, binding.request("new").kind)
        val ready = binding.markReady()
        assertEquals(PreviewSurfaceUpdateKind.APPLY, ready.kind)
        assertEquals("new", ready.surface)
    }

    @Test
    fun obsoleteSurfaceCallbacksCannotApplyAfterAReplacementOrNewActivityGeneration() {
        val binding = PreviewSurfaceBinding<String>()
        binding.begin()
        binding.request("first")
        binding.markReady()
        val oldCallback = binding.request("second")
        val currentCallback = binding.request("third")

        assertFalse(binding.isCurrent(oldCallback))
        assertTrue(binding.isCurrent(currentCallback))

        binding.begin()
        assertFalse(binding.isCurrent(currentCallback))
    }
}
