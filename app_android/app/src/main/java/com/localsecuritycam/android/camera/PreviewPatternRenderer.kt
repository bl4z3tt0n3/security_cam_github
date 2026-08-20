package com.localsecuritycam.android.camera

import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES20
import android.os.Handler
import android.os.HandlerThread
import android.view.Surface
import com.localsecuritycam.android.diagnostics.StreamErrorLogger
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Debug-only EGL probe for the real Activity SurfaceView. It intentionally
 * does not open Camera2, so a failed pattern isolates the EGL/SurfaceView path.
 */
class PreviewPatternRenderer {
    private var thread: HandlerThread? = null
    private var handler: Handler? = null
    private var display: EGLDisplay = EGL14.EGL_NO_DISPLAY
    private var context: EGLContext = EGL14.EGL_NO_CONTEXT
    private var surface: EGLSurface = EGL14.EGL_NO_SURFACE
    private var program = 0
    private var positionHandle = -1

    fun start(target: Surface, width: Int, height: Int) {
        require(target.isValid) { "pattern target Surface is invalid" }
        stop()
        val worker = HandlerThread("preview-pattern-renderer")
        thread = worker
        worker.start()
        handler = Handler(worker.looper)
        val ready = CountDownLatch(1)
        var failure: Throwable? = null
        check(handler!!.post {
            try {
                initialize(target, width, height)
            } catch (error: Throwable) {
                failure = error
                StreamErrorLogger.info(
                    "PREVIEW_PATTERN_ERROR detail=${error.message ?: error.javaClass.simpleName}",
                )
            } finally {
                ready.countDown()
            }
        }) { "pattern renderer callback rejected" }
        check(ready.await(3, TimeUnit.SECONDS)) { "pattern renderer timed out" }
        failure?.let { throw IllegalStateException("pattern renderer failed", it) }
    }

    fun stop() {
        val currentHandler = handler
        val currentThread = thread
        if (currentHandler != null) {
            val released = CountDownLatch(1)
            if (!currentHandler.post {
                    release()
                    released.countDown()
                }
            ) {
                release()
                released.countDown()
            }
            released.await(2, TimeUnit.SECONDS)
        } else {
            release()
        }
        currentThread?.quitSafely()
        if (currentThread != null && Thread.currentThread() !== currentThread) {
            currentThread.join(1_000)
        }
        handler = null
        thread = null
    }

    private fun initialize(target: Surface, width: Int, height: Int) {
        display = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        check(display != EGL14.EGL_NO_DISPLAY) { "pattern EGL display unavailable" }
        val version = IntArray(2)
        check(EGL14.eglInitialize(display, version, 0, version, 1)) {
            "pattern EGL initialization failed"
        }
        val attributes = intArrayOf(
            EGL14.EGL_RED_SIZE, 8,
            EGL14.EGL_GREEN_SIZE, 8,
            EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_ALPHA_SIZE, 8,
            EGL14.EGL_SURFACE_TYPE, EGL14.EGL_WINDOW_BIT,
            EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
            EGL_RECORDABLE_ANDROID, 1,
            EGL14.EGL_NONE,
        )
        val configs = arrayOfNulls<EGLConfig>(1)
        val count = IntArray(1)
        check(EGL14.eglChooseConfig(display, attributes, 0, configs, 0, 1, count, 0) && count[0] > 0) {
            "pattern EGL configuration unavailable"
        }
        val config = configs[0] ?: error("pattern EGL config missing")
        context = EGL14.eglCreateContext(
            display,
            config,
            EGL14.EGL_NO_CONTEXT,
            intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE),
            0,
        )
        check(context != EGL14.EGL_NO_CONTEXT) { "pattern EGL context unavailable" }
        logEglError("pattern eglCreateContext")
        surface = EGL14.eglCreateWindowSurface(
            display,
            config,
            target,
            intArrayOf(EGL14.EGL_NONE),
            0,
        )
        check(surface != EGL14.EGL_NO_SURFACE) {
            "pattern EGL window surface unavailable error=0x${EGL14.eglGetError().toString(16)}"
        }
        logEglError("pattern eglCreateWindowSurface")
        check(EGL14.eglMakeCurrent(display, surface, surface, context)) {
            "pattern EGL makeCurrent failed error=0x${EGL14.eglGetError().toString(16)}"
        }
        logEglError("pattern eglMakeCurrent")
        program = createProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        positionHandle = GLES20.glGetAttribLocation(program, "aPosition")
        check(positionHandle >= 0) { "pattern position attribute unavailable" }
        checkGlErrors("pattern program handles")
        val vertices = floatBuffer(floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f))
        GLES20.glViewport(0, 0, width.coerceAtLeast(1), height.coerceAtLeast(1))
        checkGlErrors("pattern viewport")
        GLES20.glDisable(GLES20.GL_BLEND)
        GLES20.glDisable(GLES20.GL_SCISSOR_TEST)
        checkGlErrors("pattern state")
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
        checkGlErrors("pattern clear")
        GLES20.glUseProgram(program)
        GLES20.glEnableVertexAttribArray(positionHandle)
        GLES20.glVertexAttribPointer(positionHandle, 2, GLES20.GL_FLOAT, false, 0, vertices)
        checkGlErrors("pattern attributes")
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        checkGlErrors("pattern draw")
        GLES20.glDisableVertexAttribArray(positionHandle)
        checkGlErrors("pattern draw cleanup")
        check(EGL14.eglSwapBuffers(display, surface)) {
            "pattern eglSwapBuffers failed error=0x${EGL14.eglGetError().toString(16)}"
        }
        logEglError("pattern eglSwapBuffers")
        StreamErrorLogger.info(
            "PREVIEW_PATTERN_RENDERED width=${width.coerceAtLeast(1)} " +
                "height=${height.coerceAtLeast(1)} egl=${version[0]}.${version[1]}",
        )
    }

    private fun release() {
        if (display == EGL14.EGL_NO_DISPLAY) return
        if (context != EGL14.EGL_NO_CONTEXT && surface != EGL14.EGL_NO_SURFACE) {
            EGL14.eglMakeCurrent(display, surface, surface, context)
            if (program != 0) GLES20.glDeleteProgram(program)
        }
        program = 0
        positionHandle = -1
        if (surface != EGL14.EGL_NO_SURFACE) EGL14.eglDestroySurface(display, surface)
        if (context != EGL14.EGL_NO_CONTEXT) EGL14.eglDestroyContext(display, context)
        EGL14.eglReleaseThread()
        EGL14.eglTerminate(display)
        display = EGL14.EGL_NO_DISPLAY
        context = EGL14.EGL_NO_CONTEXT
        surface = EGL14.EGL_NO_SURFACE
    }

    private fun createProgram(vertexSource: String, fragmentSource: String): Int {
        val vertex = compileShader(GLES20.GL_VERTEX_SHADER, vertexSource)
        val fragment = compileShader(GLES20.GL_FRAGMENT_SHADER, fragmentSource)
        val program = GLES20.glCreateProgram()
        GLES20.glAttachShader(program, vertex)
        GLES20.glAttachShader(program, fragment)
        GLES20.glLinkProgram(program)
        val status = IntArray(1)
        GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, status, 0)
        GLES20.glDeleteShader(vertex)
        GLES20.glDeleteShader(fragment)
        check(status[0] != 0) { GLES20.glGetProgramInfoLog(program) }
        return program
    }

    private fun compileShader(type: Int, source: String): Int {
        val shader = GLES20.glCreateShader(type)
        checkGlErrors("pattern glCreateShader")
        GLES20.glShaderSource(shader, source)
        checkGlErrors("pattern glShaderSource")
        GLES20.glCompileShader(shader)
        checkGlErrors("pattern glCompileShader")
        val status = IntArray(1)
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, status, 0)
        check(status[0] != 0) {
            val message = GLES20.glGetShaderInfoLog(shader)
            GLES20.glDeleteShader(shader)
            message
        }
        return shader
    }

    private fun checkGlErrors(stage: String) {
        var error = GLES20.glGetError()
        while (error != GLES20.GL_NO_ERROR) {
            StreamErrorLogger.info("PREVIEW_PATTERN_GLES_ERROR stage=$stage code=0x${error.toString(16)}")
            error = GLES20.glGetError()
        }
    }

    private fun logEglError(stage: String) {
        val error = EGL14.eglGetError()
        if (error != EGL14.EGL_SUCCESS) {
            StreamErrorLogger.info("PREVIEW_PATTERN_EGL_ERROR stage=$stage code=0x${error.toString(16)}")
        }
    }

    private fun floatBuffer(values: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(values.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .apply {
                put(values)
                position(0)
            }

    private companion object {
        const val EGL_RECORDABLE_ANDROID = 0x3142
        const val VERTEX_SHADER = """
            attribute vec4 aPosition;
            varying vec2 vUv;
            void main() {
                gl_Position = aPosition;
                vUv = (aPosition.xy + 1.0) * 0.5;
            }
        """
        const val FRAGMENT_SHADER = """
            precision mediump float;
            varying vec2 vUv;
            void main() {
                vec3 color;
                if (vUv.x < 0.5 && vUv.y < 0.5) color = vec3(0.9, 0.05, 0.05);
                else if (vUv.x >= 0.5 && vUv.y < 0.5) color = vec3(0.05, 0.85, 0.15);
                else if (vUv.x < 0.5 && vUv.y >= 0.5) color = vec3(0.05, 0.2, 0.95);
                else color = vec3(0.95, 0.75, 0.05);
                float crossX = step(abs(vUv.x - 0.5), 0.018);
                float crossY = step(abs(vUv.y - 0.5), 0.018);
                if (crossX + crossY > 0.0) color = vec3(1.0);
                gl_FragColor = vec4(color, 1.0);
            }
        """
    }
}
