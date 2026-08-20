package com.localsecuritycam.android.ui.components

import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.viewinterop.AndroidView
import com.localsecuritycam.android.R

/**
 * Compose host for the existing preview surface. Camera2/EGL ownership stays
 * in the foreground service; this composable only forwards Surface lifecycle.
 */
@Composable
internal fun CameraPreview(
    modifier: Modifier = Modifier,
    onSurfaceAvailable: (surface: Surface, width: Int, height: Int) -> Unit,
    onSurfaceDestroyed: () -> Unit,
) {
    AndroidView(
        modifier = modifier
            .testTag("camera_preview")
            .semantics { contentDescription = "Preview della camera" },
        factory = { context ->
            SurfaceView(context).apply {
                id = R.id.camera_preview
                contentDescription = context.getString(R.string.camera_preview_description)
                // Deliberately no background and no setZOrderOnTop: the Huawei
                // preview must remain the service-owned visible Surface.
                holder.addCallback(
                    object : SurfaceHolder.Callback {
                        override fun surfaceCreated(holder: SurfaceHolder) {
                            val frame = holder.surfaceFrame
                            onSurfaceAvailable(holder.surface, frame.width(), frame.height())
                        }

                        override fun surfaceChanged(
                            holder: SurfaceHolder,
                            format: Int,
                            width: Int,
                            height: Int,
                        ) {
                            onSurfaceAvailable(holder.surface, width, height)
                        }

                        override fun surfaceDestroyed(holder: SurfaceHolder) {
                            onSurfaceDestroyed()
                        }
                    },
                )
            }
        },
        update = { view ->
            if (view.id != R.id.camera_preview) view.id = R.id.camera_preview
        },
    )
}

