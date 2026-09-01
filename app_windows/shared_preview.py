"""Low-overhead local preview transport for the WPF frontend."""

from __future__ import annotations

import mmap
import os
import struct
from typing import Any

import numpy as np


FRAME_MAGIC = b"LSCF"
FRAME_VERSION = 1
FRAME_HEADER = struct.Struct("<4sIQQIIII")


class SharedFramePublisher:
    """Publish the latest contiguous BGR frame through a Windows named mmap.

    The fixed header uses an odd/even write epoch. Readers accept a frame only
    when the epoch is even and unchanged before/after their pixel copy.
    """

    def __init__(self, camera_id: str) -> None:
        if os.name != "nt":
            raise RuntimeError("WPF shared preview transport is available on Windows only")
        safe_id = "".join(ch if ch.isalnum() else "_" for ch in camera_id) or "camera"
        self._prefix = f"LocalSecurityCamPreview_{os.getpid()}_{safe_id}"
        self._generation = 0
        self._mapping: mmap.mmap | None = None
        self._mapping_name: str | None = None
        self._capacity = 0
        self._epoch = 0

    def close(self) -> None:
        mapping = self._mapping
        self._mapping = None
        self._mapping_name = None
        self._capacity = 0
        if mapping is not None:
            mapping.close()

    def _ensure_mapping(self, required_bytes: int) -> None:
        total = FRAME_HEADER.size + required_bytes
        if self._mapping is not None and self._capacity >= total:
            return
        old = self._mapping
        self._generation += 1
        self._mapping_name = f"{self._prefix}_{self._generation}"
        self._mapping = mmap.mmap(
            -1,
            total,
            tagname=self._mapping_name,
            access=mmap.ACCESS_WRITE,
        )
        self._capacity = total
        if old is not None:
            old.close()

    def publish(self, sequence: int, frame: Any) -> dict[str, Any]:
        image = np.asarray(frame)
        if (
            image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] <= 0
            or image.shape[1] <= 0
        ):
            raise ValueError("preview frame must be a non-empty HxWx3 BGR image")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if not image.flags.c_contiguous:
            image = np.ascontiguousarray(image)

        height, width = int(image.shape[0]), int(image.shape[1])
        stride = int(image.strides[0])
        byte_count = stride * height
        self._ensure_mapping(byte_count)
        mapping = self._mapping
        name = self._mapping_name
        assert mapping is not None and name is not None

        self._epoch += 1
        if self._epoch % 2 == 0:
            self._epoch += 1
        FRAME_HEADER.pack_into(
            mapping,
            0,
            FRAME_MAGIC,
            FRAME_VERSION,
            self._epoch,
            int(sequence),
            width,
            height,
            stride,
            byte_count,
        )
        mapping.seek(FRAME_HEADER.size)
        mapping.write(memoryview(image).cast("B"))
        self._epoch += 1
        FRAME_HEADER.pack_into(
            mapping,
            0,
            FRAME_MAGIC,
            FRAME_VERSION,
            self._epoch,
            int(sequence),
            width,
            height,
            stride,
            byte_count,
        )
        return {
            "frame_shm_name": name,
            "frame_byte_count": byte_count,
            "frame_stride": stride,
            "frame_width": width,
            "frame_height": height,
        }


__all__ = ["FRAME_HEADER", "FRAME_MAGIC", "FRAME_VERSION", "SharedFramePublisher"]
