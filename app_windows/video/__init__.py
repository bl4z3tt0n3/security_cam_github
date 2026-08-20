"""Frame providers used by the Windows monitor."""

from .frame_provider import (
    BackendFrameProvider,
    FrameProvider,
    ProviderSnapshot,
)

__all__ = ["BackendFrameProvider", "FrameProvider", "ProviderSnapshot"]
