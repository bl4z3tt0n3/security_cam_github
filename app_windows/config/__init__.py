"""Windows monitor configuration helpers."""

from .camera_config import (
    CameraDraft,
    CameraEndpoint,
    ValidatedCameraDraft,
    build_stream_url,
    draft_from_slot,
    parse_camera_url,
    runtime_stream_url,
    validate_camera_draft,
)
from .credentials import (
    CredentialStore,
    CredentialStoreError,
    DpapiCredentialStore,
    InMemoryCredentialStore,
)
from .persistence import CameraConfigRepository, CameraSaveResult
from .ui_config import DEFAULT_SLOT_COUNT, UiSettings, choose_config_path

__all__ = [
    "DEFAULT_SLOT_COUNT",
    "UiSettings",
    "choose_config_path",
    "CameraConfigRepository",
    "CameraDraft",
    "CameraEndpoint",
    "CameraSaveResult",
    "CredentialStore",
    "CredentialStoreError",
    "DpapiCredentialStore",
    "InMemoryCredentialStore",
    "ValidatedCameraDraft",
    "build_stream_url",
    "draft_from_slot",
    "parse_camera_url",
    "runtime_stream_url",
    "validate_camera_draft",
]
