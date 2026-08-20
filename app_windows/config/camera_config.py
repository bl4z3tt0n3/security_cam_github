"""Pure camera-editor models, URL parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from app.config import (
    SUPPORTED_STREAM_SCHEMES,
    CameraConfig,
    ConfigurationError,
    is_placeholder_url,
    validate_stream_url,
)

from app_windows.models.camera_view_state import CameraSlot

from .credentials import CredentialStore


SUPPORTED_TRANSPORTS: Final[frozenset[str]] = frozenset({"auto", "tcp", "udp"})
DEFAULT_SCHEME = "rtsp"
DEFAULT_TRANSPORT = "tcp"


@dataclass(frozen=True)
class CameraEndpoint:
    """Parsed endpoint without exposing a password through its representation."""

    scheme: str
    host: str
    port: int | None
    path: str
    username: str
    query: str = ""
    fragment: str = ""
    _password: str | None = field(default=None, repr=False, compare=False)

    @property
    def has_password(self) -> bool:
        return bool(self._password)


@dataclass(frozen=True)
class CameraDraft:
    """One editable camera value set; secret fields never appear in ``repr``."""

    camera_id: str
    slot_index: int
    name: str
    enabled: bool
    scheme: str = DEFAULT_SCHEME
    host: str = ""
    port: int | None = None
    path: str = ""
    username: str = ""
    transport: str = DEFAULT_TRANSPORT
    query: str = ""
    fragment: str = ""
    password: str = field(default="", repr=False, compare=False)
    clear_password: bool = field(default=False, repr=False, compare=False)
    existing_password: str | None = field(default=None, repr=False, compare=False)

    @property
    def password_is_stored(self) -> bool:
        return bool(self.existing_password)


@dataclass(frozen=True)
class ValidatedCameraDraft:
    draft: CameraDraft
    stream_url: str | None
    credential_update_required: bool
    credential_value: str | None = field(default=None, repr=False, compare=False)

    def to_slot(self) -> CameraSlot:
        name = self.draft.name.strip() or f"Camera {self.draft.slot_index}"
        return CameraSlot(
            slot_index=self.draft.slot_index,
            camera_id=self.draft.camera_id.strip(),
            name=name,
            enabled=self.draft.enabled,
            configured=self.stream_url is not None,
            stream_url=self.stream_url,
            rtsp_transport=self.draft.transport,
        )


def parse_camera_url(url: str | None) -> CameraEndpoint | None:
    """Parse one configured URL while keeping credentials out of normal output."""

    if is_placeholder_url(url):
        return None
    assert url is not None
    normalized = validate_stream_url(url)
    try:
        parsed = urlsplit(normalized)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ConfigurationError("stream URL has an invalid host or port") from exc

    if not hostname:
        raise ConfigurationError("stream URL has no host name or IP address")
    return CameraEndpoint(
        scheme=parsed.scheme.lower(),
        host=hostname,
        port=port,
        path=parsed.path,
        username=unquote(parsed.username or ""),
        query=parsed.query,
        fragment=parsed.fragment,
        _password=unquote(parsed.password) if parsed.password is not None else None,
    )


def draft_from_slot(slot: CameraSlot, credentials: CredentialStore) -> CameraDraft:
    endpoint = parse_camera_url(slot.stream_url)
    stored_password = credentials.get(slot.camera_id)
    existing_password = stored_password or (endpoint._password if endpoint else None)
    if endpoint is None:
        return CameraDraft(
            camera_id=slot.camera_id,
            slot_index=slot.slot_index,
            name=slot.name,
            enabled=slot.enabled,
            transport=slot.rtsp_transport or DEFAULT_TRANSPORT,
            existing_password=existing_password,
        )
    return CameraDraft(
        camera_id=slot.camera_id,
        slot_index=slot.slot_index,
        name=slot.name,
        enabled=slot.enabled,
        scheme=endpoint.scheme,
        host=endpoint.host,
        port=endpoint.port,
        path=endpoint.path,
        username=endpoint.username,
        transport=slot.rtsp_transport or DEFAULT_TRANSPORT,
        query=endpoint.query,
        fragment=endpoint.fragment,
        existing_password=existing_password,
    )


def validate_camera_draft(draft: CameraDraft) -> ValidatedCameraDraft:
    """Validate editor fields and return a canonical URL without a password."""

    camera_id = draft.camera_id.strip()
    if not camera_id:
        raise ConfigurationError("camera id cannot be empty")

    scheme = draft.scheme.strip().lower()
    if scheme not in SUPPORTED_STREAM_SCHEMES:
        supported = ", ".join(sorted(SUPPORTED_STREAM_SCHEMES))
        raise ConfigurationError(f"schema URL non supportato; usare: {supported}")

    transport = draft.transport.strip().lower() or DEFAULT_TRANSPORT
    if transport not in SUPPORTED_TRANSPORTS:
        raise ConfigurationError("trasporto RTSP non valido")

    host = draft.host.strip()
    path = draft.path.strip()
    username = draft.username.strip()
    endpoint_fields_present = any(
        (host, path, username, draft.port is not None, draft.query, draft.fragment)
    )

    if not host:
        if draft.enabled or endpoint_fields_present:
            raise ConfigurationError("host/IP della camera obbligatorio")
        return ValidatedCameraDraft(
            draft=CameraDraft(
                **{
                    **draft.__dict__,
                    "camera_id": camera_id,
                    "transport": transport,
                }
            ),
            stream_url=None,
            credential_update_required=draft.clear_password,
            credential_value=None,
        )

    if any(character.isspace() for character in host) or any(
        character in host for character in "/?#@"
    ):
        raise ConfigurationError("host/IP non valido")

    if draft.port is not None and not 1 <= draft.port <= 65535:
        raise ConfigurationError("la porta deve essere compresa tra 1 e 65535")

    if scheme in {"rtsp", "rtsps"} and not path:
        raise ConfigurationError("il path RTSP è obbligatorio")
    if not path:
        path = "/"
    elif not path.startswith("/"):
        path = f"/{path}"
    if any(character in path for character in "\r\n"):
        raise ConfigurationError("path stream non valido")

    if username and any(character.isspace() or character in "/?#@" for character in username):
        raise ConfigurationError("username non valido")

    password = draft.password if draft.password != "" else draft.existing_password
    if draft.clear_password:
        password = None
    if password and not username:
        raise ConfigurationError("username obbligatorio quando è presente una password")

    normalized_draft = CameraDraft(
        camera_id=camera_id,
        slot_index=draft.slot_index,
        name=draft.name.strip(),
        enabled=draft.enabled,
        scheme=scheme,
        host=host,
        port=draft.port,
        path=path,
        username=username,
        transport=transport,
        query=draft.query.strip(),
        fragment=draft.fragment.strip(),
        password="",
        clear_password=draft.clear_password,
        existing_password=password,
    )
    return ValidatedCameraDraft(
        draft=normalized_draft,
        stream_url=build_stream_url(normalized_draft),
        credential_update_required=password is not None or draft.clear_password,
        credential_value=password,
    )


def build_stream_url(draft: CameraDraft, *, password: str | None = None) -> str:
    """Build a URL from editor fields; password is omitted by default."""

    host = draft.host.strip()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    userinfo = ""
    if draft.username:
        userinfo = quote(draft.username, safe="")
        if password is not None:
            userinfo += f":{quote(password, safe='')}"
        userinfo += "@"
    netloc = f"{userinfo}{host}"
    if draft.port is not None:
        netloc += f":{draft.port}"
    path = draft.path or "/"
    if not path.startswith("/"):
        path = f"/{path}"
    return urlunsplit(
        (
            draft.scheme.lower(),
            netloc,
            quote(path, safe="/%:@!$&'()*+,;=-._~"),
            draft.query,
            draft.fragment,
        )
    )


def runtime_stream_url(stream_url: str | None, password: str | None) -> str | None:
    """Insert a credential only in the in-memory URL passed to OpenCV."""

    if stream_url is None or not password:
        return stream_url
    endpoint = parse_camera_url(stream_url)
    if endpoint is None or not endpoint.username:
        return stream_url
    draft = CameraDraft(
        camera_id="runtime",
        slot_index=0,
        name="runtime",
        enabled=True,
        scheme=endpoint.scheme,
        host=endpoint.host,
        port=endpoint.port,
        path=endpoint.path,
        username=endpoint.username,
        query=endpoint.query,
        fragment=endpoint.fragment,
    )
    return build_stream_url(draft, password=password)

