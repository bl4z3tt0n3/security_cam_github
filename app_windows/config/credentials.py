"""Local credential storage for the Windows monitor.

Passwords never belong in the central YAML document.  The Windows
implementation stores DPAPI-protected blobs in a small ignored sidecar.  The
in-memory implementation is deliberately useful for fake mode and tests so
those paths never need a platform-specific dependency.
"""

from __future__ import annotations

import base64
import ctypes
import json
import os
from pathlib import Path
import tempfile
from typing import Protocol


class CredentialStoreError(RuntimeError):
    """Raised when local credential storage cannot be read or written."""


class CredentialStore(Protocol):
    def get(self, camera_id: str) -> str | None:
        """Return one password, or ``None`` when no password is stored."""

    def apply(self, updates: dict[str, str | None]) -> None:
        """Atomically apply password updates keyed by camera id."""


class InMemoryCredentialStore:
    """Small injectable credential store used by tests and fake mode."""

    def __init__(self, values: dict[str, str] | None = None) -> None:
        self._values = dict(values or {})

    def get(self, camera_id: str) -> str | None:
        return self._values.get(camera_id)

    def apply(self, updates: dict[str, str | None]) -> None:
        for camera_id, value in updates.items():
            if value:
                self._values[camera_id] = value
            else:
                self._values.pop(camera_id, None)


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", ctypes.c_uint32),
        ("pbData", ctypes.POINTER(ctypes.c_byte)),
    ]


class DpapiCredentialStore:
    """Current-user DPAPI store persisted as encrypted JSON blobs.

    DPAPI protects the values with the current Windows user profile.  The
    sidecar contains only base64 ciphertext and camera ids, never plaintext
    passwords.  The sidecar itself is also ignored by the repository.
    """

    _VERSION = 1
    _PROTECTION_FLAGS = 0x1  # CRYPTPROTECT_UI_FORBIDDEN

    def __init__(self, path: Path | str) -> None:
        if os.name != "nt":
            raise CredentialStoreError("DPAPI credentials require Windows")
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def get(self, camera_id: str) -> str | None:
        encoded = self._load().get(camera_id)
        if encoded is None:
            return None
        try:
            encrypted = base64.b64decode(encoded.encode("ascii"), validate=True)
            return self._unprotect(encrypted).decode("utf-8")
        except (ValueError, UnicodeError, OSError, ctypes.ArgumentError) as exc:
            raise CredentialStoreError("stored camera credential is unreadable") from exc

    def apply(self, updates: dict[str, str | None]) -> None:
        current = self._load()
        for camera_id, password in updates.items():
            normalized_id = camera_id.strip()
            if not normalized_id:
                raise CredentialStoreError("camera id cannot be empty")
            if password:
                encrypted = self._protect(password.encode("utf-8"))
                current[normalized_id] = base64.b64encode(encrypted).decode("ascii")
            else:
                current.pop(normalized_id, None)

        payload = {
            "version": self._VERSION,
            "cameras": dict(sorted(current.items())),
        }
        if not current:
            try:
                self._path.unlink()
            except FileNotFoundError:
                return
            except OSError as exc:
                raise CredentialStoreError("cannot remove local credential sidecar") from exc
            return
        _atomic_write_text(
            self._path,
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        )

    def _load(self) -> dict[str, str]:
        if not self._path.is_file():
            return {}
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            values = payload.get("cameras", {})
            if not isinstance(values, dict):
                raise ValueError("cameras must be an object")
            result: dict[str, str] = {}
            for camera_id, encoded in values.items():
                if not isinstance(camera_id, str) or not isinstance(encoded, str):
                    raise ValueError("invalid credential entry")
                result[camera_id] = encoded
            return result
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise CredentialStoreError("cannot read local credential sidecar") from exc

    @classmethod
    def _protect(cls, value: bytes) -> bytes:
        return cls._crypt(value, protect=True)

    @classmethod
    def _unprotect(cls, value: bytes) -> bytes:
        return cls._crypt(value, protect=False)

    @classmethod
    def _crypt(cls, value: bytes, *, protect: bool) -> bytes:
        if os.name != "nt":
            raise CredentialStoreError("DPAPI credentials require Windows")

        crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        input_buffer = ctypes.create_string_buffer(value)
        input_blob = _DataBlob(
            len(value),
            ctypes.cast(input_buffer, ctypes.POINTER(ctypes.c_byte)),
        )
        output_blob = _DataBlob()
        description = "Local Security Camera"

        if protect:
            ok = crypt32.CryptProtectData(
                ctypes.byref(input_blob),
                description,
                None,
                None,
                None,
                cls._PROTECTION_FLAGS,
                ctypes.byref(output_blob),
            )
        else:
            ok = crypt32.CryptUnprotectData(
                ctypes.byref(input_blob),
                None,
                None,
                None,
                None,
                cls._PROTECTION_FLAGS,
                ctypes.byref(output_blob),
            )
        if not ok:
            raise CredentialStoreError("Windows DPAPI operation failed")

        try:
            return ctypes.string_at(output_blob.pbData, output_blob.cbData)
        finally:
            kernel32.LocalFree(output_blob.pbData)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError as exc:
        raise CredentialStoreError("cannot write local credential sidecar") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

