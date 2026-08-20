from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Path:
    """Keep Windows test files in the workspace-local ACL-safe directory."""

    root = ROOT / ".test-tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{request.node.name}-{uuid4().hex}"
    path.mkdir(mode=0o755)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)

