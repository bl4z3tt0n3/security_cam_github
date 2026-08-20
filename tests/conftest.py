from __future__ import annotations

import sys
from pathlib import Path
import shutil
from uuid import uuid4

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Path:
    """Use a workspace-local test directory with Windows-compatible ACLs.

    The managed execution environment rejects Python-created directories made
    with mode 0700.  Pytest's built-in ``tmp_path`` uses that mode on Windows,
    so keep the same fixture contract while creating a normal user-private
    workspace directory explicitly.
    """

    root = ROOT / ".test-tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{request.node.name}-{uuid4().hex}"
    path.mkdir(mode=0o755)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
