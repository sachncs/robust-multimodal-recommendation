"""Tests for morel.data.acquire."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from morel.core.errors import DataError
from morel.data.acquire import fetch


def test_fetch_writes_file(tmp_path: Path) -> None:
    target = tmp_path / "x.bin"
    with patch("morel.data.acquire.urllib.request.urlopen") as mocked:
        mocked.return_value.__enter__.return_value.read.side_effect = [b"hello", b""]
        path = fetch("https://example.invalid/x.bin", target, timeout=5, retries=0)
    assert path.exists()
    assert path.read_bytes() == b"hello"


def test_fetch_retries_then_raises(tmp_path: Path) -> None:
    target = tmp_path / "x.bin"
    with patch("morel.data.acquire.urllib.request.urlopen", side_effect=OSError("boom")):
        with pytest.raises(DataError):
            fetch("https://example.invalid/x.bin", target, timeout=1, retries=2, backoff=1.0)
