"""Acquisition: download remote datasets with retries, timeouts, and checksums."""

from __future__ import annotations

import gzip
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from morel.core.errors import DataError

DEFAULT_TIMEOUT = 60
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 1.5
USER_AGENT = "morel/0.1"

DEFAULT_BASE = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/"
LEGACY_BASE = "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/"


def fetch(
    url: str,
    dest: Path | str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
    sha: str | None = None,
    progress: Callable[[int, int | None], None] | None = None,
) -> Path:
    """Fetch a remote URL to a local file with retries and optional checksum.

    Args:
        url: HTTPS URL.
        dest: Local destination path.
        timeout: Per-attempt timeout in seconds.
        retries: Number of retries after the first failure.
        backoff: Exponential backoff factor between attempts.
        sha: Optional SHA256 expected digest; raises DataError on mismatch.
        progress: Optional callback ``(bytes_done, total_bytes_or_None)``.

    Returns
    -------
        The destination path.

    Raises
    ------
        DataError: If the URL cannot be fetched, or the checksum mismatches.
    """
    target = Path(dest).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    last_error: Exception | None = None
    while attempt <= retries:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with (
                urllib.request.urlopen(request, timeout=timeout) as response,
                target.open("wb") as out,
            ):
                total = response.length
                done = 0
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(done, total)
            if sha is not None:
                from morel.data.manifest import checksum

                if checksum(target) != sha:
                    raise DataError(
                        f"checksum mismatch for {target}: expected {sha}, got {checksum(target)}"
                    )
            return target
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            attempt += 1
            if attempt > retries:
                break
            time.sleep(backoff**attempt)
    raise DataError(f"failed to fetch {url} after {retries + 1} attempts: {last_error}")


def download_from_base(
    base: str, category: str, dest: Path | str, *, timeout: float
) -> list[Path]:
    files = [f"{category}_5.json.gz", f"{category}_metadata.json.gz"]
    root = Path(dest).resolve()
    root.mkdir(parents=True, exist_ok=True)
    decompressed: list[Path] = []
    for fname in files:
        archive = root / fname
        final = archive.with_suffix("")
        if final.exists():
            decompressed.append(final)
            continue
        fetch(base + fname, archive, timeout=timeout)
        with gzip.open(archive, "rb") as src, final.open("wb") as out:
            shutil.copyfileobj(src, out)
        os.remove(archive)
        decompressed.append(final)
    return decompressed


def download(category: str, dest: Path | str, *, timeout: float = DEFAULT_TIMEOUT) -> list[Path]:
    """Download Amazon 5-core review and metadata for a category.

    Defaults to the Amazon-Reviews-2023 mirror. Raises an actionable
    :class:`DataError` if the mirror is unreachable.

    Args:
        category: Amazon category slug (e.g. ``"Beauty"``).
        dest: Destination directory.
        timeout: Per-attempt download timeout.

    Returns
    -------
        The list of decompressed file paths.

    Raises
    ------
        DataError: When neither the default nor legacy URL responds.
    """
    try:
        return download_from_base(DEFAULT_BASE, category, dest, timeout=timeout)
    except DataError as primary:
        try:
            return download_from_base(LEGACY_BASE, category, dest, timeout=timeout)
        except DataError as legacy:
            raise DataError(
                f"failed to fetch {category} from {DEFAULT_BASE} ({primary}) "
                f"and from {LEGACY_BASE} ({legacy}); "
                "check network access or use download_legacy directly"
            ) from legacy


def download_legacy(
    category: str, dest: Path | str, *, timeout: float = DEFAULT_TIMEOUT
) -> list[Path]:
    """Download Amazon 5-core review and metadata from the legacy McAuley URL."""
    return download_from_base(LEGACY_BASE, category, dest, timeout=timeout)


__all__ = [
    "DEFAULT_TIMEOUT",
    "DEFAULT_RETRIES",
    "DEFAULT_BACKOFF",
    "DEFAULT_BASE",
    "LEGACY_BASE",
    "fetch",
    "download",
    "download_legacy",
]
