"""Paper-fidelity registry.

Every algorithm component declares a `Fidelity` entry. The registry renders
``docs/FIDELITY.md`` and ``docs/FIDELITY.json`` from this state, so the
documentation can never drift from the implementation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

Status = Literal["EXACT", "APPROXIMATE", "INCORRECT", "UNKNOWN"]


@dataclass(frozen=True)
class Entry:
    """One component's fidelity declaration."""

    name: str
    paper: str
    equation: str
    status: Status
    implementation: str
    test: str
    deviation: str | None = None
    notes: str | None = None


registry: dict[str, Entry] = {}


def register(entry: Entry) -> Entry:
    """Register a fidelity entry. Use as a decorator."""
    registry[entry.name] = entry
    return entry


def render_markdown(target: Path | str) -> None:
    """Render the registry as a Markdown report."""
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Paper Fidelity Report",
        "",
        f"Generated: {datetime.now(tz=UTC).isoformat()}",
        "",
        "| Component | Status | Paper | Equation | Implementation | Test | Deviation |",
        "|-----------|--------|-------|----------|----------------|------|-----------|",
    ]
    for entry in sorted(registry.values(), key=lambda e: e.name):
        status = entry.status
        deviation = entry.deviation or "—"
        lines.append(
            f"| `{entry.name}` | **{status}** | {entry.paper} | {entry.equation} | "
            f"`{entry.implementation}` | `{entry.test}` | {deviation} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_json(target: Path | str) -> None:
    """Render the registry as a JSON file."""
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": datetime.now(tz=UTC).isoformat(),
        "entries": [asdict(entry) for entry in sorted(registry.values(), key=lambda e: e.name)],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def clear() -> None:
    """Clear the registry. Used by tests."""
    registry.clear()


def all() -> list[Entry]:
    """Return all registered entries sorted by name."""
    return sorted(registry.values(), key=lambda e: e.name)


__all__ = [
    "Entry",
    "Status",
    "registry",
    "register",
    "render_markdown",
    "render_json",
    "clear",
    "all",
]
