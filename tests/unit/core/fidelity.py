"""Tests for morel.core.fidelity."""

from __future__ import annotations

import json
from pathlib import Path

from morel.core.fidelity import Entry, clear, register, render, render_json


class Checker:
    """Aggregated test methods for this module."""

    def register(self, tmp_path: Path) -> None:
        clear()
        entry = Entry(
            name="ACS",
            paper="Li et al. 2026",
            equation="Alg 1",
            status="EXACT",
            implementation="morel.retrieve.acs.compute",
            test="tests/research/test_retrieval_paper.py",
        )
        register(entry)
        md_path = tmp_path / "fid.md"
        render(md_path)
        text = md_path.read_text(encoding="utf-8")
        assert "ACS" in text
        assert "EXACT" in text
        json_path = tmp_path / "fid.json"
        render_json(json_path)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert any(e["name"] == "ACS" for e in payload["entries"])
        clear()