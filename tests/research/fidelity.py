"""Verify that the fidelity registry entries point to real, passing tests."""

from __future__ import annotations

import importlib
from pathlib import Path

from morel.core.fidelity import all as fidelity_all
from morel.core.fidelity import registry, render_json, render_markdown


def split(test_ref: str) -> tuple[str, str]:
    """Convert 'tests/foo/test_bar.py::test_baz' to (module_path, attribute)."""
    parts = test_ref.split("::")
    module = parts[0]
    attr = parts[1] if len(parts) > 1 else None
    return module, attr or ""


def path(test_ref: str) -> bool:
    module, _ = split(test_ref)
    return Path(module).is_file()


def exists(test_ref: str) -> bool:
    module, attr = split(test_ref)
    if not attr or ":" in attr:
        return True
    try:
        mod = importlib.import_module(module.replace("/", ".").rstrip(".py"))
    except Exception:
        return False
    if not hasattr(mod, attr):
        return False
    return callable(getattr(mod, attr)) or hasattr(mod, attr)


class Checker:
    """Aggregated test methods for this module."""

    def fidelity() -> None:
        for entry in fidelity_all():
            assert entry.name
            assert entry.paper
            assert entry.equation
            assert entry.implementation
            assert entry.test
            assert entry.status in {"EXACT", "APPROXIMATE", "INCORRECT", "UNKNOWN"}

    def test() -> None:
        """Every registered test path must point to an existing file on disk."""
        missing: list[str] = []
        for entry in fidelity_all():
            if not path(entry.test):
                missing.append(f"{entry.name}: {entry.test}")
        assert not missing, f"missing test files: {missing}"

    def functions() -> None:
        """Every registered test function name must be importable."""
        missing: list[str] = []
        for entry in fidelity_all():
            if not exists(entry.test):
                missing.append(f"{entry.name}: {entry.test}")
        assert not missing, f"missing test functions: {missing}"

    def registry() -> None:
        """morel.core.__init__ side-effect-imports morel.core.fidelity.register_all."""
        assert len(registry) >= 10

    def render(tmp_path: Path) -> None:
        md_path = tmp_path / "FIDELITY.md"
        json_path = tmp_path / "FIDELITY.json"
        render_markdown(md_path)
        render_json(json_path)
        assert md_path.read_text().count("\n") > 5
        assert json_path.read_text().startswith("{")

    def table(tmp_path: Path) -> None:
        md_path = tmp_path / "FIDELITY.md"
        render_markdown(md_path)
        body = md_path.read_text()
        for entry in fidelity_all():
            assert entry.name in body, f"{entry.name} not in rendered table"
