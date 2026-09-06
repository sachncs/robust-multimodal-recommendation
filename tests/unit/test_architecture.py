"""Architecture contracts.

``import-linter`` used to enforce the package layering; it was dropped as a
dependency, which left the layering unenforced and free to rot. These tests
re-establish the contract using only the standard library, so it is checked on
every ordinary test run with no extra tooling.

Two properties are enforced:

1. **Layering** — a package may only import from packages at a strictly lower
   layer (or from itself). This is what keeps the codebase modular: the
   foundation never depends on the things built on top of it.
2. **Acyclicity** — no import cycles between packages, which layering already
   implies but which is worth asserting directly with a clearer failure
   message.
"""

from __future__ import annotations

import ast
import pathlib
from collections import defaultdict

#: Package layers, lowest (most foundational) first. A package may import from
#: its own layer's siblings only if that does not create a cycle; the layer
#: check below allows same-layer imports and the cycle check catches the rest.
LAYERS: tuple[tuple[str, ...], ...] = (
    ("core",),
    ("data", "graph"),
    ("codebook", "complete", "encode", "eval", "recommend", "retrieve", "route"),
    ("pipeline",),
    ("train",),
    ("app", "serve"),
    ("cli",),
)

#: ``morel/__init__.py`` and ``morel/__main__.py`` are the package's own facade
#: and sit above everything; they are allowed to import anything.
ROOT = "<root>"

SOURCE = pathlib.Path(__file__).resolve().parents[2] / "morel"


def layer_of(package: str) -> int:
    """Return the layer index of a package, or a sentinel above all layers."""
    for index, names in enumerate(LAYERS):
        if package in names:
            return index
    return len(LAYERS)


def import_graph() -> dict[str, set[str]]:
    """Map each morel subpackage to the set of morel subpackages it imports."""
    assert SOURCE.is_dir(), f"source tree not found at {SOURCE}"
    edges: dict[str, set[str]] = defaultdict(set)
    for path in sorted(SOURCE.rglob("*.py")):
        if path.name == "_version.py":
            continue
        parts = path.relative_to(SOURCE).parts
        source_pkg = parts[0] if len(parts) > 1 else ROOT
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules = [node.module]
            else:
                continue
            for module in modules:
                if not module.startswith("morel."):
                    continue
                target = module.split(".")[1]
                if target != source_pkg and not target.startswith("_"):
                    edges[source_pkg].add(target)
    return dict(edges)


def test_every_package_is_assigned_to_a_layer() -> None:
    """A new subpackage must be placed in the layering deliberately."""
    packages = {
        path.name
        for path in SOURCE.iterdir()
        if path.is_dir() and (path / "__init__.py").exists() and not path.name.startswith("_")
    }
    declared = {name for names in LAYERS for name in names}
    assert packages == declared, (
        f"packages missing from LAYERS: {sorted(packages - declared)}; "
        f"declared but absent from the tree: {sorted(declared - packages)}"
    )


def test_imports_respect_the_layering() -> None:
    violations = []
    for source, targets in sorted(import_graph().items()):
        if source == ROOT:
            continue
        source_layer = layer_of(source)
        for target in sorted(targets):
            if layer_of(target) > source_layer:
                violations.append(
                    f"{source} (layer {source_layer}) imports {target} (layer {layer_of(target)})"
                )
    assert not violations, "layering violations:\n  " + "\n  ".join(violations)


def test_package_imports_are_acyclic() -> None:
    graph = import_graph()
    visiting: set[str] = set()
    done: set[str] = set()
    cycles: list[str] = []

    def walk(node: str, trail: list[str]) -> None:
        if node in done:
            return
        if node in visiting:
            start = trail.index(node)
            cycles.append(" -> ".join([*trail[start:], node]))
            return
        visiting.add(node)
        for neighbour in sorted(graph.get(node, ())):
            walk(neighbour, [*trail, node])
        visiting.discard(node)
        done.add(node)

    for node in sorted(graph):
        walk(node, [])

    assert not cycles, "import cycles between packages:\n  " + "\n  ".join(cycles)


def test_core_depends_on_nothing_else_in_morel() -> None:
    """The foundation layer must stay free of upward dependencies."""
    assert import_graph().get("core", set()) == set()
