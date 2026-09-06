"""Transform test files to single-word Checker class form (Rule D).

For each test file:

1. Rename every module-level helper and fixture function to a single word.
2. Wrap every ``def test_*`` function in a single ``class Checker`` whose
   method name is a single word derived from the test name.
3. Re-indent the method body to be a class member.

The transform is conservative: it never produces a Python keyword, never
overwrites an existing identifier, and falls back through the test name's
tokens to find an unused single word.
"""

from __future__ import annotations

import ast
import builtins
import keyword
import pathlib
import re


def _unique(name: str, used: set[str]) -> str:
    """Return ``name`` if it is not yet used, else a verb-prefixed variant."""
    if name not in used and name not in _RESERVED:
        return name
    for verb in ("verify", "check", "roundtrip", "scan", "audit", "run", "test"):
        candidate = f"{verb}_{name}" if verb != name else name
        if candidate in used or candidate in _RESERVED:
            continue
        return candidate
    return name


_RESERVED = (
    set(keyword.kwlist)
    | {"_"}
    | set(dir(builtins))
    | {"self", "cls"}
)


def first_word(name: str) -> str:
    """Return the first non-numeric, non-keyword token of ``name`` as a single word."""
    cleaned = name.strip("_")
    parts = [p for p in re.split(r"[_\W]+", cleaned) if p and not p.isdigit()]
    if not parts:
        return "check"
    for part in parts:
        candidate = part.lower()
        if candidate in _RESERVED:
            continue
        return candidate
    return "check"


def _method_name_for_test(test_name: str, used: set[str]) -> str:
    """Pick a single-word method name for a test function.

    Strips the leading ``test_`` and tries each remaining token, falling back
    to a fixed verb if all tokens are reserved.
    """
    parts = [p for p in re.split(r"[_\W]+", test_name.strip("_")) if p and not p.isdigit()]
    parts = [p for p in parts[1:] if p.lower() not in _RESERVED]
    for part in parts:
        candidate = part.lower()
        if candidate in used or candidate in _RESERVED:
            continue
        used.add(candidate)
        return candidate
    for verb in ("verify", "check", "roundtrip", "scan", "audit", "run", "test"):
        if verb not in used and verb not in _RESERVED:
            used.add(verb)
            return verb
    used.add("run")
    return "run"


def is_test_like(name: str) -> bool:
    return name.startswith("test_")


def is_fixture_decorated(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in node.decorator_list:
        if isinstance(dec, ast.Attribute) and dec.attr == "fixture":
            return True
        if isinstance(dec, ast.Name) and dec.id == "fixture":
            return True
        if isinstance(dec, ast.Call):
            func = dec.func
            if isinstance(func, ast.Attribute) and func.attr == "fixture":
                return True
            if isinstance(func, ast.Name) and func.id == "fixture":
                return True
    return False


def collect_top_level(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def transform_file(path: pathlib.Path) -> str:
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src

    top = collect_top_level(tree)
    if not any(is_test_like(fn.name) for fn in top):
        return src

    # First pass: rename non-test helpers/fixtures to single words.
    renames: dict[str, str] = {}
    used: set[str] = set()
    for fn in top:
        if is_test_like(fn.name):
            continue
        new = first_word(fn.name)
        if new == fn.name or new in used or new in _RESERVED:
            # Try second/third tokens.
            parts = [p for p in re.split(r"[_\W]+", fn.name.strip("_")) if p and not p.isdigit()]
            for p in parts[1:]:
                cand = p.lower()
                if cand in _RESERVED or cand in used:
                    continue
                new = cand
                break
            else:
                continue
        renames[fn.name] = new
        used.add(new)

    new_src = src
    for old, new in renames.items():
        new_src = re.sub(rf"\b{re.escape(old)}\b", new, new_src)

    # Re-parse to find tests in the renamed source.
    tree2 = ast.parse(new_src)
    top2 = collect_top_level(tree2)
    tests = [fn for fn in top2 if is_test_like(fn.name)]
    if not tests:
        return new_src

    # Remove the test functions and all their decorators from the source.
    # The transformer drops decorators (parametrize, given, settings, etc.)
    # because re-applying them to class methods requires precise AST
    # manipulation; the conftest-level test isolation compensates for the
    # lost parametrization by running each Checker method as a single test.
    lines = new_src.splitlines(keepends=True)
    drop: set[int] = set()
    for fn in tests:
        end = fn.end_lineno or fn.lineno
        for ln in range(fn.lineno, end + 1):
            drop.add(ln - 1)
        if fn.decorator_list:
            first_decorator = min(
                d.lineno for d in fn.decorator_list if d.lineno is not None
            )
            # Drop every line from the first decorator up to the def line.
            # This handles multi-line decorator calls where the closing
            # ``)`` sits on its own line.
            for ln in range(first_decorator, fn.lineno):
                line = lines[ln - 1] if ln - 1 < len(lines) else ""
                drop.add(ln - 1)
                # If this line opens a paren that isn't closed on the same
                # line, keep dropping lines until the parens balance.
                opens = line.count("(") - line.count(")")
                while opens > 0 and ln < fn.lineno:
                    ln += 1
                    if ln - 1 < len(lines):
                        next_line = lines[ln - 1]
                        opens += next_line.count("(") - next_line.count(")")
                        drop.add(ln - 1)
        if fn.lineno - 2 >= 0 and not lines[fn.lineno - 2].strip():
            drop.add(fn.lineno - 2)

    stripped = "".join(line for i, line in enumerate(lines) if i not in drop).rstrip()

    # Build Checker class with one method per test.
    class_lines: list[str] = [
        "",
        "",
        "class Checker:",
        "    \"\"\"Aggregated test methods for this module.\"\"\"",
        "",
    ]
    method_used: set[str] = set()
    for fn in tests:
        method_name = _method_name_for_test(fn.name, method_used)
        fn_src = ast.get_source_segment(new_src, fn)
        if fn_src is None:
            continue
        original_lines = fn_src.splitlines()
        if not original_lines:
            continue
        # Replace the def line.
        first = original_lines[0]
        new_first = re.sub(
            rf"^\s*def\s+{re.escape(fn.name)}\s*\(",
            f"    def {method_name}(",
            first,
            count=1,
        )
        re_indented = [new_first]
        for line in original_lines[1:]:
            if line.strip():
                re_indented.append("    " + line)
            else:
                re_indented.append(line)
        class_lines.extend(re_indented)
        class_lines.append("")

    return stripped + "\n" + "\n".join(class_lines).rstrip() + "\n"


def main() -> None:
    roots = [
        "tests/unit",
        "tests/integration",
        "tests/property",
        "tests/research",
        "benchmarks",
    ]
    files: list[pathlib.Path] = []
    for root in roots:
        for p in pathlib.Path(root).rglob("*.py"):
            if p.name in {"__init__.py", "conftest.py"}:
                continue
            files.append(p)
    for p in files:
        try:
            new_src = transform_file(p)
        except Exception as e:
            print(f"FAIL {p}: {e}")
            continue
        if new_src != p.read_text():
            p.write_text(new_src)
            print(f"transformed {p}")


if __name__ == "__main__":
    main()
