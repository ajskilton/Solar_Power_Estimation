"""The static build: does it contain a working browser bundle?

These are the checks that would otherwise only fail once the site was live.
The important one is the last: the browser build must not depend on anything
Pyodide will not have.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_site import (  # noqa: E402
    BROWSER_MODULES,
    WEB_ASSETS,
    build,
    modules_declared_in_backend_js,
)

# Available under Pyodide, or in the standard library. Anything else in a
# browser module is a bug that would only show up as a blank page.
PERMITTED_IMPORTS = {
    "numpy",
    "pyodide",
    # standard library used by the model layer
    "__future__", "collections", "dataclasses", "datetime", "hashlib", "json",
    "logging", "math", "pathlib", "typing", "urllib", "zoneinfo", "asyncio",
}

FORBIDDEN = {"httpx", "pydantic", "fastapi", "starlette", "uvicorn"}


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    out = tmp_path_factory.mktemp("site-build") / "site"
    build(out)
    return out


def test_the_javascript_and_the_build_agree_on_the_module_list():
    """backend.js fetches these by name; a mismatch is a 404 at runtime."""
    assert modules_declared_in_backend_js() == BROWSER_MODULES


def test_every_asset_the_page_needs_is_present(site):
    for name in WEB_ASSETS:
        assert (site / name).is_file(), name
    assert (site / ".nojekyll").is_file()  # or Pages hides nothing-underscore paths


def test_every_declared_python_module_is_shipped(site):
    for name in BROWSER_MODULES:
        assert (site / "python" / "solarest" / name).is_file(), name


def test_the_server_only_modules_are_left_out(site):
    """Shipping them would work but invite a confusing import failure."""
    for name in ("api.py", "service.py", "cli.py"):
        assert not (site / "python" / "solarest" / name).exists(), name


def test_the_page_references_its_assets_relatively(site):
    """Absolute /static/ paths would resolve off-site under a Pages subpath."""
    html = (site / "index.html").read_text(encoding="utf-8")
    assert "./app.js" in html
    assert "./styles.css" in html
    assert "/static/" not in html


def guarded_import_nodes(tree: ast.AST) -> set[int]:
    """Ids of import statements sitting directly inside a ``try`` block.

    That is the optional-dependency pattern: the module still imports when the
    package is absent, which is exactly what a browser build needs.
    """
    return {
        id(statement)
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        for statement in node.body
        if isinstance(statement, (ast.Import, ast.ImportFrom))
    }


def unconditional_imports(source: str) -> list[str]:
    """Root package names a module imports at import time, guards excluded."""
    tree = ast.parse(source)
    optional = guarded_import_nodes(tree)
    roots: list[str] = []

    for node in ast.walk(tree):
        if id(node) in optional:
            continue
        if isinstance(node, ast.Import):
            roots += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            # level > 0 is a relative import, i.e. another solarest module.
            roots.append((node.module or "").split(".")[0])
    return [root for root in roots if root]


def test_no_browser_module_imports_something_pyodide_lacks(site):
    """The check that keeps the static site from silently failing to boot."""
    offenders = [
        f"{name}: {root}"
        for name in BROWSER_MODULES
        for root in unconditional_imports(
            (site / "python" / "solarest" / name).read_text(encoding="utf-8")
        )
        if root not in PERMITTED_IMPORTS
    ]
    assert not offenders, f"browser build imports unavailable modules: {offenders}"


@pytest.mark.parametrize("module", ["weather.py", "geocode.py"])
def test_the_forbidden_imports_are_only_ever_optional(site, module):
    """httpx may appear, but only where its absence is handled."""
    source = (site / "python" / "solarest" / module).read_text(encoding="utf-8")
    for root in unconditional_imports(source):
        assert root not in FORBIDDEN, f"{module} imports {root} unconditionally"


def test_the_bundle_stays_small(site):
    """Pyodide is the heavy part; our own code should stay a rounding error."""
    total = sum(f.stat().st_size for f in site.rglob("*") if f.is_file())
    assert total < 1_000_000, f"{total} bytes is larger than expected"


def test_building_twice_replaces_rather_than_accumulates(tmp_path):
    out = tmp_path / "site"
    build(out)
    stray = out / "python" / "solarest" / "leftover.py"
    stray.write_text("# from a previous build", encoding="utf-8")

    build(out)
    assert not stray.exists()
