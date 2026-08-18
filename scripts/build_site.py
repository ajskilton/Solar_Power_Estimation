"""Assemble the static site that GitHub Pages serves.

There is no bundler and no transpiler. The site is the same frontend the
FastAPI server hosts, plus the Python package copied in beside it, so that
`backend.js` can load the model into Pyodide at runtime. What runs in the
browser is byte-for-byte what is in `src/`.

    python scripts/build_site.py [--out site]

The only real work is deciding which Python modules to ship. The browser build
needs nothing that imports FastAPI or pydantic, so `api.py`, `service.py` and
`cli.py` are deliberately left out -- shipping them would work, since the
imports are never executed, but it would invite someone to import one and be
puzzled when Pyodide could not find pydantic.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "src" / "solarest"
WEB = PACKAGE / "web"

# Kept in step with PYTHON_MODULES in web/backend.js, which fetches these by
# name at runtime. The check below fails the build if the two ever diverge.
BROWSER_MODULES = [
    "__init__.py",
    "solarpos.py",
    "irradiance.py",
    "pvmodel.py",
    "simulate.py",
    "results.py",
    "optimize.py",
    "load.py",
    "battery.py",
    "sizing.py",
    "geocode.py",
    "weather.py",
    "presenter.py",
    "browser.py",
    "webapp.py",
]

# Static assets copied verbatim from the frontend the server already serves.
WEB_ASSETS = ["index.html", "app.js", "charts.js", "backend.js", "styles.css"]

# `solarest/__init__.py` imports the whole public surface, including modules
# that need pydantic. In the browser the package is only ever a namespace for
# the modules below, so it ships with a stub instead.
BROWSER_INIT = '''"""solarest, trimmed for the browser build.

The full package's ``__init__`` re-exports the HTTP service layer, which needs
pydantic and FastAPI. Neither is available -- or wanted -- under Pyodide, so
this build leaves the package a bare namespace and callers import the modules
they need directly. :mod:`solarest.webapp` is the entry point.
"""

__version__ = "0.1.0"
'''


def modules_declared_in_backend_js() -> list[str]:
    """Parse the module list out of backend.js, to check it against ours."""
    text = (WEB / "backend.js").read_text(encoding="utf-8")
    start = text.index("const PYTHON_MODULES = [")
    end = text.index("];", start)
    body = text[start + len("const PYTHON_MODULES = ["): end]
    return [
        line.strip().strip(",").strip('"')
        for line in body.splitlines()
        if line.strip().startswith('"')
    ]


def build(out: Path, pyodide_base: str | None = None) -> None:
    """Write the complete static site into ``out``.

    Args:
        out: Output directory; replaced if it already exists.
        pyodide_base: Serve the Python runtime from here instead of the public
            CDN. Used for self-hosted deployments, and by the tests.
    """
    declared = modules_declared_in_backend_js()
    if declared != BROWSER_MODULES:
        raise SystemExit(
            "backend.js and build_site.py disagree about which Python modules "
            f"the browser build needs.\n  backend.js: {declared}\n  build:      {BROWSER_MODULES}"
        )

    if out.exists():
        shutil.rmtree(out)
    python_out = out / "python" / "solarest"
    python_out.mkdir(parents=True)

    for name in WEB_ASSETS:
        shutil.copy2(WEB / name, out / name)

    if pyodide_base:
        index = out / "index.html"
        index.write_text(
            index.read_text(encoding="utf-8").replace(
                "<head>", f'<head>\n<meta name="pyodide-base" content="{pyodide_base}">', 1
            ),
            encoding="utf-8",
        )

    for name in BROWSER_MODULES:
        if name == "__init__.py":
            (python_out / name).write_text(BROWSER_INIT, encoding="utf-8")
            continue
        source = PACKAGE / name
        if not source.exists():
            raise SystemExit(f"missing Python module {source}")
        shutil.copy2(source, python_out / name)

    # Tell Pages not to run the output through Jekyll, which would otherwise
    # ignore any file or directory beginning with an underscore.
    (out / ".nojekyll").write_text("", encoding="utf-8")

    (out / "build-info.json").write_text(
        json.dumps(
            {
                "modules": BROWSER_MODULES,
                "assets": WEB_ASSETS,
                "note": "Static build; the model runs in the browser under Pyodide.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"built {out} — {len(list(out.rglob('*')))} files, {total / 1024:.0f} kB")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="site", help="output directory (default: site)")
    parser.add_argument(
        "--pyodide-base",
        default=None,
        help="serve the Python runtime from here instead of the public CDN",
    )
    args = parser.parse_args(argv)
    build(Path(args.out).resolve(), pyodide_base=args.pyodide_base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
