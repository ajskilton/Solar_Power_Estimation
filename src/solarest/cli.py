"""Command-line entry points: run an estimate, or serve the website."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .service import EstimateRequest, SystemRequest, outcome_to_dict, run_estimate
from .weather import ArchiveError, OpenMeteoArchive, SyntheticClearSky


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="solarest",
        description="Estimate a year of solar PV generation at a location.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the web application")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true", help="reload on code changes")

    est = sub.add_parser("estimate", help="print an estimate as JSON")
    est.add_argument("latitude", type=float)
    est.add_argument("longitude", type=float)
    est.add_argument("--kwp", type=float, default=4.0, help="DC capacity in kW")
    est.add_argument("--tilt", type=float, default=None, help="degrees from horizontal")
    est.add_argument("--azimuth", type=float, default=None, help="degrees clockwise from north")
    est.add_argument(
        "--mount",
        default="roof_mount",
        choices=["ground_mount", "roof_mount", "roof_integrated"],
    )
    est.add_argument("--years", type=int, default=10)
    est.add_argument("--optimise", action="store_true", help="also search tilt and azimuth")
    est.add_argument("--hourly", action="store_true", help="include the 8760-hour series")
    est.add_argument(
        "--synthetic",
        action="store_true",
        help="use the offline clear-sky generator instead of real weather",
    )

    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "solarest.api:app", host=args.host, port=args.port, reload=args.reload
        )
        return 0

    request = EstimateRequest(
        latitude=args.latitude,
        longitude=args.longitude,
        years=args.years,
        optimise=args.optimise,
        include_hourly=args.hourly,
        system=SystemRequest(
            dc_capacity_kw=args.kwp,
            tilt_deg=args.tilt,
            azimuth_deg=args.azimuth,
            mount=args.mount,
        ),
    )
    provider = SyntheticClearSky() if args.synthetic else OpenMeteoArchive()

    try:
        outcome = asyncio.run(run_estimate(request, provider))
    except (ArchiveError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    json.dump(outcome_to_dict(outcome, include_hourly=args.hourly), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
