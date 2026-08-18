"""Command-line entry points: run an estimate, or serve the website."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .service import (
    BatteryRequest,
    ConsumptionRequest,
    EstimateRequest,
    HouseholdRequest,
    SizingRequest,
    SystemRequest,
    outcome_to_dict,
    run_estimate,
    run_sizing,
    sizing_to_dict,
)
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
    _add_site_arguments(est)
    est.add_argument("--optimise", action="store_true", help="also search tilt and azimuth")
    est.add_argument("--hourly", action="store_true", help="include the 8760-hour series")

    size = sub.add_parser(
        "size",
        help="estimate the grid import a battery would avoid",
        description=(
            "Line a household's consumption up against the typical year and "
            "work out how much electricity it would not have to buy. Give "
            "metered data for a firm answer, or bills plus an assumption "
            "about when electricity is used."
        ),
    )
    _add_site_arguments(size)
    demand = size.add_mutually_exclusive_group(required=True)
    demand.add_argument("--annual-kwh", type=float, help="one figure for the year")
    demand.add_argument(
        "--monthly-kwh",
        help="twelve comma-separated totals off the bills, January first",
    )
    demand.add_argument(
        "--hourly-csv",
        help=(
            "metered interval data: a file of one reading per line, or CSV "
            "whose last column is the reading in kWh"
        ),
    )
    size.add_argument(
        "--archetype",
        default="nine_to_five",
        choices=["nine_to_five", "home_all_day", "working_from_home", "flat"],
        help="when the household uses electricity (default: out at work)",
    )
    size.add_argument(
        "--daytime-fraction",
        type=float,
        default=None,
        help=(
            "share of a weekday's use between 09:00 and 17:00, overriding the "
            "archetype; the single most important input"
        ),
    )
    size.add_argument("--battery-kwh", type=float, default=5.0, help="usable capacity")
    size.add_argument("--json", action="store_true", help="print the full payload")

    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "solarest.api:app", host=args.host, port=args.port, reload=args.reload
        )
        return 0

    provider = SyntheticClearSky() if args.synthetic else OpenMeteoArchive()

    if args.command == "size":
        return _run_sizing(args, provider)

    request = EstimateRequest(
        latitude=args.latitude,
        longitude=args.longitude,
        years=args.years,
        optimise=args.optimise,
        include_hourly=args.hourly,
        system=_system_request(args),
    )

    try:
        outcome = asyncio.run(run_estimate(request, provider))
    except (ArchiveError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    json.dump(outcome_to_dict(outcome, include_hourly=args.hourly), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _add_site_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the location and system arguments shared by every estimate."""
    parser.add_argument("latitude", type=float)
    parser.add_argument("longitude", type=float)
    parser.add_argument("--kwp", type=float, default=4.0, help="DC capacity in kW")
    parser.add_argument("--tilt", type=float, default=None, help="degrees from horizontal")
    parser.add_argument(
        "--azimuth", type=float, default=None, help="degrees clockwise from north"
    )
    parser.add_argument(
        "--mount",
        default="roof_mount",
        choices=["ground_mount", "roof_mount", "roof_integrated"],
    )
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="use the offline clear-sky generator instead of real weather",
    )


def _system_request(args: argparse.Namespace) -> SystemRequest:
    return SystemRequest(
        dc_capacity_kw=args.kwp,
        tilt_deg=args.tilt,
        azimuth_deg=args.azimuth,
        mount=args.mount,
    )


def _run_sizing(args: argparse.Namespace, provider) -> int:
    """Build and run a sizing request, then report it."""
    try:
        consumption = ConsumptionRequest(
            annual_kwh=args.annual_kwh,
            monthly_kwh=_parse_monthly(args.monthly_kwh),
            hourly_kwh=_read_interval_file(args.hourly_csv),
            household=HouseholdRequest(
                archetype=args.archetype, daytime_fraction=args.daytime_fraction
            ),
        )
        request = SizingRequest(
            latitude=args.latitude,
            longitude=args.longitude,
            years=args.years,
            include_hourly=False,
            system=_system_request(args),
            consumption=consumption,
            battery=BatteryRequest(usable_capacity_kwh=args.battery_kwh),
        )
        outcome = asyncio.run(run_sizing(request, provider))
    except (ArchiveError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    payload = sizing_to_dict(outcome)
    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_sizing(payload)
    return 0


def _parse_monthly(raw: str | None) -> list[float] | None:
    """Parse twelve comma-separated monthly totals."""
    if raw is None:
        return None
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if len(values) != 12:
        raise ValueError(f"--monthly-kwh needs 12 values, got {len(values)}")
    try:
        return [float(value) for value in values]
    except ValueError as exc:
        raise ValueError(f"--monthly-kwh must be numeric: {exc}") from exc


def _read_interval_file(path: str | None) -> list[float] | None:
    """Read metered interval data from a plain or comma-separated file.

    Tolerant by design, because smart-meter exports vary: blank lines, comment
    lines and a header row are skipped, and where a line has several fields the
    last numeric one is taken. That covers ``timestamp,kwh`` without needing to
    know which column is which.
    """
    if path is None:
        return None

    values: list[float] = []
    with open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for field in reversed(line.split(",")):
                try:
                    values.append(float(field.strip()))
                except ValueError:
                    continue
                break

    if not values:
        raise ValueError(f"no numeric readings found in {path}")
    return values


def _print_sizing(payload: dict) -> None:
    """Report the sizing result as something a person can read."""
    balance = payload["balance"]
    consumption = payload["consumption"]
    battery = payload["battery"]

    print(f"\nGeneration      {balance['generation_kwh']:>8,.0f} kWh/year")
    print(f"Consumption     {balance['load_kwh']:>8,.0f} kWh/year   ({consumption['source']})")
    print(f"Daytime use     {consumption['daytime_fraction']:>8.0%}         09:00-17:00")

    print(f"\nWith a {battery['usable_capacity_kwh']:g} kWh battery:")
    print(f"  Not bought from the grid   {balance['avoided_import_kwh']:>8,.0f} kWh/year")
    print(f"    used as generated        {balance['direct_use_kwh']:>8,.0f} kWh")
    print(f"    via the battery          {balance['from_battery_kwh']:>8,.0f} kWh")
    print(f"  Still imported             {balance['import_kwh']:>8,.0f} kWh")
    print(f"  Exported                   {balance['export_kwh']:>8,.0f} kWh")
    print(f"  Self-sufficiency           {balance['self_sufficiency_pct']:>8.1f} %")
    print(f"  Self-consumption           {balance['self_consumption_pct']:>8.1f} %")
    print(
        f"  The battery itself adds    {balance['battery_contribution_kwh']:>8,.0f} kWh"
        f"  (PV alone: {balance['pv_only_avoided_import_kwh']:,.0f} kWh)"
    )

    print("\n  kWh   avoided  self-suff  per extra kWh")
    for row in payload["sizing_curve"]:
        marker = " <-" if row["capacity_kwh"] == payload["suggested_capacity_kwh"] else ""
        print(
            f"{row['capacity_kwh']:5.1f} {row['avoided_import_kwh']:9,.0f} "
            f"{row['self_sufficiency_pct']:9.1f}% {row['marginal_kwh_per_kwh']:12,.0f}{marker}"
        )
    print(f"\nSuggested capacity: {payload['suggested_capacity_kwh']:g} kWh")

    band = payload["confidence"]["band"]
    if band:
        print(
            f"Range from the load-shape assumption: "
            f"{band['low_kwh']:,.0f}-{band['high_kwh']:,.0f} kWh avoided"
        )
    print(f"\n{payload['confidence']['note']}\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
