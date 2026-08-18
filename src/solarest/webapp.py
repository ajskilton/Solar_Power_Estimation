"""The surface the browser page calls into.

Deliberately narrow and deliberately plain: every function takes and returns
ordinary dicts and lists, because everything crossing the JavaScript boundary
has to survive Pyodide's conversion anyway. No pydantic, no FastAPI -- the
validation here is the model layer's own, which raises ``ValueError`` for a
bad system and ``ArchiveError`` when Open-Meteo will not play.

The shapes returned match the HTTP API's exactly, so the same front end talks
to either without knowing which it has. That is what keeps one set of charts
working against a local FastAPI server and a static GitHub Pages build.
"""

from __future__ import annotations

from typing import Any

from .browser import BrowserArchive, browser_geocode
from .load import HouseholdShape
from .optimize import optimise_orientation
from .presenter import (
    EstimateOutcome,
    SizingOutcome,
    build_battery_spec,
    build_load_profile,
    build_system_spec,
    default_loss_breakdown,
    household_options,
    outcome_to_dict,
    sizing_to_dict,
)
from .pvmodel import MODULE_TYPES, MOUNTS
from .results import REFERENCE_YEAR, summarise
from .simulate import precompute
from .sizing import size_for_household
from .weather import (
    DEFAULT_YEARS,
    MAX_YEARS,
    SyntheticClearSky,
    default_year_range,
)

# One archive per page load, so its in-memory cache survives between runs:
# changing the tilt and pressing estimate again costs no network at all.
_archive: BrowserArchive | None = None


def _provider(synthetic: bool = False):
    global _archive
    if synthetic:
        return SyntheticClearSky()
    if _archive is None:
        _archive = BrowserArchive()
    return _archive


def options() -> dict:
    """The same payload as ``GET /api/options``."""
    return {
        "mounts": [{"value": key, "label": value.label} for key, value in MOUNTS.items()],
        "module_types": [
            {
                "value": key,
                "label": key.replace("_", " ").title(),
                "temperature_coefficient_pct_per_c": round(value * 100, 3),
            }
            for key, value in MODULE_TYPES.items()
        ],
        "default_losses_pct": default_loss_breakdown(),
        "default_years": DEFAULT_YEARS,
        "max_years": MAX_YEARS,
        "reference_year": REFERENCE_YEAR,
        "household_archetypes": household_options(),
    }


async def geocode(query: str, count: int = 8) -> dict:
    """The same payload as ``GET /api/geocode``."""
    places = await browser_geocode(query, count=count)
    return {"results": [place.to_dict() for place in places]}


async def _run(request: dict[str, Any]) -> EstimateOutcome:
    """Fetch weather and model the system described by ``request``."""
    latitude = float(request["latitude"])
    longitude = float(request["longitude"])
    years = default_year_range(int(request.get("years", DEFAULT_YEARS)))
    provider = _provider(bool(request.get("synthetic", False)))

    site = await provider.site_info(latitude, longitude)
    weather = await provider.fetch(latitude, longitude, years)

    spec = build_system_spec(site.latitude, **(request.get("system") or {}))
    prepared = precompute(weather, utc_offset_seconds=site.utc_offset_seconds)

    result = summarise(prepared.simulate(spec))
    orientation = (
        optimise_orientation(prepared, spec) if request.get("optimise") else None
    )
    return EstimateOutcome(
        site=site, weather=weather, spec=spec, result=result, orientation=orientation
    )


async def estimate(request: dict[str, Any]) -> dict:
    """The same payload as ``POST /api/estimate``.

    Args:
        request: Latitude, longitude, years, system, optimise, include_hourly
            -- the JSON body the HTTP endpoint takes.

    Returns:
        The estimate as a plain dict.
    """
    outcome = await _run(request)
    return outcome_to_dict(outcome, include_hourly=bool(request.get("include_hourly")))


async def sizing(request: dict[str, Any]) -> dict:
    """The same payload as ``POST /api/sizing``.

    Args:
        request: An estimate request plus ``consumption`` and ``battery``.

    Returns:
        The estimate, the energy balance, the sizing curve and the typical day.
    """
    outcome = await _run(request)

    consumption = dict(request.get("consumption") or {})
    household = consumption.pop("household", None) or {}
    load = build_load_profile(
        outcome.site.latitude,
        household=HouseholdShape(**household),
        **consumption,
    )

    sized = size_for_household(
        outcome.result.typical_year,
        load,
        build_battery_spec(**(request.get("battery") or {})),
        sweep_kwh=request.get("sweep_kwh"),
        band_spread=float(request.get("band_spread", 0.07)),
        latitude=outcome.site.latitude,
    )
    return sizing_to_dict(
        SizingOutcome(estimate=outcome, sizing=sized),
        include_hourly=bool(request.get("include_hourly")),
    )
