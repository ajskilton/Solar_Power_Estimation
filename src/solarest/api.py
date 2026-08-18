"""HTTP API and static site for the solar yield estimator."""

from __future__ import annotations

import csv
import io
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import geocode
from .pvmodel import MODULE_TYPES, MOUNTS
from .results import REFERENCE_YEAR
from .service import (
    EstimateRequest,
    SizingRequest,
    default_loss_breakdown,
    household_options,
    outcome_to_dict,
    run_estimate,
    run_sizing,
    sizing_to_dict,
)
from .weather import (
    DEFAULT_YEARS,
    MAX_YEARS,
    ArchiveError,
    OpenMeteoArchive,
    SyntheticClearSky,
    default_year_range,
)

log = logging.getLogger(__name__)

WEB_ROOT = Path(__file__).parent / "web"

app = FastAPI(
    title="Solar Power Estimation",
    version="0.1.0",
    description=(
        "Estimates the electricity a rooftop or ground-mounted PV system would "
        "generate over a year at any location, from ECMWF ERA5 reanalysis "
        "weather served by Open-Meteo."
    ),
)


def _build_provider():
    """Choose the weather source from the environment.

    ``SOLAREST_SYNTHETIC=1`` swaps in the offline clear-sky generator, which is
    useful for demos and development but is not a real estimate.
    """
    if os.getenv("SOLAREST_SYNTHETIC", "").lower() in {"1", "true", "yes"}:
        log.warning("SOLAREST_SYNTHETIC is set: serving modelled clear-sky data")
        return SyntheticClearSky()
    return OpenMeteoArchive(cache_dir=os.getenv("SOLAREST_CACHE_DIR", ".cache/weather"))


provider = _build_provider()


@app.get("/api/health")
async def health() -> dict:
    """Liveness probe and a description of the configured data source."""
    return {
        "status": "ok",
        "synthetic": isinstance(provider, SyntheticClearSky),
        "available_years": default_year_range(MAX_YEARS),
    }


@app.get("/api/options")
async def options() -> dict:
    """Enumerate the choices the UI offers, so the two cannot drift apart."""
    return {
        "mounts": [
            {"value": key, "label": value.label} for key, value in MOUNTS.items()
        ],
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


@app.get("/api/geocode")
async def geocode_search(
    q: str = Query(..., min_length=2, max_length=120, description="Place name to look up"),
    count: int = Query(8, ge=1, le=20),
) -> dict:
    """Search for a place by name.

    Returns:
        ``{"results": [...]}`` with the best match first.
    """
    try:
        places = await geocode.search(q, count=count)
    except geocode.GeocodingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"results": [place.to_dict() for place in places]}


@app.post("/api/estimate")
async def estimate(request: EstimateRequest) -> dict:
    """Estimate a year of solar generation for a location and system.

    Returns:
        Annual, monthly, diurnal and typical-year figures. See the README for
        the response shape.
    """
    outcome = await _run(request)
    return outcome_to_dict(outcome, include_hourly=request.include_hourly)


@app.post("/api/estimate.csv")
async def estimate_csv(request: EstimateRequest) -> StreamingResponse:
    """The typical-year hourly series as a CSV download.

    This is the file the battery- and panel-sizing work will consume: 8760
    rows of AC output that can be lined up against metered consumption.
    """
    outcome = await _run(request)
    typical = outcome.result.typical_year

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["# Solar Power Estimation - typical year hourly AC output"])
    writer.writerow([f"# site: {outcome.site.latitude:.4f}, {outcome.site.longitude:.4f}"])
    writer.writerow([f"# timezone: {outcome.site.timezone} (local standard time, no DST)"])
    writer.writerow([f"# system: {outcome.spec.dc_capacity_kw} kWp, tilt {outcome.spec.tilt_deg:.0f} deg, azimuth {outcome.spec.azimuth_deg:.0f} deg"])
    writer.writerow([f"# source: {outcome.weather.source}"])
    writer.writerow([f"# month sources: {typical.month_sources}"])
    writer.writerow(["timestamp", "ac_kw", "energy_kwh", "poa_w_m2", "air_temp_c"])

    import numpy as np

    start = np.datetime64(f"{typical.reference_year}-01-01T00:00:00", "h")
    for index, (ac, poa, temp) in enumerate(
        zip(typical.ac_kw, typical.poa_w_m2, typical.air_temp_c)
    ):
        stamp = start + np.timedelta64(index, "h")
        writer.writerow(
            [str(stamp), f"{ac:.4f}", f"{ac:.4f}", f"{poa:.1f}", f"{temp:.2f}"]
        )

    buffer.seek(0)
    filename = f"solar_typical_year_{outcome.site.latitude:.3f}_{outcome.site.longitude:.3f}.csv"
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/sizing")
async def sizing(request: SizingRequest) -> dict:
    """Estimate how much grid import a battery would avoid for one household.

    Takes the same site and system as ``/api/estimate``, plus the household's
    consumption. Supply metered interval data for a firm answer, or monthly
    bills with an assumption about when electricity is used -- the response
    then carries a confidence band showing what that assumption is worth.

    Returns:
        The estimate payload, plus the energy balance, a battery sizing curve
        and the monthly breakdown.
    """
    try:
        outcome = await run_sizing(request, provider)
    except ArchiveError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return sizing_to_dict(outcome, include_hourly=request.include_hourly)


async def _run(request: EstimateRequest):
    """Shared error handling for the estimate endpoints."""
    try:
        return await run_estimate(request, provider)
    except ArchiveError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if WEB_ROOT.is_dir():

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        """Serve the single-page frontend."""
        return FileResponse(WEB_ROOT / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")
