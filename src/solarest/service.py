"""Application layer: request models and the estimate use case.

Kept separate from :mod:`solarest.api` so the same logic is reachable from the
CLI, from tests, and later from the battery-sizing work, without going through
HTTP.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .optimize import OrientationResult, equator_facing_azimuth, optimise_orientation, rule_of_thumb_tilt
from .pvmodel import DEFAULT_LOSSES, MODULE_TYPES, MOUNTS, SystemSpec, combine_losses
from .results import EstimateResult, result_to_dict, summarise
from .simulate import SitePrecompute, precompute
from .weather import (
    DEFAULT_YEARS,
    MAX_YEARS,
    SiteInfo,
    WeatherProvider,
    WeatherSeries,
    default_year_range,
)


class SystemRequest(BaseModel):
    """PV system parameters supplied by the caller.

    Defaults describe a typical domestic rooftop installation.
    """

    dc_capacity_kw: float = Field(
        4.0, gt=0, le=1000, description="Nameplate DC capacity at STC, kW"
    )
    tilt_deg: float | None = Field(
        None, ge=0, le=90, description="Array tilt; defaults to a latitude rule of thumb"
    )
    azimuth_deg: float | None = Field(
        None, ge=0, le=360, description="Array azimuth clockwise from north; defaults to equator-facing"
    )
    mount: Literal["ground_mount", "roof_mount", "roof_integrated"] = "roof_mount"
    module_type: Literal["standard", "premium", "thin_film"] = "premium"
    dc_ac_ratio: float = Field(1.15, gt=0.5, le=2.5)
    inverter_efficiency: float = Field(0.96, gt=0.5, le=1.0)
    albedo: float = Field(0.20, ge=0.0, le=1.0)
    system_losses_pct: float | None = Field(
        None,
        ge=0.0,
        lt=90.0,
        description="Combined DC-side losses as a percentage; defaults to the PVWatts stack",
    )
    shading_pct: float | None = Field(
        None, ge=0.0, lt=90.0, description="Override just the shading loss, as a percentage"
    )
    transposition_model: Literal["perez", "hdkr"] = "perez"

    def to_spec(self, latitude: float) -> SystemSpec:
        """Resolve defaults against the site and build a :class:`SystemSpec`."""
        losses = dict(DEFAULT_LOSSES)
        if self.shading_pct is not None:
            losses["shading"] = self.shading_pct / 100.0
        if self.system_losses_pct is not None:
            # An explicit total replaces the itemised stack entirely.
            losses = {"system": self.system_losses_pct / 100.0}

        return SystemSpec(
            dc_capacity_kw=self.dc_capacity_kw,
            tilt_deg=(
                rule_of_thumb_tilt(latitude) if self.tilt_deg is None else self.tilt_deg
            ),
            azimuth_deg=(
                equator_facing_azimuth(latitude)
                if self.azimuth_deg is None
                else self.azimuth_deg
            ),
            mount=self.mount,
            module_type=self.module_type,
            dc_ac_ratio=self.dc_ac_ratio,
            inverter_efficiency=self.inverter_efficiency,
            albedo=self.albedo,
            losses=losses,
            transposition_model=self.transposition_model,
        )


class EstimateRequest(BaseModel):
    """A request to estimate annual yield at one location."""

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    name: str | None = Field(None, max_length=200, description="Display label for the site")
    years: int = Field(
        DEFAULT_YEARS,
        ge=1,
        le=MAX_YEARS,
        description="How many recent complete calendar years to average over",
    )
    system: SystemRequest = Field(default_factory=SystemRequest)
    include_hourly: bool = Field(
        True, description="Include the 8760-hour typical-year series in the response"
    )
    optimise: bool = Field(
        False, description="Also search for the best tilt and azimuth at this site"
    )

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str | None) -> str | None:
        return value.strip() or None if value else None


@dataclass(frozen=True)
class EstimateOutcome:
    """Everything the estimate use case produces."""

    site: SiteInfo
    weather: WeatherSeries
    spec: SystemSpec
    result: EstimateResult
    orientation: OrientationResult | None


async def run_estimate(
    request: EstimateRequest, provider: WeatherProvider
) -> EstimateOutcome:
    """Fetch weather, model the system, and aggregate the results.

    Args:
        request: The validated request.
        provider: Weather source (Open-Meteo, or the synthetic provider).

    Returns:
        An :class:`EstimateOutcome`.
    """
    years = default_year_range(request.years)
    site, weather = await asyncio.gather(
        provider.site_info(request.latitude, request.longitude),
        provider.fetch(request.latitude, request.longitude, years),
    )

    spec = request.system.to_spec(site.latitude)
    prepared = precompute(weather, utc_offset_seconds=site.utc_offset_seconds)

    # numpy releases the GIL for the heavy work, but the aggregation is still
    # a few hundred milliseconds of Python; keep it off the event loop so
    # concurrent requests are not blocked.
    result, orientation = await asyncio.to_thread(
        _model, prepared, spec, request.optimise
    )
    return EstimateOutcome(
        site=site, weather=weather, spec=spec, result=result, orientation=orientation
    )


def _model(
    prepared: SitePrecompute, spec: SystemSpec, optimise: bool
) -> tuple[EstimateResult, OrientationResult | None]:
    result = summarise(prepared.simulate(spec))
    orientation = optimise_orientation(prepared, spec) if optimise else None
    return result, orientation


def outcome_to_dict(outcome: EstimateOutcome, include_hourly: bool = True) -> dict:
    """Serialise an :class:`EstimateOutcome` into the public JSON shape."""
    payload = {
        "site": {
            "latitude": round(outcome.site.latitude, 4),
            "longitude": round(outcome.site.longitude, 4),
            "elevation_m": round(outcome.site.elevation_m, 1),
            "timezone": outcome.site.timezone,
            "utc_offset_hours": outcome.site.utc_offset_seconds / 3600.0,
        },
        "weather": {
            "source": outcome.weather.source,
            "synthetic": outcome.weather.synthetic,
            "years": outcome.result.annual.years_used,
            "hours": len(outcome.weather),
        },
        "system": {
            "dc_capacity_kw": outcome.spec.dc_capacity_kw,
            "tilt_deg": round(outcome.spec.tilt_deg, 1),
            "azimuth_deg": round(outcome.spec.azimuth_deg, 1),
            "mount": outcome.spec.mount,
            "mount_label": MOUNTS[outcome.spec.mount].label,
            "module_type": outcome.spec.module_type,
            "temperature_coefficient_pct_per_c": round(
                MODULE_TYPES[outcome.spec.module_type] * 100, 3
            ),
            "dc_ac_ratio": outcome.spec.dc_ac_ratio,
            "inverter_ac_kw": round(outcome.spec.ac_capacity_kw, 3),
            "albedo": outcome.spec.albedo,
            "system_losses_pct": round(outcome.spec.total_loss_fraction * 100, 2),
            "transposition_model": outcome.spec.transposition_model,
        },
        **result_to_dict(outcome.result, hourly=include_hourly),
    }

    if outcome.orientation is not None:
        best = outcome.orientation
        payload["orientation"] = {
            "best_tilt_deg": round(best.best_tilt_deg, 1),
            "best_azimuth_deg": round(best.best_azimuth_deg, 1),
            "best_annual_kwh": round(best.best_annual_kwh, 1),
            "baseline_annual_kwh": round(best.baseline_annual_kwh, 1),
            "improvement_pct": round(best.improvement_pct, 2),
            "surface": [
                {
                    "tilt_deg": round(c.tilt_deg, 1),
                    "azimuth_deg": round(c.azimuth_deg, 1),
                    "annual_kwh": round(c.annual_kwh, 1),
                }
                for c in best.surface
            ],
        }
    return payload


def default_loss_breakdown() -> dict[str, float]:
    """The default itemised loss stack, as percentages, plus its combined total."""
    items = {name: round(value * 100, 2) for name, value in DEFAULT_LOSSES.items()}
    items["combined_total"] = round(combine_losses(DEFAULT_LOSSES) * 100, 2)
    return items
