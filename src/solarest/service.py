"""Application layer: request validation, and the two use cases.

Kept separate from :mod:`solarest.api` so the same logic is reachable from the
CLI and from tests without going through HTTP.

This module owns only the pydantic request models and the async orchestration.
Defaults and response shaping live in :mod:`solarest.presenter`, which depends
on nothing but numpy -- that is what lets the browser build run the same model
chain without pydantic or FastAPI. Request models here validate, then hand
straight over to the presenter's builders, so there is one definition of what
a default system or a default household is.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .battery import BatterySpec
from .load import HouseholdShape, LoadProfile
from .optimize import OrientationResult, optimise_orientation
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
from .pvmodel import SystemSpec
from .results import EstimateResult, summarise
from .simulate import SitePrecompute, precompute
from .sizing import size_for_household
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
        return build_system_spec(latitude, **self.model_dump())


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


class HouseholdRequest(BaseModel):
    """When a household uses electricity.

    Only relevant when demand is synthesised from bills; metered data speaks
    for itself.
    """

    archetype: Literal["nine_to_five", "home_all_day", "working_from_home", "flat"] = (
        "nine_to_five"
    )
    daytime_fraction: float | None = Field(
        None,
        gt=0.0,
        lt=1.0,
        description=(
            "Share of a weekday's electricity used between 09:00 and 17:00. "
            "Overrides the archetype. The single most important input here."
        ),
    )
    weekend_archetype: (
        Literal["nine_to_five", "home_all_day", "working_from_home", "flat"] | None
    ) = None
    weekend_daytime_fraction: float | None = Field(None, gt=0.0, lt=1.0)

    def to_shape(self) -> HouseholdShape:
        """Build the domain object."""
        return HouseholdShape(
            archetype=self.archetype,
            daytime_fraction=self.daytime_fraction,
            weekend_archetype=self.weekend_archetype,
            weekend_daytime_fraction=self.weekend_daytime_fraction,
        )


class BatteryRequest(BaseModel):
    """The battery to headline."""

    usable_capacity_kwh: float = Field(5.0, ge=0.0, le=200.0)
    max_charge_kw: float | None = Field(None, gt=0.0, le=100.0)
    max_discharge_kw: float | None = Field(None, gt=0.0, le=100.0)
    round_trip_efficiency: float = Field(0.90, gt=0.0, le=1.0)
    initial_soc_frac: float = Field(0.0, ge=0.0, le=1.0)

    def to_spec(self) -> BatterySpec:
        """Build the domain object."""
        return build_battery_spec(**self.model_dump())


class ConsumptionRequest(BaseModel):
    """How much electricity the household uses, and how well that is known.

    Three ways in, in descending order of how much the answer can be trusted:

    * ``hourly_kwh`` -- metered interval data, at any regular resolution.
    * ``monthly_kwh`` -- twelve totals read off the bills.
    * ``annual_kwh`` -- one figure, split across the year by a seasonal curve.

    The last two are paired with ``household`` to decide *when* the energy is
    used, and the response then carries a confidence band on the result.
    """

    annual_kwh: float | None = Field(None, gt=0.0, le=1_000_000.0)
    monthly_kwh: list[float] | None = Field(None, min_length=12, max_length=12)
    hourly_kwh: list[float] | None = Field(
        None,
        min_length=8760,
        max_length=527_040,
        description="Metered interval data for a whole year, any regular resolution",
    )
    household: HouseholdRequest = Field(default_factory=HouseholdRequest)

    @model_validator(mode="after")
    def _exactly_one_source(self) -> ConsumptionRequest:
        given = [
            name
            for name in ("annual_kwh", "monthly_kwh", "hourly_kwh")
            if getattr(self, name) is not None
        ]
        if len(given) != 1:
            raise ValueError(
                "give exactly one of annual_kwh, monthly_kwh or hourly_kwh, "
                f"got {given or 'none'}"
            )
        return self

    def to_profile(self, latitude: float) -> LoadProfile:
        """Build the 8760-hour demand series this request describes."""
        return build_load_profile(
            latitude,
            annual_kwh=self.annual_kwh,
            monthly_kwh=self.monthly_kwh,
            hourly_kwh=self.hourly_kwh,
            household=self.household.to_shape(),
        )


class SizingRequest(EstimateRequest):
    """An estimate request, plus the household the system has to serve."""

    consumption: ConsumptionRequest
    battery: BatteryRequest = Field(default_factory=BatteryRequest)
    sweep_kwh: list[float] | None = Field(
        None, max_length=40, description="Battery capacities for the sizing curve"
    )
    band_spread: float = Field(
        0.07,
        gt=0.0,
        le=0.3,
        description="Half-width of the daytime-fraction band explored, absolute",
    )


async def run_sizing(
    request: SizingRequest, provider: WeatherProvider
) -> SizingOutcome:
    """Estimate generation, then work out what a battery would avoid importing.

    Args:
        request: The validated request.
        provider: Weather source.

    Returns:
        A :class:`SizingOutcome`.
    """
    estimate = await run_estimate(request, provider)
    load = request.consumption.to_profile(estimate.site.latitude)

    sizing = await asyncio.to_thread(
        size_for_household,
        estimate.result.typical_year,
        load,
        request.battery.to_spec(),
        sweep_kwh=request.sweep_kwh,
        band_spread=request.band_spread,
        latitude=estimate.site.latitude,
    )
    return SizingOutcome(estimate=estimate, sizing=sizing)


# Re-exported so existing importers of solarest.service keep working; the
# definitions live in solarest.presenter.
__all__ = [
    "BatteryRequest",
    "ConsumptionRequest",
    "EstimateOutcome",
    "EstimateRequest",
    "HouseholdRequest",
    "SizingOutcome",
    "SizingRequest",
    "SystemRequest",
    "default_loss_breakdown",
    "household_options",
    "outcome_to_dict",
    "run_estimate",
    "run_sizing",
    "sizing_to_dict",
]
