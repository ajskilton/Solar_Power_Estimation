"""Application layer: request models and the estimate use case.

Kept separate from :mod:`solarest.api` so the same logic is reachable from the
CLI, from tests, and later from the battery-sizing work, without going through
HTTP.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .battery import BatterySpec
from .load import ARCHETYPES, HouseholdShape, LoadProfile, synthesise
from .optimize import OrientationResult, equator_facing_azimuth, optimise_orientation, rule_of_thumb_tilt
from .pvmodel import DEFAULT_LOSSES, MODULE_TYPES, MOUNTS, SystemSpec, combine_losses
from .results import EstimateResult, result_to_dict, summarise
from .simulate import SitePrecompute, precompute
from .sizing import ConfidenceBand, SizingResult, size_for_household
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
        return BatterySpec(
            usable_capacity_kwh=self.usable_capacity_kwh,
            max_charge_kw=self.max_charge_kw,
            max_discharge_kw=self.max_discharge_kw,
            round_trip_efficiency=self.round_trip_efficiency,
            initial_soc_frac=self.initial_soc_frac,
        )


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
        if self.hourly_kwh is not None:
            return LoadProfile.from_hourly(self.hourly_kwh)
        return synthesise(
            annual_kwh=self.annual_kwh,
            monthly_kwh=self.monthly_kwh,
            household=self.household.to_shape(),
            latitude=latitude,
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


@dataclass(frozen=True)
class SizingOutcome:
    """An estimate and the sizing built on top of it."""

    estimate: EstimateOutcome
    sizing: SizingResult


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


def _band_to_dict(band: ConfidenceBand | None) -> dict | None:
    if band is None:
        return None
    return {
        "low_kwh": round(band.low_kwh, 1),
        "high_kwh": round(band.high_kwh, 1),
        "low_daytime_fraction": round(band.low_daytime_fraction, 4),
        "high_daytime_fraction": round(band.high_daytime_fraction, 4),
        "spread_pct": round(band.spread_pct, 2),
    }


def sizing_to_dict(outcome: SizingOutcome, include_hourly: bool = False) -> dict:
    """Serialise a :class:`SizingOutcome` into the public JSON shape."""
    sizing = outcome.sizing
    dispatch = sizing.result

    payload = outcome_to_dict(outcome.estimate, include_hourly=include_hourly)
    payload["consumption"] = {
        "annual_kwh": round(dispatch.load_kwh, 1),
        "source": sizing.load_source,
        "measured": sizing.measured_load,
        "daytime_fraction": round(sizing.daytime_fraction, 4),
    }
    payload["battery"] = {
        "usable_capacity_kwh": sizing.battery.usable_capacity_kwh,
        "max_charge_kw": round(sizing.battery.charge_limit_kw, 2),
        "max_discharge_kw": round(sizing.battery.discharge_limit_kw, 2),
        "round_trip_efficiency": sizing.battery.round_trip_efficiency,
    }
    payload["balance"] = {
        "generation_kwh": round(dispatch.generation_kwh, 1),
        "load_kwh": round(dispatch.load_kwh, 1),
        "avoided_import_kwh": round(dispatch.avoided_import_kwh, 1),
        "import_kwh": round(dispatch.total_import_kwh, 1),
        "export_kwh": round(dispatch.total_export_kwh, 1),
        "direct_use_kwh": round(float(dispatch.direct_kwh.sum()), 1),
        "from_battery_kwh": round(float(dispatch.discharge_kwh.sum()), 1),
        "storage_loss_kwh": round(dispatch.storage_loss_kwh, 1),
        "self_sufficiency_pct": round(dispatch.self_sufficiency_pct, 1),
        "self_consumption_pct": round(dispatch.self_consumption_pct, 1),
        "equivalent_full_cycles": round(dispatch.equivalent_full_cycles, 1),
        "battery_contribution_kwh": round(sizing.battery_contribution_kwh, 1),
        "pv_only_avoided_import_kwh": round(sizing.pv_only.avoided_import_kwh, 1),
        "pv_only_self_sufficiency_pct": round(sizing.pv_only.self_sufficiency_pct, 1),
    }
    payload["confidence"] = {
        "band": _band_to_dict(sizing.band),
        "band_pv_only": _band_to_dict(sizing.band_pv_only),
        "note": _confidence_note(sizing),
    }
    payload["sizing_curve"] = [
        {
            "capacity_kwh": entry.capacity_kwh,
            "avoided_import_kwh": round(entry.avoided_import_kwh, 1),
            "self_sufficiency_pct": round(entry.self_sufficiency_pct, 1),
            "self_consumption_pct": round(entry.self_consumption_pct, 1),
            "export_kwh": round(entry.export_kwh, 1),
            "equivalent_full_cycles": round(entry.equivalent_full_cycles, 1),
            "marginal_kwh_per_kwh": round(entry.marginal_kwh_per_kwh, 1),
        }
        for entry in sizing.curve
    ]
    payload["suggested_capacity_kwh"] = sizing.suggested_capacity_kwh
    payload["monthly_balance"] = [
        {
            "month": m.month,
            "generation_kwh": round(m.generation_kwh, 1),
            "load_kwh": round(m.load_kwh, 1),
            "direct_kwh": round(m.direct_kwh, 1),
            "from_battery_kwh": round(m.discharge_kwh, 1),
            "import_kwh": round(m.import_kwh, 1),
            "export_kwh": round(m.export_kwh, 1),
            "self_sufficiency_pct": round(m.self_sufficiency_pct, 1),
        }
        for m in sizing.monthly
    ]
    return payload


def _confidence_note(sizing: SizingResult) -> str:
    """Say plainly what the numbers are worth."""
    if sizing.measured_load:
        return (
            "Demand came from metered interval data, so the load shape is known "
            "rather than assumed. The remaining uncertainty is weather variability "
            "and the accuracy of the generation model."
        )
    if sizing.band is None or sizing.band_pv_only is None:
        return "Demand was synthesised from billing data."
    return (
        "Demand was synthesised from billing data, so when the household uses "
        "electricity is an assumption, not a measurement. Varying that "
        f"assumption moves the battery result by {sizing.band.spread_pct:.0f}% "
        f"but the no-battery result by {sizing.band_pv_only.spread_pct:.0f}% -- "
        "a battery absorbs the mismatch it governs. Upload half-hourly meter "
        "data to remove the assumption entirely."
    )


def household_options() -> list[dict]:
    """Describe the load archetypes, for the UI to offer."""
    from .load import DAYTIME_WINDOW, archetype_daytime_fraction

    labels = {
        "nine_to_five": "Out at work during the day",
        "home_all_day": "Someone home all day",
        "working_from_home": "Working from home",
        "flat": "Flat (diagnostic baseline)",
    }
    start, end = DAYTIME_WINDOW
    return [
        {
            "value": name,
            "label": labels.get(name, name.replace("_", " ").title()),
            "daytime_fraction": round(archetype_daytime_fraction(name), 4),
            "daytime_window": f"{start:02d}:00-{end:02d}:00",
        }
        for name in ARCHETYPES
    ]


def default_loss_breakdown() -> dict[str, float]:
    """The default itemised loss stack, as percentages, plus its combined total."""
    items = {name: round(value * 100, 2) for name, value in DEFAULT_LOSSES.items()}
    items["combined_total"] = round(combine_losses(DEFAULT_LOSSES) * 100, 2)
    return items
