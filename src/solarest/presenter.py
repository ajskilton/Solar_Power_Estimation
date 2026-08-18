"""Turning model results into the plain dicts every front end consumes.

Split out of :mod:`solarest.service` so that it depends on nothing but numpy
and the model layer. The HTTP service validates requests with pydantic and the
CLI parses argv, but both end up here, and so does the browser build -- where
pydantic and FastAPI are not available and the whole point is to ship the
smallest possible runtime.

This module also owns the defaults: how a partial description of a system
becomes a :class:`~solarest.pvmodel.SystemSpec`, and how a description of a
household becomes a :class:`~solarest.load.LoadProfile`. Keeping that in one
place is what stops the website, the CLI and the browser build from quietly
disagreeing about what "a 4 kWp array" means.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .battery import BatterySpec
from .load import ARCHETYPES, HouseholdShape, LoadProfile, synthesise
from .optimize import OrientationResult, equator_facing_azimuth, rule_of_thumb_tilt
from .pvmodel import DEFAULT_LOSSES, MODULE_TYPES, MOUNTS, SystemSpec, combine_losses
from .results import EstimateResult, result_to_dict
from .sizing import ConfidenceBand, SizingResult
from .weather import SiteInfo, WeatherSeries


# ------------------------------------------------------------- the defaults


def build_system_spec(
    latitude: float,
    *,
    dc_capacity_kw: float = 4.0,
    tilt_deg: float | None = None,
    azimuth_deg: float | None = None,
    mount: str = "roof_mount",
    module_type: str = "premium",
    dc_ac_ratio: float = 1.15,
    inverter_efficiency: float = 0.96,
    albedo: float = 0.20,
    system_losses_pct: float | None = None,
    shading_pct: float | None = None,
    transposition_model: str = "perez",
) -> SystemSpec:
    """Resolve a partial system description against the site.

    Tilt and azimuth default to the latitude rule of thumb and to facing the
    equator. Shading may be overridden on its own; an explicit total replaces
    the itemised PVWatts stack entirely.

    Args:
        latitude: Site latitude, used only to resolve the two defaults.
        dc_capacity_kw: Nameplate DC capacity at STC, kW.
        tilt_deg: Array tilt, or ``None`` for the rule of thumb.
        azimuth_deg: Array azimuth, or ``None`` for equator-facing.
        mount: Key into :data:`~solarest.pvmodel.MOUNTS`.
        module_type: Key into :data:`~solarest.pvmodel.MODULE_TYPES`.
        dc_ac_ratio: DC nameplate over inverter AC rating.
        inverter_efficiency: Inverter nominal efficiency, 0-1.
        albedo: Ground reflectance, 0-1.
        system_losses_pct: Combined DC-side losses, replacing the stack.
        shading_pct: Shading loss alone, as a percentage.
        transposition_model: ``"perez"`` or ``"hdkr"``.

    Returns:
        A validated :class:`~solarest.pvmodel.SystemSpec`.
    """
    losses = dict(DEFAULT_LOSSES)
    if shading_pct is not None:
        losses["shading"] = shading_pct / 100.0
    if system_losses_pct is not None:
        losses = {"system": system_losses_pct / 100.0}

    return SystemSpec(
        dc_capacity_kw=dc_capacity_kw,
        tilt_deg=rule_of_thumb_tilt(latitude) if tilt_deg is None else tilt_deg,
        azimuth_deg=(
            equator_facing_azimuth(latitude) if azimuth_deg is None else azimuth_deg
        ),
        mount=mount,
        module_type=module_type,
        dc_ac_ratio=dc_ac_ratio,
        inverter_efficiency=inverter_efficiency,
        albedo=albedo,
        losses=losses,
        transposition_model=transposition_model,
    )


def build_battery_spec(
    *,
    usable_capacity_kwh: float = 5.0,
    max_charge_kw: float | None = None,
    max_discharge_kw: float | None = None,
    round_trip_efficiency: float = 0.90,
    initial_soc_frac: float = 0.0,
) -> BatterySpec:
    """Build a :class:`~solarest.battery.BatterySpec` from loose keywords."""
    return BatterySpec(
        usable_capacity_kwh=usable_capacity_kwh,
        max_charge_kw=max_charge_kw,
        max_discharge_kw=max_discharge_kw,
        round_trip_efficiency=round_trip_efficiency,
        initial_soc_frac=initial_soc_frac,
    )


def build_load_profile(
    latitude: float,
    *,
    annual_kwh: float | None = None,
    monthly_kwh: Sequence[float] | None = None,
    hourly_kwh: Sequence[float] | None = None,
    household: HouseholdShape | None = None,
) -> LoadProfile:
    """Build the demand series from whichever of the three inputs was given.

    Args:
        latitude: Used only to orient the seasonal curve by hemisphere.
        annual_kwh: One figure for the year.
        monthly_kwh: Twelve totals, January first.
        hourly_kwh: Metered interval data at any regular resolution.
        household: When electricity is used; ignored for metered data, which
            already says so.

    Returns:
        A :class:`~solarest.load.LoadProfile`.

    Raises:
        ValueError: Unless exactly one of the three inputs is given.
    """
    given = [
        name
        for name, value in (
            ("annual_kwh", annual_kwh),
            ("monthly_kwh", monthly_kwh),
            ("hourly_kwh", hourly_kwh),
        )
        if value is not None
    ]
    if len(given) != 1:
        raise ValueError(
            "give exactly one of annual_kwh, monthly_kwh or hourly_kwh, "
            f"got {given or 'none'}"
        )

    if hourly_kwh is not None:
        return LoadProfile.from_hourly(hourly_kwh)
    return synthesise(
        annual_kwh=annual_kwh,
        monthly_kwh=monthly_kwh,
        household=household or HouseholdShape(),
        latitude=latitude,
    )


@dataclass(frozen=True)
class EstimateOutcome:
    """Everything the estimate use case produces."""

    site: SiteInfo
    weather: WeatherSeries
    spec: SystemSpec
    result: EstimateResult
    orientation: OrientationResult | None




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




@dataclass(frozen=True)
class SizingOutcome:
    """An estimate and the sizing built on top of it."""

    estimate: EstimateOutcome
    sizing: SizingResult




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
    payload["diurnal_balance"] = {
        "by_month": {
            name: [[round(float(v), 4) for v in row] for row in grid]
            for name, grid in sizing.diurnal.by_month.items()
        },
        "by_year": {
            name: [round(float(v), 4) for v in values]
            for name, values in sizing.diurnal.by_year.items()
        },
    }
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
