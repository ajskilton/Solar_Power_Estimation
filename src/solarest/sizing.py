"""Battery sizing: the estimate, and how much to trust it.

This is where the typical year meets a household. The arithmetic is the easy
part -- :mod:`solarest.battery` does it in a page. The judgement is in knowing
what the answer is worth, and that depends entirely on where the load profile
came from:

* **Metered interval data** gives a single number worth quoting. The remaining
  error is weather variability, not the load.
* **Monthly bills** give a number that depends on an assumption about when the
  household uses electricity. Quoting one figure would be false precision, so
  this module re-runs the dispatch across a plausible band of daytime
  fractions and reports the range alongside the central case.

The two paths differ only in that treatment; both run identical physics.

The band is reported twice, with and without the battery, because the
difference between them is the most useful thing this module has to say. A
battery absorbs the mismatch between generation and demand, which is precisely
what the load-shape assumption governs -- so it also absorbs the error in that
assumption. Sizing a battery from monthly bills is defensible; estimating
no-battery self-consumption from the same data is not nearly as sound, and the
two bands make that visible instead of leaving the reader to assume both
numbers are equally firm.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .battery import BatterySpec, DispatchResult, SizeResult, dispatch, sweep_sizes, suggest_capacity
from .load import LoadProfile, daytime_fraction_band, synthesise
from .results import HOURS_PER_YEAR, REFERENCE_YEAR, TypicalYear

# Capacities swept by default, kWh usable. Dense at the bottom because that is
# where the curve bends; a domestic pack above about 20 kWh is unusual.
DEFAULT_SWEEP_KWH: tuple[float, ...] = (
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.5, 8.0, 10.0, 12.0, 15.0, 20.0,
)


@dataclass(frozen=True)
class MonthlyBalance:
    """One calendar month of the energy balance, kWh."""

    month: int
    generation_kwh: float
    load_kwh: float
    direct_kwh: float
    discharge_kwh: float
    import_kwh: float
    export_kwh: float

    @property
    def avoided_import_kwh(self) -> float:
        """Grid energy displaced in the month."""
        return self.direct_kwh + self.discharge_kwh

    @property
    def self_sufficiency_pct(self) -> float:
        """Share of the month's demand met on site."""
        if self.load_kwh <= 0:
            return 0.0
        return 100.0 * self.avoided_import_kwh / self.load_kwh


@dataclass(frozen=True)
class ConfidenceBand:
    """How far the headline moves under alternative load-shape assumptions.

    Attributes:
        low_kwh: Avoided import with the household using less during the day.
        high_kwh: Avoided import with the household using more during the day.
        low_daytime_fraction: The pessimistic assumption behind ``low_kwh``.
        high_daytime_fraction: The optimistic assumption behind ``high_kwh``.
    """

    low_kwh: float
    high_kwh: float
    low_daytime_fraction: float
    high_daytime_fraction: float

    @property
    def spread_pct(self) -> float:
        """Width of the band as a percentage of its midpoint."""
        midpoint = 0.5 * (self.low_kwh + self.high_kwh)
        if midpoint <= 0:
            return 0.0
        return 100.0 * (self.high_kwh - self.low_kwh) / midpoint


@dataclass(frozen=True)
class SizingResult:
    """Everything the sizing question produces for one household.

    Attributes:
        battery: The battery modelled for the headline figures.
        result: The full hourly dispatch.
        monthly: The balance month by month.
        curve: Avoided import against battery capacity.
        suggested_capacity_kwh: Where the curve stops paying its way.
        band: Confidence band on the headline, present only for synthesised
            profiles.
        band_pv_only: The same band computed without the battery. Compare the
            two: the battery is what makes a bills-based estimate trustworthy.
        load_source: Provenance of the demand series.
        measured_load: Whether the demand series was metered.
        daytime_fraction: Realised daytime share of the demand series.
        pv_only: The same year with no battery, for comparison.
    """

    battery: BatterySpec
    result: DispatchResult
    monthly: list[MonthlyBalance]
    curve: list[SizeResult]
    suggested_capacity_kwh: float
    band: ConfidenceBand | None
    band_pv_only: ConfidenceBand | None
    load_source: str
    measured_load: bool
    daytime_fraction: float
    pv_only: DispatchResult

    @property
    def battery_contribution_kwh(self) -> float:
        """Extra grid energy displaced by the battery, over PV alone."""
        return self.result.avoided_import_kwh - self.pv_only.avoided_import_kwh


def size_for_household(
    typical_year: TypicalYear,
    load: LoadProfile,
    battery: BatterySpec | None = None,
    *,
    sweep_kwh: Sequence[float] | None = None,
    band_spread: float = 0.07,
    latitude: float = 51.5,
) -> SizingResult:
    """Estimate avoided grid import for one household and battery.

    Args:
        typical_year: The 8760-hour generation series from
            :func:`solarest.results.build_typical_year`.
        load: Demand on the same 8760-hour grid.
        battery: The battery to headline. Defaults to a 5 kWh pack.
        sweep_kwh: Capacities for the sizing curve. Defaults to
            :data:`DEFAULT_SWEEP_KWH`.
        band_spread: Half-width of the daytime-fraction band explored for
            synthesised profiles, in absolute fraction.
        latitude: Passed through when regenerating band variants, so the
            seasonal curve stays oriented to the right hemisphere.

    Returns:
        A :class:`SizingResult`.

    Raises:
        ValueError: If the two series are not both a full typical year.
    """
    generation = np.asarray(typical_year.ac_kw, dtype=float)
    if generation.size != HOURS_PER_YEAR or load.kwh.size != HOURS_PER_YEAR:
        raise ValueError(
            f"generation and load must both be {HOURS_PER_YEAR} hours, "
            f"got {generation.size} and {load.kwh.size}"
        )

    battery = battery or BatterySpec()
    outcome = dispatch(generation, load.kwh, battery)
    pv_only = dispatch(generation, load.kwh, BatterySpec(usable_capacity_kwh=0.0))

    curve = sweep_sizes(
        generation, load.kwh, sweep_kwh or DEFAULT_SWEEP_KWH, template=battery
    )

    band = band_pv_only = None
    if not load.measured and load.household is not None:
        variants = _shape_variants(load, band_spread=band_spread, latitude=latitude)
        band = _confidence_band(generation, variants, battery)
        band_pv_only = _confidence_band(
            generation, variants, BatterySpec(usable_capacity_kwh=0.0)
        )

    return SizingResult(
        battery=battery,
        result=outcome,
        monthly=_monthly_balance(generation, load.kwh, outcome),
        curve=curve,
        suggested_capacity_kwh=suggest_capacity(curve),
        band=band,
        band_pv_only=band_pv_only,
        load_source=load.source,
        measured_load=load.measured,
        daytime_fraction=load.daytime_fraction,
        pv_only=pv_only,
    )


def _shape_variants(
    load: LoadProfile, *, band_spread: float, latitude: float
) -> tuple[LoadProfile, LoadProfile]:
    """Low- and high-daytime versions of a synthesised profile.

    Both are regenerated from the *same* monthly totals, so only the shape
    changes and annual consumption stays fixed. That isolates the effect of the
    assumption being tested, which is the whole point of the band.
    """
    assert load.household is not None  # guarded by the caller
    monthly = load.monthly_kwh()
    return tuple(  # type: ignore[return-value]
        synthesise(monthly_kwh=monthly, household=shape, latitude=latitude)
        for shape in daytime_fraction_band(load.household, band_spread)
    )


def _confidence_band(
    generation: np.ndarray,
    variants: tuple[LoadProfile, LoadProfile],
    battery: BatterySpec,
) -> ConfidenceBand:
    """Dispatch both shape variants and report the range they span."""
    low, high = variants
    return ConfidenceBand(
        low_kwh=dispatch(generation, low.kwh, battery).avoided_import_kwh,
        high_kwh=dispatch(generation, high.kwh, battery).avoided_import_kwh,
        low_daytime_fraction=low.daytime_fraction,
        high_daytime_fraction=high.daytime_fraction,
    )


def _monthly_balance(
    generation: np.ndarray, demand: np.ndarray, outcome: DispatchResult
) -> list[MonthlyBalance]:
    """Total the hourly balance into calendar months."""
    month = _month_index(REFERENCE_YEAR)

    def by_month(values: np.ndarray) -> np.ndarray:
        return np.bincount(month, weights=values, minlength=12)

    generation_by = by_month(generation)
    load_by = by_month(demand)
    direct_by = by_month(outcome.direct_kwh)
    discharge_by = by_month(outcome.discharge_kwh)
    import_by = by_month(outcome.import_kwh)
    export_by = by_month(outcome.export_kwh)

    return [
        MonthlyBalance(
            month=m + 1,
            generation_kwh=float(generation_by[m]),
            load_kwh=float(load_by[m]),
            direct_kwh=float(direct_by[m]),
            discharge_kwh=float(discharge_by[m]),
            import_kwh=float(import_by[m]),
            export_kwh=float(export_by[m]),
        )
        for m in range(12)
    ]


def _month_index(year: int) -> np.ndarray:
    """Zero-based calendar month of each hour of a 365-day year."""
    starts = np.array(
        [np.datetime64(f"{year}-{m:02d}-01", "D") for m in range(1, 13)]
        + [np.datetime64(f"{year + 1}-01-01", "D")]
    )
    return np.repeat(np.arange(12), np.diff(starts).astype(int) * 24)
