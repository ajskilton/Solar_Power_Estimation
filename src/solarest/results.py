"""Turn hourly simulation output into the statistics a user actually wants.

Three views of the same simulation:

* **Annual** -- expected yield with its year-to-year spread, so a household
  sees the range rather than one number pretending to be certain.
* **Monthly and diurnal** -- when the energy arrives, across the year and
  across the day.
* **Typical year** -- an 8760-hour reference series. This is built by the
  Sandia TMY method (pick, for each calendar month, the real month from the
  record that best matches long-run conditions) rather than by averaging
  years together. Averaging would smooth away cloudy runs and quietly make a
  battery look better than it is; keeping real months preserves the
  day-to-day variability that storage sizing depends on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .simulate import Simulation

# A non-leap placeholder year for typical-year timestamps. Real TMY files
# splice months from different years; a single stable year keeps the series a
# clean 8760 and is easier for downstream consumers to align against.
REFERENCE_YEAR = 2023

HOURS_PER_YEAR = 8760

# Weights for the Finkelstein-Schafer statistic. Irradiance dominates PV
# output; temperature matters for module derate and for load correlation.
_FS_WEIGHTS = {"ghi": 0.7, "temp": 0.3}


@dataclass(frozen=True)
class YearResult:
    """One calendar year's totals."""

    year: int
    energy_kwh: float
    poa_kwh_m2: float
    ghi_kwh_m2: float
    mean_temp_c: float
    hours: int
    complete: bool


@dataclass(frozen=True)
class AnnualSummary:
    """Expected annual performance and its year-to-year spread.

    Attributes:
        energy_kwh_mean: Mean annual AC energy across complete years.
        energy_kwh_p10: Tenth-percentile year (a poor year).
        energy_kwh_p50: Median year.
        energy_kwh_p90: Ninetieth-percentile year (a good year).
        energy_kwh_min: Worst year in the record.
        energy_kwh_max: Best year in the record.
        variability_pct: Standard deviation as a percentage of the mean.
        specific_yield_kwh_per_kwp: Annual energy per kW of installed DC.
        performance_ratio: AC energy over the energy an ideal, loss-free array
            of the same rating would make from the same plane irradiance.
        capacity_factor: Mean output as a fraction of nameplate DC.
        poa_kwh_m2: Annual irradiation on the array plane.
        ghi_kwh_m2: Annual irradiation on the horizontal, for comparison.
        clipping_loss_kwh: Annual energy lost to the inverter AC ceiling.
        peak_ac_kw: Highest hourly AC output seen in the record.
        full_load_hours: Equivalent hours per year at nameplate output.
        years_used: Calendar years the statistics are based on.
    """

    energy_kwh_mean: float
    energy_kwh_p10: float
    energy_kwh_p50: float
    energy_kwh_p90: float
    energy_kwh_min: float
    energy_kwh_max: float
    variability_pct: float
    specific_yield_kwh_per_kwp: float
    performance_ratio: float
    capacity_factor: float
    poa_kwh_m2: float
    ghi_kwh_m2: float
    clipping_loss_kwh: float
    peak_ac_kw: float
    full_load_hours: float
    years_used: list[int]
    per_year: list[YearResult] = field(default_factory=list)


@dataclass(frozen=True)
class MonthResult:
    """One calendar month, averaged over the years in the record."""

    month: int
    energy_kwh: float
    energy_kwh_min: float
    energy_kwh_max: float
    daily_mean_kwh: float
    poa_kwh_m2: float
    mean_temp_c: float
    mean_cell_temp_c: float
    peak_ac_kw: float


@dataclass(frozen=True)
class TypicalYear:
    """An 8760-hour reference series assembled from real months.

    Attributes:
        reference_year: Placeholder year stamped on the timestamps.
        month_sources: Which real year each calendar month was taken from.
        ac_kw: Hourly AC power, kW (numerically also kWh per hour).
        poa_w_m2: Hourly plane-of-array irradiance, W/m^2.
        air_temp_c: Hourly ambient temperature, degC.
        annual_kwh: Total AC energy of this particular year.
    """

    reference_year: int
    month_sources: dict[int, int]
    ac_kw: np.ndarray
    poa_w_m2: np.ndarray
    air_temp_c: np.ndarray
    annual_kwh: float


@dataclass(frozen=True)
class LossBreakdown:
    """Where the plane-of-array energy goes, as a fraction of POA energy."""

    poa_kwh_m2: float
    reflection_pct: float
    temperature_pct: float
    system_losses_pct: float
    inverter_pct: float
    clipping_pct: float
    delivered_pct: float


@dataclass(frozen=True)
class EstimateResult:
    """The complete answer for one site and system."""

    annual: AnnualSummary
    monthly: list[MonthResult]
    diurnal: list[list[float]]
    typical_year: TypicalYear
    losses: LossBreakdown


def _year_of(times: np.ndarray) -> np.ndarray:
    return times.astype("datetime64[Y]").astype(int) + 1970


def _month_of(times: np.ndarray) -> np.ndarray:
    return times.astype("datetime64[M]").astype(int) % 12 + 1


def _day_of(times: np.ndarray) -> np.ndarray:
    return (times.astype("datetime64[D]") - times.astype("datetime64[M]")).astype(int) + 1


def _hour_of(times: np.ndarray) -> np.ndarray:
    return (times.astype("datetime64[h]") - times.astype("datetime64[D]")).astype(int)


def summarise(sim: Simulation) -> EstimateResult:
    """Aggregate an hourly simulation into annual, monthly and typical-year views.

    Args:
        sim: A completed :class:`~solarest.simulate.Simulation`.

    Returns:
        The full :class:`EstimateResult`.

    Raises:
        ValueError: If the simulation contains no complete calendar year.
    """
    times = sim.local_times
    years = _year_of(times)
    months = _month_of(times)

    energy = sim.chain.ac_kw  # kWh per hourly step
    poa_kwh = sim.chain.poa.global_ / 1000.0
    ghi_kwh = sim.weather.ghi / 1000.0

    per_year: list[YearResult] = []
    for year in sorted(set(int(y) for y in years)):
        mask = years == year
        hours = int(mask.sum())
        expected = 8784 if _is_leap(year) else HOURS_PER_YEAR
        per_year.append(
            YearResult(
                year=year,
                energy_kwh=float(energy[mask].sum()),
                poa_kwh_m2=float(poa_kwh[mask].sum()),
                ghi_kwh_m2=float(ghi_kwh[mask].sum()),
                mean_temp_c=float(sim.weather.temp_c[mask].mean()),
                hours=hours,
                # Local-standard-time shifting clips a few hours off each end
                # of the record; allow a day of slack before calling a year
                # incomplete.
                complete=hours >= expected - 24,
            )
        )

    complete = [y for y in per_year if y.complete]
    if not complete:
        raise ValueError(
            "the weather record does not contain a complete calendar year; "
            "request at least one full year"
        )

    annual_energy = np.array([y.energy_kwh for y in complete])
    annual_poa = np.array([y.poa_kwh_m2 for y in complete])
    annual_ghi = np.array([y.ghi_kwh_m2 for y in complete])

    mean_energy = float(annual_energy.mean())
    mean_poa = float(annual_poa.mean())
    capacity = sim.system.dc_capacity_kw

    complete_years = {y.year for y in complete}
    in_complete = np.isin(years, list(complete_years))
    n_complete = len(complete)

    annual = AnnualSummary(
        energy_kwh_mean=mean_energy,
        energy_kwh_p10=float(np.percentile(annual_energy, 10)),
        energy_kwh_p50=float(np.percentile(annual_energy, 50)),
        energy_kwh_p90=float(np.percentile(annual_energy, 90)),
        energy_kwh_min=float(annual_energy.min()),
        energy_kwh_max=float(annual_energy.max()),
        variability_pct=(
            float(annual_energy.std(ddof=1) / mean_energy * 100.0)
            if n_complete > 1 and mean_energy > 0
            else 0.0
        ),
        specific_yield_kwh_per_kwp=mean_energy / capacity,
        # PR compares delivered AC against a perfect array of the same rating
        # under the same plane irradiance.
        performance_ratio=(mean_energy / (mean_poa * capacity)) if mean_poa > 0 else 0.0,
        capacity_factor=mean_energy / (capacity * HOURS_PER_YEAR),
        poa_kwh_m2=mean_poa,
        ghi_kwh_m2=float(annual_ghi.mean()),
        clipping_loss_kwh=float(sim.chain.clipped_kw[in_complete].sum()) / n_complete,
        peak_ac_kw=float(sim.chain.ac_kw.max()),
        full_load_hours=mean_energy / capacity,
        years_used=sorted(complete_years),
        per_year=per_year,
    )

    monthly = _monthly(sim, years, months, in_complete, n_complete)
    diurnal = _diurnal(sim, months, _hour_of(times), in_complete)
    typical = build_typical_year(sim)
    losses = _losses(sim, in_complete, n_complete)

    return EstimateResult(
        annual=annual,
        monthly=monthly,
        diurnal=diurnal,
        typical_year=typical,
        losses=losses,
    )


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _monthly(
    sim: Simulation,
    years: np.ndarray,
    months: np.ndarray,
    in_complete: np.ndarray,
    n_complete: int,
) -> list[MonthResult]:
    """Per-calendar-month means over the complete years in the record."""
    days_in_month = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # Bucket every hour by (year, month) once, then reduce -- a mask per month
    # per year would rescan the whole record dozens of times.
    keep = np.flatnonzero(in_complete)
    year_values = np.unique(years[keep])
    year_slot = np.searchsorted(year_values, years[keep])
    bucket = year_slot * 12 + (months[keep] - 1)
    n_buckets = year_values.size * 12

    def bucket_sum(values: np.ndarray) -> np.ndarray:
        return np.bincount(bucket, weights=values[keep], minlength=n_buckets).reshape(
            year_values.size, 12
        )

    counts = np.bincount(bucket, minlength=n_buckets).reshape(year_values.size, 12)
    energy_by = bucket_sum(sim.chain.ac_kw)
    poa_by = bucket_sum(sim.chain.poa.global_) / 1000.0
    temp_by = bucket_sum(sim.weather.temp_c)
    cell_by = bucket_sum(sim.chain.cell_temp_c)

    peak_by = np.zeros(n_buckets)
    np.maximum.at(peak_by, bucket, sim.chain.ac_kw[keep])

    results: list[MonthResult] = []
    for month in range(1, 13):
        column = counts[:, month - 1]
        if not column.any():
            results.append(MonthResult(month, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        present = column > 0
        totals = energy_by[present, month - 1]
        hours = column[present].sum()
        mean_energy = float(totals.mean())
        results.append(
            MonthResult(
                month=month,
                energy_kwh=mean_energy,
                energy_kwh_min=float(totals.min()),
                energy_kwh_max=float(totals.max()),
                daily_mean_kwh=mean_energy / days_in_month[month - 1],
                poa_kwh_m2=float(poa_by[:, month - 1].sum()) / n_complete,
                mean_temp_c=float(temp_by[:, month - 1].sum() / hours),
                mean_cell_temp_c=float(cell_by[:, month - 1].sum() / hours),
                peak_ac_kw=float(peak_by.reshape(year_values.size, 12)[:, month - 1].max()),
            )
        )
    return results


def _diurnal(
    sim: Simulation,
    months: np.ndarray,
    hours: np.ndarray,
    in_complete: np.ndarray,
) -> list[list[float]]:
    """Mean AC power by month and hour of local standard day, a 12x24 grid."""
    keep = np.flatnonzero(in_complete)
    bucket = (months[keep] - 1) * 24 + hours[keep]
    totals = np.bincount(bucket, weights=sim.chain.ac_kw[keep], minlength=12 * 24)
    counts = np.bincount(bucket, minlength=12 * 24)
    mean = np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)
    return [[float(v) for v in row] for row in mean.reshape(12, 24)]


def _losses(sim: Simulation, in_complete: np.ndarray, n_complete: int) -> LossBreakdown:
    """Attribute the gap between plane irradiance and delivered AC energy.

    Each stage is expressed as a percentage of the plane-of-array energy that
    landed on the array, so the stages plus the delivered share add to 100%.
    """
    capacity = sim.system.dc_capacity_kw

    def annual_mean(values: np.ndarray) -> float:
        return float(values[in_complete].sum()) / n_complete

    # Energy an ideal array of this rating would make from the POA that hit it.
    ideal = annual_mean(sim.chain.poa.global_) / 1000.0 * capacity
    if ideal <= 0:
        return LossBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    after_reflection = annual_mean(sim.chain.poa.effective) / 1000.0 * capacity

    # DC before system losses, i.e. after the temperature derate only.
    dc_after_temp = annual_mean(sim.chain.dc_kw) / (1.0 - sim.system.total_loss_fraction)
    dc_final = annual_mean(sim.chain.dc_kw)
    ac = annual_mean(sim.chain.ac_kw)
    clipped = annual_mean(sim.chain.clipped_kw)

    pct = lambda value: float(value / ideal * 100.0)  # noqa: E731
    return LossBreakdown(
        poa_kwh_m2=annual_mean(sim.chain.poa.global_) / 1000.0,
        reflection_pct=pct(ideal - after_reflection),
        temperature_pct=pct(after_reflection - dc_after_temp),
        system_losses_pct=pct(dc_after_temp - dc_final),
        inverter_pct=pct(dc_final - ac - clipped),
        clipping_pct=pct(clipped),
        delivered_pct=pct(ac),
    )


def build_typical_year(sim: Simulation, reference_year: int = REFERENCE_YEAR) -> TypicalYear:
    """Assemble an 8760-hour typical year by the Sandia TMY method.

    For each calendar month, the year whose daily distributions of irradiance
    and temperature best match the long-run distribution is selected via the
    Finkelstein-Schafer statistic, and that real month is spliced in whole.

    Args:
        sim: A completed simulation spanning one or more whole years.
        reference_year: Non-leap year stamped on the output timestamps.

    Returns:
        A :class:`TypicalYear`.
    """
    if _is_leap(reference_year):
        raise ValueError("reference_year must not be a leap year")

    times = sim.local_times
    years = _year_of(times)
    months = _month_of(times)
    days = _day_of(times)
    hours = _hour_of(times)

    daily_key = years.astype("int64") * 10000 + months * 100 + days
    unique_days, day_index = np.unique(daily_key, return_inverse=True)
    daily_ghi = np.bincount(day_index, weights=sim.weather.ghi, minlength=unique_days.size)
    daily_temp = np.bincount(
        day_index, weights=sim.weather.temp_c, minlength=unique_days.size
    ) / np.bincount(day_index, minlength=unique_days.size)
    day_year = (unique_days // 10000).astype(int)
    day_month = ((unique_days // 100) % 100).astype(int)

    # Only years with a usable amount of data can supply a month.
    present, counts = np.unique(years, return_counts=True)
    candidate_years = [int(y) for y, n in zip(present, counts) if n > 24 * 300]

    ac_out = np.zeros(HOURS_PER_YEAR)
    poa_out = np.zeros(HOURS_PER_YEAR)
    temp_out = np.zeros(HOURS_PER_YEAR)
    sources: dict[int, int] = {}

    cursor = 0
    for month in range(1, 13):
        length = _days_in_month(reference_year, month) * 24
        chosen = _select_tmy_month(
            month, candidate_years, day_month, day_year, daily_ghi, daily_temp
        )
        sources[month] = chosen

        mask = (years == chosen) & (months == month)
        # Drop 29 February so every assembled year is exactly 8760 hours.
        if month == 2:
            mask &= days <= 28

        order = np.lexsort((hours[mask], days[mask]))
        slot = slice(cursor, cursor + length)
        ac_out[slot] = _fit(sim.chain.ac_kw[mask][order], length)
        poa_out[slot] = _fit(sim.chain.poa.global_[mask][order], length)
        temp_out[slot] = _fit(sim.weather.temp_c[mask][order], length)
        cursor += length

    return TypicalYear(
        reference_year=reference_year,
        month_sources=sources,
        ac_kw=ac_out,
        poa_w_m2=poa_out,
        air_temp_c=temp_out,
        annual_kwh=float(ac_out.sum()),
    )


def _days_in_month(year: int, month: int) -> int:
    start = np.datetime64(f"{year}-{month:02d}-01", "D")
    end = (
        np.datetime64(f"{year + 1}-01-01", "D")
        if month == 12
        else np.datetime64(f"{year}-{month + 1:02d}-01", "D")
    )
    return int((end - start).astype(int))


def _fit(values: np.ndarray, length: int) -> np.ndarray:
    """Pad or trim a month's hours to the exact slot length.

    Shifting UTC into local standard time can leave a month a few hours short
    at one end of the record; padding with the edge value is a harmless fix at
    that scale and keeps the typical year exactly 8760 hours.
    """
    if values.size == length:
        return values
    if values.size > length:
        return values[:length]
    if values.size == 0:
        return np.zeros(length)
    return np.concatenate([values, np.full(length - values.size, values[-1])])


def _select_tmy_month(
    month: int,
    candidate_years: list[int],
    day_month: np.ndarray,
    day_year: np.ndarray,
    daily_ghi: np.ndarray,
    daily_temp: np.ndarray,
) -> int:
    """Pick the year supplying this calendar month, lowest weighted FS wins."""
    month_mask = day_month == month
    if not month_mask.any() or not candidate_years:
        return candidate_years[0] if candidate_years else int(day_year[0])

    long_run = {"ghi": daily_ghi[month_mask], "temp": daily_temp[month_mask]}

    best_year = candidate_years[0]
    best_score = np.inf
    for year in candidate_years:
        year_mask = month_mask & (day_year == year)
        if year_mask.sum() < 20:  # too little data to represent the month
            continue
        score = sum(
            weight
            * _finkelstein_schafer(
                {"ghi": daily_ghi, "temp": daily_temp}[key][year_mask], long_run[key]
            )
            for key, weight in _FS_WEIGHTS.items()
        )
        if score < best_score:
            best_score, best_year = score, year
    return best_year


def _finkelstein_schafer(candidate: np.ndarray, long_run: np.ndarray) -> float:
    """Mean absolute difference between two empirical CDFs.

    Args:
        candidate: Daily values for one candidate month.
        long_run: Daily values for that calendar month across all years.

    Returns:
        The FS statistic; zero means the candidate matches the long-run
        distribution exactly.
    """
    if candidate.size == 0 or long_run.size == 0:
        return float("inf")
    ordered = np.sort(candidate)
    # CDF of the candidate at its own sorted values, mid-rank convention.
    candidate_cdf = (np.arange(ordered.size) + 0.5) / ordered.size
    long_run_cdf = np.searchsorted(np.sort(long_run), ordered, side="right") / long_run.size
    return float(np.mean(np.abs(candidate_cdf - long_run_cdf)))


def result_to_dict(result: EstimateResult, hourly: bool = True) -> dict:
    """Serialise an :class:`EstimateResult` to JSON-friendly primitives.

    Args:
        result: The result to serialise.
        hourly: Include the 8760-hour typical-year series. Turning this off
            keeps the payload small when only headline figures are needed.

    Returns:
        A nested dict of plain Python types, with floats rounded for transport.
    """
    typical: dict = {
        "reference_year": result.typical_year.reference_year,
        "month_sources": {str(k): v for k, v in result.typical_year.month_sources.items()},
        "annual_kwh": round(result.typical_year.annual_kwh, 1),
    }
    if hourly:
        typical["ac_kw"] = [round(float(v), 4) for v in result.typical_year.ac_kw]
        typical["poa_w_m2"] = [round(float(v), 1) for v in result.typical_year.poa_w_m2]
        typical["air_temp_c"] = [round(float(v), 2) for v in result.typical_year.air_temp_c]

    return {
        "annual": _round_floats(asdict(result.annual), 2),
        "monthly": [_round_floats(asdict(m), 2) for m in result.monthly],
        "diurnal": [[round(v, 4) for v in row] for row in result.diurnal],
        "typical_year": typical,
        "losses": _round_floats(asdict(result.losses), 2),
    }


def _round_floats(obj, places: int):
    if isinstance(obj, dict):
        return {k: _round_floats(v, places) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, places) for v in obj]
    if isinstance(obj, float):
        return round(obj, places)
    return obj
