"""Household electricity demand, at the resolution a battery answer needs.

How much grid import a battery avoids is a question about *shape*, not volume.
Two households with identical monthly bills and identical generation can get
wildly different value from the same battery, purely because one is out during
the day and the other is not. Monthly totals average away exactly the
information the answer depends on.

So this module offers two ways in, and is explicit about what each is worth:

* :meth:`LoadProfile.from_hourly` -- real metered data. Half-hourly smart-meter
  exports, hourly interval data, whatever the supplier gives you. This is the
  best estimate; the residual error is then the weather, not the load.
* :func:`synthesise` -- monthly bills plus an assumption about *when* the
  household uses electricity. The assumption is a single number, the **daytime
  fraction**: the share of a day's consumption falling between 09:00 and 17:00.
  It defaults to a household that is out at work, and it is adjustable, because
  it is the one input that most changes the answer.

Everything is produced on the same 8760-hour grid as
:class:`solarest.results.TypicalYear` -- local standard time, no daylight
saving, every day 24 hours long -- so generation and demand line up hour for
hour with no resampling.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from .results import HOURS_PER_YEAR, REFERENCE_YEAR

# The window the daytime fraction refers to: 09:00 up to but not including
# 17:00. Chosen to match the working day rather than the solar day -- it is
# meant to be a number a householder can reason about, not a physical quantity.
DAYTIME_WINDOW: tuple[int, int] = (9, 17)

# Weekend days consume a little more than weekdays in most domestic records,
# because the house is occupied for more of them.
WEEKEND_UPLIFT = 1.08

# Normalised 24-hour demand shapes, midnight first. Relative weights only --
# they are normalised on load, so they need not sum to one. Shapes follow the
# familiar domestic pattern: an overnight base load of fridges and standby, a
# morning peak, and a dominant evening peak. What separates them is the middle
# of the day.
_BASE_SHAPES: dict[str, list[float]] = {
    # Out of the house 09:00-17:00. The daytime is little more than base load.
    "nine_to_five": [
        0.021, 0.019, 0.018, 0.017, 0.017, 0.019,
        0.029, 0.048, 0.048, 0.032, 0.026, 0.026,
        0.030, 0.026, 0.025, 0.026, 0.036, 0.060,
        0.080, 0.085, 0.078, 0.065, 0.048, 0.031,
    ],
    # Retired, shift work, small children -- someone is in all day.
    "home_all_day": [
        0.020, 0.018, 0.017, 0.016, 0.016, 0.018,
        0.026, 0.040, 0.045, 0.048, 0.050, 0.052,
        0.058, 0.050, 0.045, 0.045, 0.050, 0.062,
        0.075, 0.078, 0.070, 0.058, 0.043, 0.028,
    ],
    # Desk work at home: no commute dip, a lunchtime bump, evening still peaks.
    "working_from_home": [
        0.020, 0.018, 0.017, 0.016, 0.016, 0.018,
        0.027, 0.044, 0.046, 0.042, 0.042, 0.043,
        0.052, 0.044, 0.040, 0.040, 0.044, 0.060,
        0.078, 0.082, 0.074, 0.061, 0.045, 0.029,
    ],
    # A diagnostic baseline, not a real household.
    "flat": [1.0] * 24,
}

# Which shape a household falls back to at weekends. A nine-to-five household
# is at home on Saturday, so its weekend looks like somebody else's weekday.
_WEEKEND_DEFAULTS: dict[str, str] = {
    "nine_to_five": "home_all_day",
    "working_from_home": "home_all_day",
}

# Domestic electricity demand per day, relative to the annual mean, by calendar
# month for a northern-hemisphere temperate climate. Driven mostly by lighting
# hours and time spent indoors. Used only when the caller gives a single annual
# figure instead of twelve monthly ones.
_SEASONAL_DAILY_WEIGHTS = [
    1.24, 1.14, 1.05, 0.94, 0.85, 0.79,
    0.79, 0.82, 0.88, 1.00, 1.13, 1.25,
]

ARCHETYPES: tuple[str, ...] = tuple(_BASE_SHAPES)


def base_shape(archetype: str) -> np.ndarray:
    """The normalised 24-hour demand shape for an archetype, summing to 1.

    Args:
        archetype: One of :data:`ARCHETYPES`.

    Returns:
        Twenty-four fractions of a day's consumption, midnight first.

    Raises:
        ValueError: If the archetype is not recognised.
    """
    try:
        weights = _BASE_SHAPES[archetype]
    except KeyError:
        raise ValueError(
            f"unknown archetype {archetype!r}; expected one of {list(ARCHETYPES)}"
        ) from None
    shape = np.asarray(weights, dtype=float)
    return shape / shape.sum()


def daytime_fraction_of(shape: np.ndarray) -> float:
    """The share of a 24-hour shape that falls inside :data:`DAYTIME_WINDOW`."""
    start, end = DAYTIME_WINDOW
    return float(np.asarray(shape, dtype=float)[start:end].sum())


def archetype_daytime_fraction(archetype: str) -> float:
    """The natural daytime fraction of an archetype, before any override."""
    return daytime_fraction_of(base_shape(archetype))


def reshape_to_daytime_fraction(shape: np.ndarray, target: float) -> np.ndarray:
    """Rescale a daily shape to hit a target daytime fraction.

    Daytime hours are scaled by one factor and the rest of the day by another,
    so the daily total is preserved exactly and the shape *within* each window
    is untouched. Sliding the fraction therefore moves energy between day and
    evening without inventing a new pattern of use.

    Args:
        shape: A 24-hour shape summing to 1.
        target: Desired share of the day's energy in :data:`DAYTIME_WINDOW`.

    Returns:
        A new 24-hour shape summing to 1.

    Raises:
        ValueError: If the target is not strictly between 0 and 1, or the input
            shape has no energy to scale in one of the two windows.
    """
    if not 0.0 < target < 1.0:
        raise ValueError(f"daytime fraction must be in (0, 1), got {target}")

    shape = np.asarray(shape, dtype=float)
    start, end = DAYTIME_WINDOW
    mask = np.zeros(24, dtype=bool)
    mask[start:end] = True

    current = float(shape[mask].sum())
    if not 0.0 < current < 1.0:
        raise ValueError("cannot reshape a profile with all or no energy in the daytime")

    out = shape.copy()
    out[mask] *= target / current
    out[~mask] *= (1.0 - target) / (1.0 - current)
    return out / out.sum()


def seasonal_daily_weights(latitude: float = 51.5) -> np.ndarray:
    """Relative demand per day by calendar month, normalised to a mean of 1.

    Southern-hemisphere sites get the same curve shifted by six months.

    Args:
        latitude: Site latitude; only its sign is used.

    Returns:
        Twelve weights, January first.
    """
    weights = np.asarray(_SEASONAL_DAILY_WEIGHTS, dtype=float)
    if latitude < 0:
        weights = np.roll(weights, 6)
    return weights / weights.mean()


@dataclass(frozen=True)
class HouseholdShape:
    """When a household uses electricity, as distinct from how much.

    Attributes:
        archetype: Weekday pattern; one of :data:`ARCHETYPES`.
        daytime_fraction: Override the weekday share of consumption falling in
            :data:`DAYTIME_WINDOW`. ``None`` keeps the archetype's own value.
            This is the single most important knob for battery sizing.
        weekend_archetype: Weekend pattern. ``None`` picks a sensible partner
            for the weekday archetype -- a household that is out on Tuesday is
            usually in on Sunday.
        weekend_daytime_fraction: Override for the weekend shape.
    """

    archetype: str = "nine_to_five"
    daytime_fraction: float | None = None
    weekend_archetype: str | None = None
    weekend_daytime_fraction: float | None = None

    def __post_init__(self) -> None:
        base_shape(self.archetype)  # validates
        if self.weekend_archetype is not None:
            base_shape(self.weekend_archetype)
        for name in ("daytime_fraction", "weekend_daytime_fraction"):
            value = getattr(self, name)
            if value is not None and not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {value}")

    def weekday_shape(self) -> np.ndarray:
        """The resolved 24-hour weekday shape."""
        shape = base_shape(self.archetype)
        if self.daytime_fraction is None:
            return shape
        return reshape_to_daytime_fraction(shape, self.daytime_fraction)

    def weekend_shape(self) -> np.ndarray:
        """The resolved 24-hour weekend shape."""
        name = self.weekend_archetype or _WEEKEND_DEFAULTS.get(
            self.archetype, self.archetype
        )
        shape = base_shape(name)
        if self.weekend_daytime_fraction is None:
            return shape
        return reshape_to_daytime_fraction(shape, self.weekend_daytime_fraction)


@dataclass(frozen=True)
class LoadProfile:
    """An 8760-hour demand series aligned with the typical year.

    Attributes:
        kwh: Hourly consumption, kWh. Numerically also the mean kW in the hour.
        source: Human-readable provenance, shown to the user so a synthesised
            profile is never mistaken for a metered one.
        measured: True when built from real interval data.
        daytime_fraction: Realised share of consumption in
            :data:`DAYTIME_WINDOW`, computed from the series either way. For a
            metered profile this is a measurement, and it is the number to feed
            back into a synthesised one for a comparable household.
        household: The shape assumptions used, when synthesised.
    """

    kwh: np.ndarray
    source: str
    measured: bool
    daytime_fraction: float
    household: HouseholdShape | None = None

    def __post_init__(self) -> None:
        if self.kwh.shape != (HOURS_PER_YEAR,):
            raise ValueError(f"load profile must be {HOURS_PER_YEAR} hours")
        if not np.all(np.isfinite(self.kwh)):
            raise ValueError("load profile contains non-finite values")
        if np.any(self.kwh < 0):
            raise ValueError("load profile contains negative consumption")

    @property
    def annual_kwh(self) -> float:
        """Total annual consumption, kWh."""
        return float(self.kwh.sum())

    def monthly_kwh(self, reference_year: int = REFERENCE_YEAR) -> np.ndarray:
        """Consumption totalled by calendar month, kWh."""
        return np.bincount(
            _month_index(reference_year), weights=self.kwh, minlength=12
        )

    def scaled_to(self, annual_kwh: float) -> LoadProfile:
        """The same shape, rescaled to a different annual total."""
        if annual_kwh <= 0:
            raise ValueError("annual_kwh must be positive")
        factor = annual_kwh / self.annual_kwh
        return replace(self, kwh=self.kwh * factor)

    @classmethod
    def from_hourly(
        cls,
        values: Sequence[float] | np.ndarray,
        *,
        source: str = "Metered interval data",
    ) -> LoadProfile:
        """Build a profile from real interval data.

        Accepts any resolution that divides the year evenly -- hourly,
        half-hourly, quarter-hourly or minute data -- and sums it down to
        hours. A leap year is accepted and 29 February dropped, matching the
        typical year's fixed 8760 hours.

        The series is aligned to the typical year *by position*, from the first
        hour of January. Since the typical year is assembled from months of
        different real years, calendar dates could not be made to agree anyway;
        what this does mean is that weekdays may land a few days out of step.
        For annual totals that is immaterial, and no alternative is better.

        Args:
            values: Consumption per interval, kWh. Not average power -- if the
                meter reports kW, divide by the intervals per hour first.
            source: Provenance label carried through to the result.

        Returns:
            A :class:`LoadProfile`.

        Raises:
            ValueError: If the length is not a whole number of intervals per
                hour for a 365- or 366-day year.
        """
        series = np.asarray(values, dtype=float).ravel()

        for days, leap in ((365, False), (366, True)):
            hours = days * 24
            if series.size % hours == 0 and series.size > 0:
                per_hour = series.size // hours
                hourly = series.reshape(hours, per_hour).sum(axis=1)
                if leap:
                    hourly = _drop_leap_day(hourly)
                return cls(
                    kwh=hourly,
                    source=source,
                    measured=True,
                    daytime_fraction=_realised_daytime_fraction(hourly),
                )

        raise ValueError(
            f"expected a whole year at a fixed interval, got {series.size} values; "
            "8760 (hourly), 17520 (half-hourly) and 35040 (quarter-hourly) all work"
        )


def synthesise(
    annual_kwh: float | None = None,
    *,
    monthly_kwh: Sequence[float] | None = None,
    household: HouseholdShape | None = None,
    latitude: float = 51.5,
    reference_year: int = REFERENCE_YEAR,
) -> LoadProfile:
    """Build a plausible 8760-hour profile from billing data.

    Give it either a single annual figure -- which is split across the months
    by a seasonal curve -- or twelve monthly totals read straight off the
    bills, which is considerably better. Within each month, energy is spread
    over weekdays and weekends by the household's shape.

    The result is a *plausible* profile, not the household's actual one. Use
    :func:`daytime_fraction_band` to see how much the answer moves under
    reasonable alternative assumptions, and quote the range.

    Args:
        annual_kwh: Total annual consumption, kWh. Mutually exclusive with
            ``monthly_kwh``.
        monthly_kwh: Twelve monthly totals, January first, kWh.
        household: When the household uses electricity. Defaults to a
            nine-to-five household.
        latitude: Used only to orient the seasonal curve by hemisphere.
        reference_year: Non-leap year defining the calendar and weekdays.

    Returns:
        A :class:`LoadProfile`.

    Raises:
        ValueError: If neither or both of the totals are given, if any total is
            negative, or if the annual total is not positive.
    """
    if (annual_kwh is None) == (monthly_kwh is None):
        raise ValueError("give exactly one of annual_kwh or monthly_kwh")

    household = household or HouseholdShape()

    if monthly_kwh is not None:
        months = np.asarray(monthly_kwh, dtype=float)
        if months.shape != (12,):
            raise ValueError(f"monthly_kwh must have 12 entries, got {months.size}")
        if np.any(months < 0):
            raise ValueError("monthly_kwh cannot be negative")
        if not np.all(np.isfinite(months)):
            raise ValueError("monthly_kwh contains non-finite values")
        detail = "twelve monthly totals"
    else:
        if not annual_kwh > 0:
            raise ValueError("annual_kwh must be positive")
        weights = seasonal_daily_weights(latitude)
        days = _days_per_month(reference_year)
        share = weights * days
        months = annual_kwh * share / share.sum()
        detail = "an annual total and a seasonal curve"

    if months.sum() <= 0:
        raise ValueError("total consumption must be positive")

    day_month = _month_index(reference_year, per="day")
    weekend = _is_weekend(reference_year)

    # Weekends carry a little more energy than weekdays; normalise the day
    # weights within each month so the monthly totals come out exactly right.
    day_weight = np.where(weekend, WEEKEND_UPLIFT, 1.0)
    month_weight = np.bincount(day_month, weights=day_weight, minlength=12)
    day_total = day_weight * months[day_month] / month_weight[day_month]

    shapes = np.where(
        weekend[:, None],
        household.weekend_shape()[None, :],
        household.weekday_shape()[None, :],
    )
    hourly = (day_total[:, None] * shapes).ravel()

    return LoadProfile(
        kwh=hourly,
        source=f"Synthesised from {detail}",
        measured=False,
        daytime_fraction=_realised_daytime_fraction(hourly),
        household=household,
    )


def daytime_fraction_band(
    household: HouseholdShape, spread: float = 0.07
) -> tuple[HouseholdShape, HouseholdShape]:
    """Plausible low and high variants of a household's daytime fraction.

    A synthesised profile is a guess about behaviour, and the honest way to
    report a result built on one is as a range. This brackets the assumption by
    ``spread`` either side, clamped to the range of real domestic behaviour.

    Args:
        household: The central assumption.
        spread: How far either side to reach, in absolute daytime fraction.

    Returns:
        The low-daytime and high-daytime variants, in that order.
    """
    centre = household.daytime_fraction
    if centre is None:
        centre = daytime_fraction_of(household.weekday_shape())
    low = float(np.clip(centre - spread, 0.10, 0.70))
    high = float(np.clip(centre + spread, 0.10, 0.70))
    return (
        replace(household, daytime_fraction=low),
        replace(household, daytime_fraction=high),
    )


def _realised_daytime_fraction(hourly: np.ndarray) -> float:
    """The daytime share of a whole-year hourly series."""
    total = float(hourly.sum())
    if total <= 0:
        return 0.0
    start, end = DAYTIME_WINDOW
    daytime = hourly.reshape(-1, 24)[:, start:end].sum()
    return float(daytime / total)


def _days_per_month(year: int) -> np.ndarray:
    """Days in each calendar month of ``year``."""
    starts = np.array(
        [np.datetime64(f"{year}-{m:02d}-01", "D") for m in range(1, 13)]
        + [np.datetime64(f"{year + 1}-01-01", "D")]
    )
    return np.diff(starts).astype(int)


def _month_index(year: int, per: str = "hour") -> np.ndarray:
    """Zero-based calendar month of each hour (or day) of a 365-day year."""
    days = np.repeat(np.arange(12), _days_per_month(year))
    return days if per == "day" else np.repeat(days, 24)


def _is_weekend(year: int) -> np.ndarray:
    """True for each Saturday and Sunday of a 365-day year."""
    start = np.datetime64(f"{year}-01-01", "D").astype(int)
    # 1970-01-01 was a Thursday, index 3 counting Monday as 0.
    weekday = (start + np.arange(365) + 3) % 7
    return weekday >= 5


def _drop_leap_day(hourly: np.ndarray) -> np.ndarray:
    """Remove 29 February so a leap year becomes a clean 8760 hours."""
    feb29 = 31 + 28  # zero-based day index of 29 February
    keep = np.ones(366 * 24, dtype=bool)
    keep[feb29 * 24 : (feb29 + 1) * 24] = False
    return hourly[keep]
