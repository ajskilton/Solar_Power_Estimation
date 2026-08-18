"""Household demand profiles: the shapes, and the two ways in."""

from __future__ import annotations

import numpy as np
import pytest

from solarest.load import (
    ARCHETYPES,
    DAYTIME_WINDOW,
    HouseholdShape,
    LoadProfile,
    archetype_daytime_fraction,
    base_shape,
    daytime_fraction_band,
    daytime_fraction_of,
    reshape_to_daytime_fraction,
    seasonal_daily_weights,
    synthesise,
)
from solarest.results import HOURS_PER_YEAR


# ------------------------------------------------------------------ shapes


@pytest.mark.parametrize("archetype", ARCHETYPES)
def test_every_shape_is_a_normalised_day(archetype):
    shape = base_shape(archetype)
    assert shape.shape == (24,)
    assert shape.sum() == pytest.approx(1.0)
    assert np.all(shape > 0)


def test_unknown_archetypes_are_rejected():
    with pytest.raises(ValueError, match="unknown archetype"):
        base_shape("submarine")


def test_archetypes_are_ordered_by_how_much_they_use_in_the_daytime():
    """The whole point of the archetypes is that they differ here."""
    out_at_work = archetype_daytime_fraction("nine_to_five")
    at_a_desk = archetype_daytime_fraction("working_from_home")
    at_home = archetype_daytime_fraction("home_all_day")
    assert out_at_work < at_a_desk < at_home


def test_the_nine_to_five_shape_peaks_in_the_evening():
    shape = base_shape("nine_to_five")
    assert 17 <= int(np.argmax(shape)) <= 21


def test_a_flat_day_puts_the_window_share_in_the_window():
    start, end = DAYTIME_WINDOW
    assert daytime_fraction_of(base_shape("flat")) == pytest.approx((end - start) / 24)


# --------------------------------------------------------------- reshaping


@pytest.mark.parametrize("target", [0.12, 0.25, 0.40, 0.65])
def test_reshaping_hits_the_target_exactly(target):
    shape = reshape_to_daytime_fraction(base_shape("nine_to_five"), target)
    assert daytime_fraction_of(shape) == pytest.approx(target)
    assert shape.sum() == pytest.approx(1.0)


def test_reshaping_leaves_the_pattern_within_each_window_alone():
    """Energy moves between day and evening, but the day's own shape is kept."""
    original = base_shape("nine_to_five")
    moved = reshape_to_daytime_fraction(original, 0.45)
    start, end = DAYTIME_WINDOW

    inside = moved[start:end] / original[start:end]
    assert inside.max() == pytest.approx(inside.min())  # one uniform factor

    evening = moved[end:] / original[end:]
    assert evening.max() == pytest.approx(evening.min())


def test_reshaping_is_a_no_op_at_the_natural_fraction():
    original = base_shape("home_all_day")
    same = reshape_to_daytime_fraction(original, daytime_fraction_of(original))
    assert same == pytest.approx(original)


@pytest.mark.parametrize("target", [0.0, 1.0, -0.2, 1.5])
def test_impossible_daytime_fractions_are_rejected(target):
    with pytest.raises(ValueError, match="must be in"):
        reshape_to_daytime_fraction(base_shape("flat"), target)


# ---------------------------------------------------------------- seasons


def test_seasonal_demand_is_higher_in_winter():
    weights = seasonal_daily_weights(51.5)
    assert weights.mean() == pytest.approx(1.0)
    assert weights[0] > weights[6]  # January over July


def test_the_southern_hemisphere_gets_its_winter_in_july():
    north = seasonal_daily_weights(51.5)
    south = seasonal_daily_weights(-33.9)
    assert south[6] > south[0]
    assert south == pytest.approx(np.roll(north, 6))


# ------------------------------------------------------------ synthesising


def test_an_annual_total_is_preserved():
    profile = synthesise(3500.0)
    assert profile.annual_kwh == pytest.approx(3500.0)
    assert profile.kwh.shape == (HOURS_PER_YEAR,)
    assert not profile.measured


def test_monthly_bills_are_reproduced_month_by_month():
    bills = [420.0, 380.0, 340.0, 280.0, 240.0, 210.0,
             205.0, 220.0, 260.0, 310.0, 380.0, 430.0]
    profile = synthesise(monthly_kwh=bills)
    assert profile.monthly_kwh() == pytest.approx(bills)
    assert profile.annual_kwh == pytest.approx(sum(bills))


def test_a_bare_annual_total_still_lands_more_in_winter():
    monthly = synthesise(3500.0).monthly_kwh()
    assert monthly[0] > monthly[6]  # January over July


def test_exactly_one_kind_of_total_is_required():
    with pytest.raises(ValueError, match="exactly one"):
        synthesise()
    with pytest.raises(ValueError, match="exactly one"):
        synthesise(3000.0, monthly_kwh=[250.0] * 12)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"annual_kwh": 0.0}, "must be positive"),
        ({"annual_kwh": -100.0}, "must be positive"),
        ({"monthly_kwh": [100.0] * 11}, "12 entries"),
        ({"monthly_kwh": [-1.0] * 12}, "cannot be negative"),
        ({"monthly_kwh": [float("nan")] * 12}, "non-finite"),
    ],
)
def test_nonsense_totals_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        synthesise(**kwargs)


def test_the_daytime_fraction_knob_moves_the_realised_profile():
    low = synthesise(3500.0, household=HouseholdShape(daytime_fraction=0.15))
    high = synthesise(3500.0, household=HouseholdShape(daytime_fraction=0.50))
    assert low.daytime_fraction < high.daytime_fraction
    assert low.annual_kwh == pytest.approx(high.annual_kwh)


def test_the_override_lands_on_the_weekday_shape_exactly():
    household = HouseholdShape(daytime_fraction=0.31)
    assert daytime_fraction_of(household.weekday_shape()) == pytest.approx(0.31)


def test_weekends_default_to_somebody_being_home():
    """A household out on Tuesday is usually in on Sunday."""
    household = HouseholdShape(archetype="nine_to_five")
    weekday = daytime_fraction_of(household.weekday_shape())
    weekend = daytime_fraction_of(household.weekend_shape())
    assert weekend > weekday


def test_weekends_can_be_shaped_independently():
    household = HouseholdShape(
        archetype="nine_to_five", weekend_archetype="flat", weekend_daytime_fraction=0.5
    )
    assert daytime_fraction_of(household.weekend_shape()) == pytest.approx(0.5)


def test_weekend_days_carry_a_little_more_than_weekdays():
    profile = synthesise(3650.0, household=HouseholdShape(archetype="flat"))
    daily = profile.kwh.reshape(365, 24).sum(axis=1)
    # 2023-01-01 was a Sunday, so the year opens with a weekend day.
    assert daily[0] > daily[2]


def test_invalid_household_shapes_are_rejected():
    with pytest.raises(ValueError, match="unknown archetype"):
        HouseholdShape(archetype="lighthouse")
    with pytest.raises(ValueError, match="daytime_fraction"):
        HouseholdShape(daytime_fraction=1.4)


# ------------------------------------------------------------ metered data


def test_hourly_data_passes_straight_through():
    values = np.linspace(0.1, 0.9, HOURS_PER_YEAR)
    profile = LoadProfile.from_hourly(values)
    assert profile.kwh == pytest.approx(values)
    assert profile.measured
    assert profile.household is None


def test_half_hourly_data_is_summed_into_hours():
    """A smart-meter export is half-hourly, and each hour is the sum of two."""
    values = np.full(HOURS_PER_YEAR * 2, 0.25)
    profile = LoadProfile.from_hourly(values)
    assert profile.kwh == pytest.approx(np.full(HOURS_PER_YEAR, 0.5))
    assert profile.annual_kwh == pytest.approx(values.sum())


@pytest.mark.parametrize("per_hour", [1, 2, 4, 60])
def test_any_regular_interval_is_accepted(per_hour):
    values = np.full(HOURS_PER_YEAR * per_hour, 1.0 / per_hour)
    assert LoadProfile.from_hourly(values).annual_kwh == pytest.approx(HOURS_PER_YEAR)


def test_a_leap_year_loses_the_twenty_ninth_of_february():
    values = np.zeros(366 * 24)
    feb29 = (31 + 28) * 24
    values[feb29 : feb29 + 24] = 99.0  # the day that must disappear
    values[:24] = 1.0

    profile = LoadProfile.from_hourly(values)
    assert profile.kwh.size == HOURS_PER_YEAR
    assert profile.kwh.max() == pytest.approx(1.0)


def test_a_ragged_series_is_rejected_with_a_useful_message():
    with pytest.raises(ValueError, match="whole year at a fixed interval"):
        LoadProfile.from_hourly(np.ones(1000))


def test_measured_profiles_report_their_own_daytime_fraction():
    """The number to feed back into a synthesised profile for a similar house."""
    day = np.zeros(24)
    day[DAYTIME_WINDOW[0] : DAYTIME_WINDOW[1]] = 1.0
    profile = LoadProfile.from_hourly(np.tile(day, 365))
    assert profile.daytime_fraction == pytest.approx(1.0)


# -------------------------------------------------------------- the record


def test_profiles_reject_impossible_series():
    with pytest.raises(ValueError, match="8760 hours"):
        LoadProfile(np.ones(100), "x", False, 0.3)
    with pytest.raises(ValueError, match="negative"):
        LoadProfile(-np.ones(HOURS_PER_YEAR), "x", False, 0.3)
    with pytest.raises(ValueError, match="non-finite"):
        LoadProfile(np.full(HOURS_PER_YEAR, np.nan), "x", False, 0.3)


def test_monthly_totals_add_back_up_to_the_year():
    profile = synthesise(4200.0)
    assert profile.monthly_kwh().sum() == pytest.approx(profile.annual_kwh)


def test_rescaling_keeps_the_shape_and_changes_the_size():
    profile = synthesise(3000.0)
    doubled = profile.scaled_to(6000.0)
    assert doubled.annual_kwh == pytest.approx(6000.0)
    assert doubled.daytime_fraction == pytest.approx(profile.daytime_fraction)


def test_the_band_brackets_the_central_assumption():
    household = HouseholdShape(archetype="nine_to_five")
    low, high = daytime_fraction_band(household, spread=0.07)
    centre = archetype_daytime_fraction("nine_to_five")
    assert low.daytime_fraction < centre < high.daytime_fraction


def test_the_band_stays_inside_believable_behaviour():
    _, high = daytime_fraction_band(HouseholdShape(daytime_fraction=0.68), spread=0.20)
    assert high.daytime_fraction <= 0.70
