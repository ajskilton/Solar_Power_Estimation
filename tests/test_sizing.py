"""Sizing end to end: a real typical year, a household, and a battery."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from solarest.battery import BatterySpec
from solarest.load import HouseholdShape, LoadProfile, synthesise
from solarest.pvmodel import SystemSpec
from solarest.results import HOURS_PER_YEAR, build_typical_year
from solarest.simulate import precompute
from solarest.sizing import size_for_household
from solarest.weather import SyntheticClearSky

LATITUDE, LONGITUDE = 51.5, -0.13


@pytest.fixture(scope="module")
def typical_year():
    """One clear-sky year through the full PV chain, as the website would."""
    weather = asyncio.run(SyntheticClearSky().fetch(LATITUDE, LONGITUDE, [2023]))
    site = precompute(weather, utc_offset_seconds=0)
    return build_typical_year(site.simulate(SystemSpec(dc_capacity_kw=4.0)))


@pytest.fixture(scope="module")
def nine_to_five():
    return synthesise(3500.0, household=HouseholdShape(archetype="nine_to_five"))


def test_the_typical_year_and_the_load_are_on_the_same_grid(typical_year, nine_to_five):
    assert typical_year.ac_kw.size == HOURS_PER_YEAR
    assert nine_to_five.kwh.size == HOURS_PER_YEAR


def test_a_complete_result_comes_back(typical_year, nine_to_five):
    result = size_for_household(typical_year, nine_to_five)

    assert result.result.avoided_import_kwh > 0
    assert len(result.monthly) == 12
    assert len(result.curve) > 5
    assert result.band is not None
    assert not result.measured_load


def test_the_monthly_balance_adds_back_up_to_the_year(typical_year, nine_to_five):
    result = size_for_household(typical_year, nine_to_five)

    assert sum(m.load_kwh for m in result.monthly) == pytest.approx(
        nine_to_five.annual_kwh
    )
    assert sum(m.generation_kwh for m in result.monthly) == pytest.approx(
        float(typical_year.ac_kw.sum())
    )
    assert sum(m.avoided_import_kwh for m in result.monthly) == pytest.approx(
        result.result.avoided_import_kwh
    )
    assert sum(m.import_kwh for m in result.monthly) == pytest.approx(
        result.result.total_import_kwh
    )


def test_summer_months_are_more_self_sufficient_than_winter(typical_year, nine_to_five):
    result = size_for_household(typical_year, nine_to_five)
    by_month = {m.month: m.self_sufficiency_pct for m in result.monthly}
    assert by_month[6] > by_month[12]


def test_the_battery_contribution_is_reported_against_pv_alone(typical_year, nine_to_five):
    result = size_for_household(
        typical_year, nine_to_five, BatterySpec(usable_capacity_kwh=8.0)
    )
    assert result.battery_contribution_kwh > 0
    assert result.pv_only.discharge_kwh.sum() == 0.0
    assert result.result.avoided_import_kwh > result.pv_only.avoided_import_kwh


# ------------------------------------------------- the premise of the module


def test_a_battery_is_worth_far_more_to_a_household_that_is_out_all_day(typical_year):
    """The claim the daytime fraction exists to capture.

    Two households, same annual consumption, same array, same battery. The one
    that is out at work has less coincidence to start with, so the battery has
    more work to do and adds more.
    """
    out = synthesise(3500.0, household=HouseholdShape(archetype="nine_to_five"))
    home = synthesise(3500.0, household=HouseholdShape(archetype="home_all_day"))
    battery = BatterySpec(usable_capacity_kwh=8.0)

    out_result = size_for_household(typical_year, out, battery)
    home_result = size_for_household(typical_year, home, battery)

    # The household that is in all day self-consumes more without any battery.
    assert home_result.pv_only.avoided_import_kwh > out_result.pv_only.avoided_import_kwh
    # But the battery earns its keep more for the household that is out.
    assert out_result.battery_contribution_kwh > home_result.battery_contribution_kwh


def test_two_households_with_identical_bills_get_different_answers(typical_year):
    """Monthly totals alone cannot distinguish these, which is the whole problem."""
    bills = [400.0, 350.0, 320.0, 270.0, 230.0, 200.0,
             200.0, 215.0, 250.0, 300.0, 360.0, 405.0]
    out = synthesise(monthly_kwh=bills, household=HouseholdShape(daytime_fraction=0.16))
    home = synthesise(monthly_kwh=bills, household=HouseholdShape(daytime_fraction=0.45))

    assert out.annual_kwh == pytest.approx(home.annual_kwh)

    battery = BatterySpec(usable_capacity_kwh=5.0)
    out_kwh = size_for_household(typical_year, out, battery).result.avoided_import_kwh
    home_kwh = size_for_household(typical_year, home, battery).result.avoided_import_kwh

    assert home_kwh > out_kwh * 1.02  # a materially different answer


# ------------------------------------------------------------ the band


def test_the_band_brackets_the_central_estimate(typical_year, nine_to_five):
    result = size_for_household(typical_year, nine_to_five)
    band = result.band

    assert band is not None
    assert band.low_kwh < result.result.avoided_import_kwh < band.high_kwh
    assert band.low_daytime_fraction < band.high_daytime_fraction
    assert band.spread_pct > 0


def test_a_wider_assumption_gives_a_wider_band(typical_year, nine_to_five):
    narrow = size_for_household(typical_year, nine_to_five, band_spread=0.03)
    wide = size_for_household(typical_year, nine_to_five, band_spread=0.15)
    assert wide.band.spread_pct > narrow.band.spread_pct


def test_a_battery_makes_a_bills_based_estimate_far_more_trustworthy(
    typical_year, nine_to_five
):
    """The most useful thing this module has to say.

    The load-shape assumption governs how much generation coincides with
    demand. A battery absorbs that mismatch, so it also absorbs the error in
    the assumption -- leaving the battery answer robust to a guess that the
    no-battery answer is highly sensitive to.
    """
    result = size_for_household(
        typical_year, nine_to_five, BatterySpec(usable_capacity_kwh=5.0)
    )
    assert result.band.spread_pct < result.band_pv_only.spread_pct / 3


def test_metered_data_gets_no_band_because_it_needs_none(typical_year, nine_to_five):
    """The band prices an assumption. Measured data has not made one."""
    measured = LoadProfile.from_hourly(nine_to_five.kwh)
    result = size_for_household(typical_year, measured)

    assert result.band is None
    assert result.band_pv_only is None
    assert result.measured_load
    assert "Metered" in result.load_source


def test_the_band_holds_annual_consumption_fixed(typical_year):
    """Only the shape is allowed to vary, or the band would measure the wrong thing."""
    load = synthesise(3500.0, household=HouseholdShape(daytime_fraction=0.25))
    result = size_for_household(typical_year, load, band_spread=0.10)

    # Both variants are regenerated from the same monthly totals, so any
    # difference in avoided import is attributable to shape alone.
    assert result.band.low_kwh < result.band.high_kwh
    assert result.result.load_kwh == pytest.approx(3500.0)


# ---------------------------------------------------------------- the curve


def test_the_suggested_capacity_is_on_the_curve(typical_year, nine_to_five):
    result = size_for_household(typical_year, nine_to_five)
    assert result.suggested_capacity_kwh in {e.capacity_kwh for e in result.curve}


def test_the_curve_starts_at_no_battery(typical_year, nine_to_five):
    result = size_for_household(typical_year, nine_to_five)
    assert result.curve[0].capacity_kwh == 0.0
    assert result.curve[0].avoided_import_kwh == pytest.approx(
        result.pv_only.avoided_import_kwh
    )


def test_a_custom_sweep_is_honoured(typical_year, nine_to_five):
    result = size_for_household(typical_year, nine_to_five, sweep_kwh=[0.0, 7.5])
    assert [e.capacity_kwh for e in result.curve] == [0.0, 7.5]


def test_a_generation_series_that_is_not_a_full_year_is_rejected(nine_to_five):
    """LoadProfile guards its own length; this guards the other input."""
    from dataclasses import replace

    stub = replace(
        _blank_typical_year(), ac_kw=np.ones(100), annual_kwh=100.0
    )
    with pytest.raises(ValueError, match="8760 hours"):
        size_for_household(stub, nine_to_five)


def _blank_typical_year():
    from solarest.results import REFERENCE_YEAR, TypicalYear

    zeros = np.zeros(HOURS_PER_YEAR)
    return TypicalYear(
        reference_year=REFERENCE_YEAR,
        month_sources={m: REFERENCE_YEAR for m in range(1, 13)},
        ac_kw=zeros,
        poa_w_m2=zeros,
        air_temp_c=zeros,
        annual_kwh=0.0,
    )
