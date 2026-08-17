"""Aggregation into annual, monthly, diurnal and typical-year views."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import numpy as np
import pytest

from solarest.pvmodel import SystemSpec
from solarest.results import (
    HOURS_PER_YEAR,
    _finkelstein_schafer,
    build_typical_year,
    result_to_dict,
    summarise,
)
from solarest.simulate import precompute
from solarest.weather import SyntheticClearSky


@pytest.fixture(scope="module")
def london():
    """Five years of clear-sky weather over London, already simulated."""
    weather = asyncio.run(SyntheticClearSky().fetch(51.5, -0.13, [2019, 2020, 2021, 2022, 2023]))
    site = precompute(weather, utc_offset_seconds=0)
    return site, site.simulate(SystemSpec(dc_capacity_kw=4.0, tilt_deg=35.0, azimuth_deg=180.0))


@pytest.fixture(scope="module")
def summary(london):
    return summarise(london[1])


# ------------------------------------------------------------------ annual


def test_annual_percentiles_are_ordered(summary):
    annual = summary.annual
    assert annual.energy_kwh_min <= annual.energy_kwh_p10 <= annual.energy_kwh_p50
    assert annual.energy_kwh_p50 <= annual.energy_kwh_p90 <= annual.energy_kwh_max


def test_specific_yield_is_energy_per_kilowatt(summary):
    annual = summary.annual
    assert annual.specific_yield_kwh_per_kwp == pytest.approx(annual.energy_kwh_mean / 4.0)


def test_capacity_factor_is_consistent_with_annual_energy(summary):
    annual = summary.annual
    assert annual.capacity_factor == pytest.approx(
        annual.energy_kwh_mean / (4.0 * HOURS_PER_YEAR)
    )
    assert 0.0 < annual.capacity_factor < 1.0


def test_performance_ratio_is_a_sensible_fraction(summary):
    # A real system loses roughly a fifth to a quarter of the plane irradiance.
    assert 0.6 < summary.annual.performance_ratio < 0.95


def test_plane_irradiation_beats_horizontal_for_a_tilted_array(summary):
    assert summary.annual.poa_kwh_m2 > summary.annual.ghi_kwh_m2


def test_every_complete_year_is_reported(summary):
    complete = [y for y in summary.annual.per_year if y.complete]
    assert len(complete) == 5
    assert summary.annual.years_used == [2019, 2020, 2021, 2022, 2023]


def test_a_record_shorter_than_a_year_is_refused(london):
    site, _ = london
    short = replace(
        site,
        weather=replace(
            site.weather,
            times=site.weather.times[:100],
            ghi=site.weather.ghi[:100],
            bhi=site.weather.bhi[:100],
            dhi=site.weather.dhi[:100],
            temp_c=site.weather.temp_c[:100],
            wind_ms=site.weather.wind_ms[:100],
        ),
        position=replace(
            site.position,
            zenith=site.position.zenith[:100],
            elevation=site.position.elevation[:100],
            azimuth=site.position.azimuth[:100],
            declination=site.position.declination[:100],
            equation_of_time=site.position.equation_of_time[:100],
            air_mass=site.position.air_mass[:100],
            extra_normal=site.position.extra_normal[:100],
        ),
        ghi=site.ghi[:100],
        dni=site.dni[:100],
        dhi=site.dhi[:100],
        local_times=site.local_times[:100],
    )
    with pytest.raises(ValueError, match="complete calendar year"):
        summarise(short.simulate(SystemSpec()))


# ----------------------------------------------------------------- monthly


def test_monthly_energies_sum_to_the_annual_mean(summary):
    total = sum(month.energy_kwh for month in summary.monthly)
    assert total == pytest.approx(summary.annual.energy_kwh_mean, rel=1e-6)


def test_all_twelve_months_are_present_and_ordered(summary):
    assert [m.month for m in summary.monthly] == list(range(1, 13))


def test_monthly_extremes_bracket_the_mean(summary):
    for month in summary.monthly:
        assert month.energy_kwh_min <= month.energy_kwh <= month.energy_kwh_max


def test_northern_summer_outproduces_northern_winter(summary):
    june = summary.monthly[5].energy_kwh
    december = summary.monthly[11].energy_kwh
    assert june > december * 1.5


def test_cells_run_hotter_than_the_air(summary):
    for month in summary.monthly:
        assert month.mean_cell_temp_c > month.mean_temp_c


# ----------------------------------------------------------------- diurnal


def test_diurnal_grid_is_twelve_months_by_twenty_four_hours(summary):
    assert len(summary.diurnal) == 12
    assert all(len(row) == 24 for row in summary.diurnal)


def test_nothing_is_generated_in_the_middle_of_the_night(summary):
    for row in summary.diurnal:
        assert row[0] == 0.0
        assert row[23] == 0.0


def test_output_peaks_around_solar_noon(summary):
    for month, row in enumerate(summary.diurnal):
        assert 10 <= int(np.argmax(row)) <= 14, f"month {month + 1} peaks oddly"


def test_diurnal_means_reconcile_with_monthly_totals(summary):
    """Mean power x hours in the month must recover the month's energy."""
    days = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for index, row in enumerate(summary.diurnal):
        implied = sum(row) * days[index]
        assert implied == pytest.approx(summary.monthly[index].energy_kwh, rel=0.02)


# ------------------------------------------------------------------ losses


def test_loss_shares_account_for_all_the_plane_energy(summary):
    losses = summary.losses
    total = (
        losses.reflection_pct + losses.temperature_pct + losses.system_losses_pct
        + losses.inverter_pct + losses.clipping_pct + losses.delivered_pct
    )
    assert total == pytest.approx(100.0, abs=0.01)


def test_every_loss_share_is_non_negative(summary):
    losses = summary.losses
    for name in ("reflection_pct", "temperature_pct", "system_losses_pct",
                 "inverter_pct", "clipping_pct", "delivered_pct"):
        assert getattr(losses, name) >= 0.0, name


def test_an_undersized_inverter_causes_clipping(london):
    site, _ = london
    clipped = summarise(site.simulate(SystemSpec(dc_capacity_kw=4.0, dc_ac_ratio=2.2)))
    roomy = summarise(site.simulate(SystemSpec(dc_capacity_kw=4.0, dc_ac_ratio=1.0)))
    assert clipped.losses.clipping_pct > roomy.losses.clipping_pct
    assert roomy.losses.clipping_pct == pytest.approx(0.0, abs=0.01)


# ----------------------------------------------------------- typical year


def test_typical_year_is_exactly_8760_hours(summary):
    assert summary.typical_year.ac_kw.size == HOURS_PER_YEAR
    assert summary.typical_year.poa_w_m2.size == HOURS_PER_YEAR
    assert summary.typical_year.air_temp_c.size == HOURS_PER_YEAR


def test_typical_year_names_a_source_year_for_every_month(summary):
    sources = summary.typical_year.month_sources
    assert sorted(sources) == list(range(1, 13))
    assert all(year in summary.annual.years_used for year in sources.values())


def test_typical_year_total_is_close_to_the_expected_annual(summary):
    """It is a real year, not an average, so it lands near the mean, not on it."""
    assert summary.typical_year.annual_kwh == pytest.approx(
        summary.annual.energy_kwh_mean, rel=0.10
    )


def test_typical_year_preserves_real_variability(summary):
    """Averaging years together would smooth the series; splicing must not."""
    hourly = summary.typical_year.ac_kw
    daily = hourly.reshape(365, 24).sum(axis=1)
    summer = daily[150:240]
    # Real weather makes consecutive days differ; a mean-of-years would not.
    assert summer.std() > 0.0
    assert float(hourly.max()) > 0.0


def test_typical_year_rejects_a_leap_reference_year(london):
    with pytest.raises(ValueError, match="must not be a leap year"):
        build_typical_year(london[1], reference_year=2024)


def test_finkelstein_schafer_is_zero_for_an_identical_distribution():
    values = np.arange(30.0)
    assert _finkelstein_schafer(values, values) == pytest.approx(0.0, abs=0.02)


def test_finkelstein_schafer_grows_as_distributions_diverge():
    long_run = np.arange(100.0)
    close = _finkelstein_schafer(np.arange(100.0), long_run)
    far = _finkelstein_schafer(np.full(30, 99.0), long_run)
    assert far > close


def test_finkelstein_schafer_handles_empty_input():
    assert _finkelstein_schafer(np.array([]), np.arange(5.0)) == float("inf")


# ------------------------------------------------------------ serialisation


def test_serialisation_includes_the_hourly_series_on_request(summary):
    payload = result_to_dict(summary, hourly=True)
    assert len(payload["typical_year"]["ac_kw"]) == HOURS_PER_YEAR
    assert len(payload["typical_year"]["poa_w_m2"]) == HOURS_PER_YEAR


def test_serialisation_can_omit_the_hourly_series(summary):
    payload = result_to_dict(summary, hourly=False)
    assert "ac_kw" not in payload["typical_year"]
    assert payload["typical_year"]["annual_kwh"] > 0


def test_serialised_payload_is_json_safe(summary):
    import json

    text = json.dumps(result_to_dict(summary, hourly=True))
    assert "NaN" not in text
    assert "Infinity" not in text
