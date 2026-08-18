"""Open-Meteo parsing, caching, and irradiance reconciliation."""

from __future__ import annotations

import asyncio
import json
from datetime import date

import numpy as np
import pytest

from solarest.weather import (
    ArchiveError,
    OpenMeteoArchive,
    SyntheticClearSky,
    WeatherSeries,
    _parse_archive_response,
    concat_series,
    default_year_range,
    ineichen_clear_sky,
    split_beam,
    standard_utc_offset,
)
from solarest.solarpos import solar_position


def make_payload(hours: int = 24, **overrides) -> dict:
    times = [f"2023-01-01T{h:02d}:00" for h in range(hours)]
    payload = {
        "latitude": 51.5,
        "longitude": -0.13,
        "elevation": 25.0,
        "timezone": "GMT",
        "utc_offset_seconds": 0,
        "hourly_units": {
            "temperature_2m": "°C",
            "wind_speed_10m": "m/s",
            "shortwave_radiation": "W/m²",
        },
        "hourly": {
            "time": times,
            "shortwave_radiation": [100.0] * hours,
            "direct_radiation": [70.0] * hours,
            "diffuse_radiation": [30.0] * hours,
            "temperature_2m": [10.0] * hours,
            "wind_speed_10m": [3.0] * hours,
        },
    }
    payload["hourly"].update(overrides)
    return payload


# ------------------------------------------------------------------ parsing


def test_parses_a_well_formed_response():
    series = _parse_archive_response(make_payload(), 2023)
    assert len(series) == 24
    assert series.elevation_m == 25.0
    assert series.ghi[0] == 100.0
    assert series.bhi[0] == 70.0
    assert not series.synthetic
    assert "ERA5" in series.source


def test_missing_hourly_block_raises():
    with pytest.raises(ArchiveError, match="no hourly data"):
        _parse_archive_response({"hourly": {"time": []}}, 2023)


def test_missing_variable_raises():
    payload = make_payload()
    del payload["hourly"]["direct_radiation"]
    with pytest.raises(ArchiveError, match="missing direct_radiation"):
        _parse_archive_response(payload, 2023)


def test_null_irradiance_becomes_zero():
    payload = make_payload(shortwave_radiation=[None] * 12 + [200.0] * 12)
    series = _parse_archive_response(payload, 2023)
    assert float(series.ghi[:12].sum()) == 0.0
    assert series.ghi[12] == 200.0


def test_null_temperature_is_interpolated_across_the_gap():
    values = [0.0] + [None] * 3 + [8.0] + [5.0] * 19
    series = _parse_archive_response(make_payload(temperature_2m=values), 2023)
    assert series.temp_c[:5] == pytest.approx([0.0, 2.0, 4.0, 6.0, 8.0])
    assert not np.isnan(series.temp_c).any()


def test_all_null_temperature_falls_back_without_nans():
    series = _parse_archive_response(make_payload(temperature_2m=[None] * 24), 2023)
    assert not np.isnan(series.temp_c).any()
    assert float(series.temp_c[0]) == 15.0


def test_negative_irradiance_is_clipped_away():
    series = _parse_archive_response(make_payload(shortwave_radiation=[-5.0] * 24), 2023)
    assert float(series.ghi.min()) == 0.0


def test_wind_reported_in_kmh_is_converted():
    payload = make_payload()
    payload["hourly_units"]["wind_speed_10m"] = "km/h"
    payload["hourly"]["wind_speed_10m"] = [36.0] * 24
    series = _parse_archive_response(payload, 2023)
    assert float(series.wind_ms[0]) == pytest.approx(10.0)


def test_temperature_reported_in_fahrenheit_is_converted():
    payload = make_payload()
    payload["hourly_units"]["temperature_2m"] = "°F"
    payload["hourly"]["temperature_2m"] = [212.0] * 24
    series = _parse_archive_response(payload, 2023)
    assert float(series.temp_c[0]) == pytest.approx(100.0)


# ------------------------------------------------------------- concatenation


def test_concat_sorts_and_deduplicates():
    first = _parse_archive_response(make_payload(), 2023)
    duplicate = _parse_archive_response(make_payload(), 2023)
    joined = concat_series([duplicate, first])
    assert len(joined) == 24  # identical hours collapse
    assert np.all(np.diff(joined.times.astype("int64")) > 0)


def test_concat_of_one_returns_it_unchanged():
    only = _parse_archive_response(make_payload(), 2023)
    assert concat_series([only]) is only


def test_concat_of_nothing_raises():
    with pytest.raises(ValueError, match="nothing to concatenate"):
        concat_series([])


# ------------------------------------------------------------------- years


def test_default_year_range_uses_complete_years_only():
    years = default_year_range(5, today=date(2026, 6, 15))
    assert years == [2021, 2022, 2023, 2024, 2025]


def test_early_january_backs_off_a_year_for_the_reanalysis_lag():
    """ERA5 lags reality by about five days, so 1 January cannot use last year."""
    years = default_year_range(3, today=date(2026, 1, 2))
    assert years == [2022, 2023, 2024]


def test_year_range_is_bounded():
    with pytest.raises(ValueError, match="years must be between"):
        default_year_range(0)
    with pytest.raises(ValueError, match="years must be between"):
        default_year_range(99)


def test_standard_offset_ignores_daylight_saving():
    assert standard_utc_offset("Europe/London") == 0
    assert standard_utc_offset("Europe/Berlin") == 3600
    assert standard_utc_offset("America/Los_Angeles") == -8 * 3600
    # Southern hemisphere: standard time is the winter (mid-year) offset.
    assert standard_utc_offset("Australia/Sydney") == 10 * 3600
    assert standard_utc_offset("Not/AZone") == 0


# --------------------------------------------------------- reconciliation


def test_split_beam_conserves_the_energy_balance():
    """ghi must equal dni*cos(zenith) + dhi after reconciliation, always."""
    rng = np.random.default_rng(7)
    cos_zenith = rng.uniform(0.0, 1.0, 500)
    ghi = rng.uniform(0.0, 1000.0, 500)
    bhi = ghi * rng.uniform(0.0, 1.0, 500)
    dhi = ghi - bhi
    extra = np.full(500, 1361.0)

    out_ghi, dni, out_dhi = split_beam(ghi, bhi, dhi, cos_zenith, extra)
    assert out_ghi + 0.0 == pytest.approx(dni * cos_zenith + out_dhi, abs=1e-9)
    assert np.all(dni >= 0.0)
    assert np.all(out_dhi >= 0.0)
    assert np.all(dni <= extra)


def test_split_beam_treats_very_low_sun_as_entirely_diffuse():
    """Dividing beam by a near-zero cosine would invent absurd direct normal."""
    ghi = np.array([50.0])
    bhi = np.array([40.0])
    dhi = np.array([10.0])
    cos_zenith = np.array([0.001])  # sun essentially on the horizon
    _, dni, out_dhi = split_beam(ghi, bhi, dhi, cos_zenith, np.array([1361.0]))
    assert float(dni[0]) == 0.0
    assert float(out_dhi[0]) == pytest.approx(50.0)


def test_split_beam_caps_direct_normal_at_the_extraterrestrial_ceiling():
    _, dni, _ = split_beam(
        np.array([1200.0]), np.array([1200.0]), np.array([0.0]),
        np.array([0.1]), np.array([1361.0]),
    )
    assert float(dni[0]) <= 1361.0


def test_split_beam_handles_darkness():
    zeros = np.zeros(5)
    ghi, dni, dhi = split_beam(zeros, zeros, zeros, zeros, np.full(5, 1361.0))
    assert float(ghi.sum()) == 0.0
    assert float(dni.sum()) == 0.0
    assert float(dhi.sum()) == 0.0


# ----------------------------------------------------------- clear sky model


def test_clear_sky_is_dark_below_the_horizon():
    times = np.array(["2023-06-21T00:00:00"], dtype="datetime64[s]")
    position = solar_position(times, 51.5, -0.1)
    ghi, bhi, dhi = ineichen_clear_sky(
        position.zenith, position.air_mass, position.extra_normal, 0.0, 3.0
    )
    assert float(ghi[0]) == 0.0


def test_clear_sky_noon_irradiance_is_physically_plausible():
    times = np.array(["2023-06-21T12:00:00"], dtype="datetime64[s]")
    position = solar_position(times, 51.5, -0.1)
    ghi, bhi, dhi = ineichen_clear_sky(
        position.zenith, position.air_mass, position.extra_normal, 0.0, 3.0
    )
    assert 700.0 < float(ghi[0]) < 1000.0
    assert float(ghi[0]) == pytest.approx(float(bhi[0] + dhi[0]))


def test_clearer_skies_deliver_more_beam():
    times = np.array(["2023-06-21T12:00:00"], dtype="datetime64[s]")
    position = solar_position(times, 51.5, -0.1)
    clean = ineichen_clear_sky(position.zenith, position.air_mass, position.extra_normal, 0.0, 2.0)
    hazy = ineichen_clear_sky(position.zenith, position.air_mass, position.extra_normal, 0.0, 6.0)
    assert float(clean[0][0]) > float(hazy[0][0])
    assert float(clean[1][0]) > float(hazy[1][0])


# ----------------------------------------------------- synthetic provider


def test_synthetic_provider_generates_whole_years():
    provider = SyntheticClearSky()
    series = asyncio.run(provider.fetch(51.5, -0.1, [2022, 2023]))
    assert series.synthetic
    assert len(series) == 2 * 8760
    assert series.years == [2022, 2023, 2024]  # final hour lands at 00:00 on 1 Jan


def test_synthetic_provider_reports_a_site():
    info = asyncio.run(SyntheticClearSky().site_info(51.5, -0.1))
    assert info.utc_offset_seconds == 0
    assert info.latitude == 51.5


def test_synthetic_provider_rejects_bad_coordinates():
    with pytest.raises(ValueError, match="latitude must be"):
        asyncio.run(SyntheticClearSky().fetch(120.0, 0.0, [2023]))
    with pytest.raises(ValueError, match="longitude must be"):
        asyncio.run(SyntheticClearSky().fetch(0.0, 200.0, [2023]))


def test_synthetic_provider_needs_at_least_one_year():
    with pytest.raises(ValueError, match="at least one year"):
        asyncio.run(SyntheticClearSky().fetch(51.5, -0.1, []))


# ------------------------------------------------------------------- cache


def test_cache_round_trips_a_series(tmp_path):
    archive = OpenMeteoArchive(cache_dir=tmp_path)
    original = _parse_archive_response(make_payload(), 2023)
    archive._write_cache(51.5, -0.13, 2023, original)

    restored = archive._read_cache(51.5, -0.13, 2023)
    assert restored is not None
    assert restored.ghi == pytest.approx(original.ghi)
    assert restored.bhi == pytest.approx(original.bhi)
    assert restored.temp_c == pytest.approx(original.temp_c)
    assert restored.times.tolist() == original.times.tolist()
    assert restored.source == original.source
    assert restored.elevation_m == original.elevation_m


def test_cache_miss_returns_none(tmp_path):
    archive = OpenMeteoArchive(cache_dir=tmp_path)
    assert archive._read_cache(1.0, 2.0, 1999) is None


def test_corrupt_cache_file_is_discarded_not_fatal(tmp_path):
    archive = OpenMeteoArchive(cache_dir=tmp_path)
    path = archive._cache_path(51.5, -0.13, 2023)
    path.write_bytes(b"this is not a npz archive")
    assert archive._read_cache(51.5, -0.13, 2023) is None
    assert not path.exists()  # cleared so the next request refetches


def test_caching_can_be_disabled(tmp_path):
    archive = OpenMeteoArchive(cache_dir=None)
    assert archive._cache_path(51.5, -0.13, 2023) is None
    archive._write_cache(51.5, -0.13, 2023, _parse_archive_response(make_payload(), 2023))
    assert archive._read_cache(51.5, -0.13, 2023) is None


def test_nearby_coordinates_share_a_cache_slot(tmp_path):
    """Rounding is far finer than the ~25 km reanalysis grid, so this is safe."""
    archive = OpenMeteoArchive(cache_dir=tmp_path, cache_precision=3)
    assert archive._cache_path(51.50001, -0.13, 2023) == archive._cache_path(51.50002, -0.13, 2023)
    assert archive._cache_path(51.5, -0.13, 2023) != archive._cache_path(52.5, -0.13, 2023)


def test_site_cache_round_trips(tmp_path):
    from solarest.weather import SiteInfo

    archive = OpenMeteoArchive(cache_dir=tmp_path)
    info = SiteInfo(51.5, -0.13, 25.0, "Europe/London", 0)
    archive._write_site_cache(51.5, -0.13, info)
    assert archive._read_site_cache(51.5, -0.13) == info


def test_fetch_rejects_impossible_coordinates(tmp_path):
    archive = OpenMeteoArchive(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="latitude must be"):
        asyncio.run(archive.fetch(95.0, 0.0, [2023]))
