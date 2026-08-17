"""Cross-validation of the whole model chain against pvlib.

pvlib is the reference open-source implementation of these same published
models. Agreeing with it stage by stage is much stronger evidence than any
hand-written expectation, because the two implementations share no code and
pvlib's solar position uses the NREL SPA rather than the condensed NOAA
algorithm used here.

pvlib is a development-only dependency (``pip install -e ".[dev]"``); these
tests skip when it is absent.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

pvlib = pytest.importorskip("pvlib", reason="pvlib is a dev-only cross-check dependency")
pd = pytest.importorskip("pandas")

from solarest.pvmodel import SystemSpec  # noqa: E402
from solarest.simulate import precompute  # noqa: E402
from solarest.weather import SyntheticClearSky, split_beam  # noqa: E402

LATITUDE, LONGITUDE = 51.5, -0.13
TILT, AZIMUTH, CAPACITY_KW = 35.0, 180.0, 4.0


@pytest.fixture(scope="module")
def comparison():
    """Run one year of identical weather through both implementations."""
    weather = asyncio.run(SyntheticClearSky().fetch(LATITUDE, LONGITUDE, [2023]))
    site = precompute(weather, utc_offset_seconds=0)
    spec = SystemSpec(
        dc_capacity_kw=CAPACITY_KW, tilt_deg=TILT, azimuth_deg=AZIMUTH,
        mount="roof_mount", module_type="premium",
        dc_ac_ratio=1.15, inverter_efficiency=0.96, albedo=0.2,
    )
    mine = site.run(spec)

    midpoints = pd.DatetimeIndex(weather.interval_midpoints()).tz_localize("UTC")
    position = pvlib.solarposition.spa_python(
        midpoints, LATITUDE, LONGITUDE, altitude=0, temperature=10
    )
    zenith = position["apparent_zenith"].values
    azimuth = position["azimuth"].values
    air_mass = pvlib.atmosphere.get_relative_airmass(zenith, "kastenyoung1989")
    dni_extra = np.asarray(pvlib.irradiance.get_extra_radiation(midpoints))

    # Feed both models the same reconciled irradiance components.
    ghi, dni, dhi = split_beam(
        weather.ghi, weather.bhi, weather.dhi,
        np.maximum(0.0, np.cos(np.radians(zenith))),
        site.position.extra_normal,
    )

    poa = pvlib.irradiance.get_total_irradiance(
        TILT, AZIMUTH, zenith, azimuth, dni=dni, ghi=ghi, dhi=dhi,
        dni_extra=dni_extra, airmass=air_mass, albedo=0.2, model="perez",
    )
    component = lambda key: np.nan_to_num(np.asarray(poa[key], dtype=float))  # noqa: E731

    aoi = pvlib.irradiance.aoi(TILT, AZIMUTH, zenith, azimuth)
    sky_angle = 59.7 - 0.1388 * TILT + 0.001497 * TILT**2
    ground_angle = 90.0 - 0.5788 * TILT + 0.002693 * TILT**2
    effective = (
        component("poa_direct") * pvlib.iam.ashrae(aoi, b=0.05)
        + component("poa_sky_diffuse") * pvlib.iam.ashrae(sky_angle, 0.05)
        + component("poa_ground_diffuse") * pvlib.iam.ashrae(ground_angle, 0.05)
    )
    cell_temp = pvlib.temperature.sapm_cell(
        component("poa_global"), weather.temp_c, weather.wind_ms, -2.98, -0.0471, 1.0
    )
    dc = pvlib.pvsystem.pvwatts_dc(effective, cell_temp, CAPACITY_KW, -0.0035) * (
        1 - spec.total_loss_fraction
    )
    ac = np.asarray(
        pvlib.inverter.pvwatts(dc, CAPACITY_KW / 1.15 / 0.96, eta_inv_nom=0.96)
    )

    return mine, {
        "poa_global": component("poa_global"),
        "poa_beam": component("poa_direct"),
        "poa_sky": component("poa_sky_diffuse"),
        "poa_ground": component("poa_ground_diffuse"),
        "effective": effective,
        "cell_temp": cell_temp,
        "dc": np.asarray(dc),
        "ac": ac,
    }


# The two implementations use different solar position algorithms, so a
# residual of a few hundredths of a percent over a year is expected and is
# dominated by that difference, not by the models under test.
TOLERANCE = 0.001  # 0.1% on annual totals


@pytest.mark.parametrize(
    "attribute, reference",
    [
        ("poa_global", "poa_global"),
        ("poa_beam", "poa_beam"),
        ("poa_sky", "poa_sky"),
        ("poa_ground", "poa_ground"),
    ],
)
def test_transposition_components_match_pvlib(comparison, attribute, reference):
    mine, theirs = comparison
    ours = {
        "poa_global": mine.poa.global_,
        "poa_beam": mine.poa.beam,
        "poa_sky": mine.poa.sky_diffuse,
        "poa_ground": mine.poa.ground_diffuse,
    }[attribute]
    assert ours.sum() == pytest.approx(theirs[reference].sum(), rel=TOLERANCE)


def test_reflection_losses_match_pvlib(comparison):
    mine, theirs = comparison
    assert mine.poa.effective.sum() == pytest.approx(theirs["effective"].sum(), rel=TOLERANCE)


def test_cell_temperature_matches_pvlib(comparison):
    mine, theirs = comparison
    assert mine.cell_temp_c.sum() == pytest.approx(theirs["cell_temp"].sum(), rel=TOLERANCE)
    assert np.abs(mine.cell_temp_c - theirs["cell_temp"]).max() < 0.05


def test_dc_output_matches_pvlib(comparison):
    mine, theirs = comparison
    assert mine.dc_kw.sum() == pytest.approx(theirs["dc"].sum(), rel=TOLERANCE)


def test_annual_ac_energy_matches_pvlib(comparison):
    """The bottom line: the number the website reports."""
    mine, theirs = comparison
    assert mine.ac_kw.sum() == pytest.approx(theirs["ac"].sum(), rel=TOLERANCE)
    assert np.abs(mine.ac_kw - theirs["ac"]).max() < 0.01  # kW, hour by hour


def test_solar_position_matches_nrel_spa_across_a_year():
    """Bound the underlying geometry difference that drives the residual above."""
    from solarest.solarpos import solar_position

    times = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="h", tz="UTC")
    reference = pvlib.solarposition.spa_python(
        times, LATITUDE, LONGITUDE, altitude=0, temperature=10
    )
    mine = solar_position(times.tz_localize(None).values.astype("datetime64[s]"),
                          LATITUDE, LONGITUDE)

    daylight = reference["apparent_elevation"].values > 5.0
    elevation_error = np.abs(mine.elevation - reference["apparent_elevation"].values)[daylight]
    azimuth_error = np.abs(
        (mine.azimuth - reference["azimuth"].values + 180) % 360 - 180
    )[daylight]
    assert elevation_error.max() < 0.05
    assert azimuth_error.max() < 0.10
