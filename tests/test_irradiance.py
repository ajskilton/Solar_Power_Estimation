"""Transposition onto the array plane, and reflection losses."""

from __future__ import annotations

import numpy as np
import pytest

from solarest.irradiance import (
    ashrae_iam,
    diffuse_effective_angles,
    hdkr_sky_diffuse,
    perez_sky_diffuse,
    transpose,
)
from solarest.solarpos import solar_position


@pytest.fixture
def clear_afternoon():
    """A clear midsummer afternoon in southern England."""
    times = np.arange(
        np.datetime64("2023-06-21T05:00:00"),
        np.datetime64("2023-06-21T20:00:00"),
        np.timedelta64(1, "h"),
    )
    position = solar_position(times, 51.5, -0.1)
    cos_zenith = position.cos_zenith
    dni = np.where(cos_zenith > 0, 850.0, 0.0)
    dhi = np.where(cos_zenith > 0, 110.0, 0.0)
    ghi = dni * cos_zenith + dhi
    return position, ghi, dni, dhi


def call_transpose(position, ghi, dni, dhi, tilt, azimuth, **kwargs):
    return transpose(
        surface_tilt_deg=tilt,
        surface_azimuth_deg=azimuth,
        ghi=ghi,
        dni=dni,
        dhi=dhi,
        solar_zenith_deg=position.zenith,
        solar_azimuth_deg=position.azimuth,
        extra_normal=position.extra_normal,
        air_mass=position.air_mass,
        **kwargs,
    )


@pytest.mark.parametrize("model", ["perez", "hdkr"])
def test_horizontal_plane_recovers_ghi(clear_afternoon, model):
    """A plane at zero tilt must see exactly the global horizontal irradiance.

    This is the strongest available check on a transposition model: beam,
    sky-diffuse and ground-reflected must recombine into the input.
    """
    position, ghi, dni, dhi = clear_afternoon
    poa = call_transpose(position, ghi, dni, dhi, tilt=0.0, azimuth=180.0, model=model)

    # Both models cap the circumsolar term below ~85 deg zenith, so compare
    # only where the sun is properly up.
    high = position.zenith < 80.0
    assert poa.global_[high] == pytest.approx(ghi[high], rel=1e-9)
    assert float(poa.ground_diffuse.max()) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("model", ["perez", "hdkr"])
def test_tilting_towards_the_equator_gains_energy_in_winter(model):
    """A south-facing tilt should beat flat in a northern winter."""
    times = np.arange(
        np.datetime64("2023-12-21T08:00:00"),
        np.datetime64("2023-12-21T17:00:00"),
        np.timedelta64(1, "h"),
    )
    position = solar_position(times, 51.5, -0.1)
    dni = np.where(position.cos_zenith > 0, 700.0, 0.0)
    dhi = np.where(position.cos_zenith > 0, 90.0, 0.0)
    ghi = dni * position.cos_zenith + dhi

    flat = call_transpose(position, ghi, dni, dhi, 0.0, 180.0, model=model)
    tilted = call_transpose(position, ghi, dni, dhi, 50.0, 180.0, model=model)
    assert tilted.global_.sum() > flat.global_.sum() * 1.5


def test_north_facing_array_loses_to_south_in_the_northern_hemisphere(clear_afternoon):
    position, ghi, dni, dhi = clear_afternoon
    south = call_transpose(position, ghi, dni, dhi, 35.0, 180.0)
    north = call_transpose(position, ghi, dni, dhi, 35.0, 0.0)
    assert south.global_.sum() > north.global_.sum()


def test_no_beam_reaches_a_plane_facing_away_from_the_sun():
    position = solar_position(
        np.array(["2023-06-21T12:00:00"], dtype="datetime64[s]"), 51.5, -0.1
    )
    dni = np.array([900.0])
    dhi = np.array([100.0])
    ghi = dni * position.cos_zenith + dhi
    # Vertical wall facing due north at midday: the sun is behind it.
    poa = call_transpose(position, ghi, dni, dhi, 90.0, 0.0)
    assert float(poa.beam[0]) == 0.0
    assert float(poa.global_[0]) > 0.0  # diffuse and ground reflection remain


def test_ground_reflection_scales_with_albedo_and_tilt(clear_afternoon):
    position, ghi, dni, dhi = clear_afternoon
    dark = call_transpose(position, ghi, dni, dhi, 40.0, 180.0, albedo=0.1)
    snow = call_transpose(position, ghi, dni, dhi, 40.0, 180.0, albedo=0.8)
    assert snow.ground_diffuse.sum() == pytest.approx(dark.ground_diffuse.sum() * 8.0)

    # A vertical surface sees half the ground hemisphere; a flat one sees none.
    vertical = call_transpose(position, ghi, dni, dhi, 90.0, 180.0, albedo=0.2)
    expected = ghi.sum() * 0.2 * 0.5
    assert vertical.ground_diffuse.sum() == pytest.approx(expected)


def test_night_hours_produce_no_irradiance():
    times = np.arange(
        np.datetime64("2023-12-21T22:00:00"),
        np.datetime64("2023-12-22T04:00:00"),
        np.timedelta64(1, "h"),
    )
    position = solar_position(times, 51.5, -0.1)
    zeros = np.zeros(times.size)
    poa = call_transpose(position, zeros, zeros, zeros, 35.0, 180.0)
    assert float(poa.global_.sum()) == 0.0
    assert float(poa.effective.sum()) == 0.0


def test_perez_and_hdkr_broadly_agree(clear_afternoon):
    """Different sky models, same physics: totals should be within a few percent."""
    position, ghi, dni, dhi = clear_afternoon
    perez = call_transpose(position, ghi, dni, dhi, 35.0, 180.0, model="perez")
    hdkr = call_transpose(position, ghi, dni, dhi, 35.0, 180.0, model="hdkr")
    assert perez.global_.sum() == pytest.approx(hdkr.global_.sum(), rel=0.06)


def test_unknown_transposition_model_is_rejected(clear_afternoon):
    position, ghi, dni, dhi = clear_afternoon
    with pytest.raises(ValueError, match="unknown transposition model"):
        call_transpose(position, ghi, dni, dhi, 35.0, 180.0, model="isotropic")


def test_sky_diffuse_is_zero_without_diffuse_light():
    zeros = np.zeros(4)
    sky = perez_sky_diffuse(
        30.0, zeros, np.full(4, 900.0), np.full(4, 1361.0),
        np.full(4, 30.0), np.full(4, 20.0), np.full(4, 1.15),
    )
    assert np.all(sky == 0.0)

    hdkr = hdkr_sky_diffuse(
        30.0, zeros, zeros, np.full(4, 900.0), np.full(4, 1361.0),
        np.full(4, 30.0), np.full(4, 20.0),
    )
    assert np.all(hdkr == 0.0)


def test_iam_is_unity_at_normal_incidence_and_zero_behind():
    assert float(ashrae_iam(np.array(0.0))) == pytest.approx(1.0)
    assert float(ashrae_iam(np.array(60.0))) == pytest.approx(1 - 0.05 * (2 - 1))
    assert float(ashrae_iam(np.array(95.0))) == 0.0
    # Monotonically decreasing as light arrives more obliquely.
    angles = np.array([0.0, 20.0, 40.0, 60.0, 80.0])
    assert np.all(np.diff(ashrae_iam(angles)) < 0)


def test_diffuse_effective_angles_are_physically_sensible():
    sky, ground = diffuse_effective_angles(30.0)
    assert 50.0 < sky < 60.0
    assert 70.0 < ground < 80.0
    # A flat plane sees no ground, and its sky angle is the classic 59.7 deg.
    flat_sky, flat_ground = diffuse_effective_angles(0.0)
    assert flat_sky == pytest.approx(59.7)
    assert flat_ground == pytest.approx(90.0)


def test_effective_irradiance_never_exceeds_incident(clear_afternoon):
    position, ghi, dni, dhi = clear_afternoon
    poa = call_transpose(position, ghi, dni, dhi, 35.0, 180.0)
    assert np.all(poa.effective <= poa.global_ + 1e-9)
    assert np.all(poa.effective >= 0.0)
