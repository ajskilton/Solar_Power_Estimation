"""Solar geometry, checked against published reference positions."""

from __future__ import annotations

import numpy as np
import pytest

from solarest.solarpos import (
    SOLAR_CONSTANT,
    angle_of_incidence,
    julian_day,
    refraction_correction,
    relative_air_mass,
    solar_position,
)


def at(*timestamps: str) -> np.ndarray:
    return np.array(list(timestamps), dtype="datetime64[s]")


def test_julian_day_matches_known_epoch():
    # J2000.0 is 2000-01-01T12:00:00 TT, JD 2451545.0.
    assert julian_day(at("2000-01-01T12:00:00"))[0] == pytest.approx(2451545.0)
    assert julian_day(at("1970-01-01T00:00:00"))[0] == pytest.approx(2440587.5)


# Reference apparent elevation and azimuth from the NREL Solar Position
# Algorithm (via pvlib 0.15.2 `spa_python`), which is an independent and more
# accurate implementation than the condensed NOAA algorithm under test. Across
# a full year at eight spread-out sites the two agree to within 0.017 deg of
# elevation and 0.06 deg of azimuth.
SPA_REFERENCE = [
    ("2023-06-21T12:00:00", 51.5074, -0.1278, 61.9355, 178.8855),  # London, solstice
    ("2023-12-21T12:00:00", 51.5074, -0.1278, 15.1139, 180.3692),  # London, midwinter
    ("2023-09-23T19:00:00", 37.7749, -122.4194, 49.4358, 155.7293),  # San Francisco
    ("2023-01-15T02:00:00", -33.8688, 151.2093, 77.2953, 4.5570),  # Sydney, summer
    ("2023-03-20T09:30:00", -1.2921, 36.8219, 87.2107, 66.8953),  # Nairobi, near zenith
    ("2023-11-05T21:15:00", 64.1466, -21.9426, -27.4243, 291.5571),  # Reykjavik, night
]


@pytest.mark.parametrize("timestamp, latitude, longitude, elevation, azimuth", SPA_REFERENCE)
def test_position_matches_nrel_spa(timestamp, latitude, longitude, elevation, azimuth):
    position = solar_position(at(timestamp), latitude, longitude)
    assert position.elevation[0] == pytest.approx(elevation, abs=0.02)
    # Azimuth is ill-conditioned when the sun is near the zenith: a hair of
    # elevation error swings it a long way, so Nairobi at 87 deg gets slack.
    tolerance = 1.0 if position.elevation[0] > 85.0 else 0.07
    assert position.azimuth[0] == pytest.approx(azimuth, abs=tolerance)


def test_declination_tracks_the_seasons():
    times = at(
        "2023-03-20T12:00:00",
        "2023-06-21T12:00:00",
        "2023-09-23T12:00:00",
        "2023-12-21T12:00:00",
    )
    declination = solar_position(times, 0.0, 0.0).declination
    assert declination[0] == pytest.approx(0.0, abs=0.5)
    assert declination[1] == pytest.approx(23.44, abs=0.05)
    assert declination[2] == pytest.approx(0.0, abs=0.5)
    assert declination[3] == pytest.approx(-23.44, abs=0.05)


def test_declination_never_exceeds_axial_tilt():
    times = np.arange(
        np.datetime64("2023-01-01T00:00:00"),
        np.datetime64("2024-01-01T00:00:00"),
        np.timedelta64(6, "h"),
    )
    declination = solar_position(times, 0.0, 0.0).declination
    assert np.abs(declination).max() < 23.5


def test_equation_of_time_stays_within_known_bounds():
    times = np.arange(
        np.datetime64("2023-01-01T12:00:00"),
        np.datetime64("2024-01-01T12:00:00"),
        np.timedelta64(1, "D"),
    )
    eot = solar_position(times, 0.0, 0.0).equation_of_time
    # The analemma runs from about -14.2 to +16.4 minutes.
    assert eot.min() == pytest.approx(-14.2, abs=0.4)
    assert eot.max() == pytest.approx(16.4, abs=0.4)


def test_sun_is_due_south_at_local_solar_noon():
    # Greenwich meridian: solar noon is 12:00 UTC corrected by the equation of
    # time, so search for the moment of peak elevation on a single day.
    times = np.arange(
        np.datetime64("2023-04-15T10:00:00"),
        np.datetime64("2023-04-15T14:00:00"),
        np.timedelta64(1, "m"),
    )
    position = solar_position(times, 51.5, 0.0)
    peak = int(np.argmax(position.elevation))
    assert position.azimuth[peak] == pytest.approx(180.0, abs=0.2)


def test_azimuth_sweeps_east_to_west_through_the_day():
    times = np.arange(
        np.datetime64("2023-06-21T06:00:00"),
        np.datetime64("2023-06-21T18:00:00"),
        np.timedelta64(1, "h"),
    )
    azimuth = solar_position(times, 51.5, 0.0).azimuth
    assert np.all(np.diff(azimuth) > 0)  # monotonic, no wraparound mid-day
    assert azimuth[0] < 120.0  # morning: east
    assert azimuth[-1] > 240.0  # evening: west


def test_refraction_lifts_the_sun_most_at_the_horizon():
    assert refraction_correction(np.array(0.0)) == pytest.approx(0.48, abs=0.05)
    assert refraction_correction(np.array(10.0)) == pytest.approx(0.09, abs=0.02)
    assert float(refraction_correction(np.array(89.0))) == 0.0
    # Monotonic: the lower the sun, the more it is lifted.
    elevations = np.array([0.0, 5.0, 20.0, 45.0, 80.0])
    assert np.all(np.diff(refraction_correction(elevations)) < 0)


def test_air_mass_is_one_overhead_and_large_at_the_horizon():
    assert relative_air_mass(np.array(0.0)) == pytest.approx(1.0, abs=1e-3)
    assert relative_air_mass(np.array(60.0)) == pytest.approx(2.0, abs=0.01)
    assert relative_air_mass(np.array(90.0)) == pytest.approx(38.0, abs=1.0)
    assert np.isinf(relative_air_mass(np.array(97.0)))


def test_extraterrestrial_irradiance_peaks_at_perihelion():
    times = at("2023-01-03T00:00:00", "2023-07-04T00:00:00")
    extra = solar_position(times, 0.0, 0.0).extra_normal
    assert extra[0] > extra[1]  # earth is closest in early January
    assert extra[0] == pytest.approx(SOLAR_CONSTANT * 1.034, rel=0.002)
    assert extra[1] == pytest.approx(SOLAR_CONSTANT * 0.967, rel=0.002)


def test_angle_of_incidence_on_a_horizontal_plane_equals_zenith():
    zenith = np.array([0.0, 15.0, 40.0, 75.0])
    aoi = angle_of_incidence(0.0, 180.0, zenith, np.array([90.0, 120.0, 180.0, 260.0]))
    assert aoi == pytest.approx(zenith)


def test_angle_of_incidence_is_zero_when_the_plane_faces_the_sun():
    # A plane tilted 40 degrees facing south, with the sun 40 degrees from
    # the zenith due south, is exactly normal to the beam.
    aoi = angle_of_incidence(40.0, 180.0, np.array(40.0), np.array(180.0))
    # arccos has an infinite derivative at 1, so a rounding-level error in the
    # cosine shows up as a micro-degree here.
    assert float(aoi) == pytest.approx(0.0, abs=1e-4)


def test_angle_of_incidence_exceeds_90_when_the_sun_is_behind():
    aoi = angle_of_incidence(60.0, 180.0, np.array(50.0), np.array(0.0))
    assert float(aoi) > 90.0
