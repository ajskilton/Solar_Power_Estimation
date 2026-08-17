"""Solar position, air mass and extraterrestrial irradiance.

Implements the NOAA Solar Calculator algorithm (a condensed form of Meeus,
*Astronomical Algorithms*), which is accurate to well under 0.1 degrees for
years 1800-2100 -- far better than the ~25 km spatial resolution of the
reanalysis weather data it gets paired with.

All angles crossing a public function boundary are in **degrees**; internal
trigonometry works in radians. Azimuths are measured clockwise from true
north (north=0, east=90, south=180, west=270).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Total solar irradiance at 1 AU (W/m^2). Modern best estimate (Kopp & Lean
# 2011); older PV literature uses 1367, which runs ~0.4% high.
SOLAR_CONSTANT = 1361.1

# Julian Day of the Unix epoch (1970-01-01T00:00:00Z).
_JD_UNIX_EPOCH = 2440587.5

_SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class SolarPosition:
    """Sun position and related geometry for a series of instants.

    Every array has the same shape as the timestamps it was computed from.

    Attributes:
        zenith: Apparent (refraction-corrected) zenith angle, degrees.
        elevation: Apparent solar elevation above the horizon, degrees.
        azimuth: Solar azimuth clockwise from true north, degrees in [0, 360).
        declination: Solar declination, degrees.
        equation_of_time: Equation of time, minutes.
        air_mass: Kasten & Young (1989) relative air mass; ``inf`` below the
            horizon.
        extra_normal: Extraterrestrial irradiance normal to the sun's rays,
            W/m^2 (varies ~+/-3.4% over the year with earth-sun distance).
    """

    zenith: np.ndarray
    elevation: np.ndarray
    azimuth: np.ndarray
    declination: np.ndarray
    equation_of_time: np.ndarray
    air_mass: np.ndarray
    extra_normal: np.ndarray

    @property
    def cos_zenith(self) -> np.ndarray:
        """Cosine of the apparent zenith angle, floored at 0 below the horizon."""
        return np.maximum(0.0, np.cos(np.radians(self.zenith)))


def julian_day(times: np.ndarray) -> np.ndarray:
    """Convert UTC datetimes to Julian Day numbers.

    Args:
        times: ``datetime64`` array (any unit) interpreted as UTC.

    Returns:
        Float array of Julian Day numbers.
    """
    seconds = times.astype("datetime64[ns]").astype("int64") / 1e9
    return seconds / _SECONDS_PER_DAY + _JD_UNIX_EPOCH


def refraction_correction(true_elevation_deg: np.ndarray) -> np.ndarray:
    """Atmospheric refraction that lifts the apparent sun, in degrees.

    Uses the piecewise fit from the NOAA solar calculator, valid for standard
    conditions (1013.25 hPa, 10 degC). Refraction is ~0.57 deg at the horizon
    and negligible above ~40 deg elevation.
    """
    elev = np.asarray(true_elevation_deg, dtype=float)
    # tan() blows up at elev == 0 and at +/-90; the branches below never use
    # the offending values, but they are all evaluated eagerly, so clamp the
    # input to keep numpy quiet and the arithmetic finite.
    safe = np.clip(elev, -89.9, 89.9)
    tan_e = np.tan(np.radians(np.where(np.abs(safe) < 1e-6, 1e-6, safe)))

    high = 58.1 / tan_e - 0.07 / tan_e**3 + 0.000086 / tan_e**5
    low = 1735.0 + elev * (-518.2 + elev * (103.4 + elev * (-12.79 + elev * 0.711)))
    very_low = -20.772 / tan_e

    arcseconds = np.where(
        elev > 85.0,
        0.0,
        np.where(elev > 5.0, high, np.where(elev > -0.575, low, very_low)),
    )
    return arcseconds / 3600.0


def solar_position(
    times: np.ndarray,
    latitude: float,
    longitude: float,
) -> SolarPosition:
    """Compute sun position for UTC timestamps at one location.

    Args:
        times: ``datetime64`` array of UTC instants.
        latitude: Degrees north, -90 to 90.
        longitude: Degrees east, -180 to 180.

    Returns:
        A :class:`SolarPosition` with arrays shaped like ``times``.
    """
    jd = julian_day(times)
    t = (jd - 2451545.0) / 36525.0  # Julian centuries since J2000.0

    # Geometric mean longitude and anomaly of the sun.
    mean_long = np.mod(280.46646 + t * (36000.76983 + t * 0.0003032), 360.0)
    mean_anom = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    mean_anom_rad = np.radians(mean_anom)

    eccentricity = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)

    # Equation of the centre -> true longitude and anomaly.
    centre = (
        np.sin(mean_anom_rad) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + np.sin(2 * mean_anom_rad) * (0.019993 - 0.000101 * t)
        + np.sin(3 * mean_anom_rad) * 0.000289
    )
    true_long = mean_long + centre
    true_anom_rad = mean_anom_rad + np.radians(centre)

    # Earth-sun distance in astronomical units.
    radius_au = (1.000001018 * (1 - eccentricity**2)) / (
        1 + eccentricity * np.cos(true_anom_rad)
    )

    # Apparent longitude (corrected for nutation and aberration).
    omega = np.radians(125.04 - 1934.136 * t)
    apparent_long = np.radians(true_long - 0.00569 - 0.00478 * np.sin(omega))

    # Obliquity of the ecliptic.
    mean_obliquity = 23.0 + (
        26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0
    ) / 60.0
    obliquity = np.radians(mean_obliquity + 0.00256 * np.cos(omega))

    declination = np.degrees(np.arcsin(np.sin(obliquity) * np.sin(apparent_long)))

    # Equation of time, in minutes.
    y = np.tan(obliquity / 2.0) ** 2
    mean_long_rad = np.radians(mean_long)
    eqtime = 4.0 * np.degrees(
        y * np.sin(2 * mean_long_rad)
        - 2 * eccentricity * np.sin(mean_anom_rad)
        + 4 * eccentricity * y * np.sin(mean_anom_rad) * np.cos(2 * mean_long_rad)
        - 0.5 * y**2 * np.sin(4 * mean_long_rad)
        - 1.25 * eccentricity**2 * np.sin(2 * mean_anom_rad)
    )

    # Hour angle: negative before solar noon, positive after.
    minutes_utc = np.mod(jd + 0.5, 1.0) * 1440.0
    true_solar_time = np.mod(minutes_utc + eqtime + 4.0 * longitude, 1440.0)
    hour_angle = np.radians(true_solar_time / 4.0 - 180.0)

    lat_rad = np.radians(latitude)
    dec_rad = np.radians(declination)
    cos_zenith = np.clip(
        np.sin(lat_rad) * np.sin(dec_rad)
        + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(hour_angle),
        -1.0,
        1.0,
    )
    true_elevation = 90.0 - np.degrees(np.arccos(cos_zenith))
    elevation = true_elevation + refraction_correction(true_elevation)
    zenith = 90.0 - elevation

    # Azimuth measured from south, then rotated to a clockwise-from-north
    # convention. atan2 keeps the morning/afternoon branch right.
    azimuth_from_south = np.arctan2(
        np.sin(hour_angle),
        np.cos(hour_angle) * np.sin(lat_rad) - np.tan(dec_rad) * np.cos(lat_rad),
    )
    azimuth = np.mod(np.degrees(azimuth_from_south) + 180.0, 360.0)

    return SolarPosition(
        zenith=zenith,
        elevation=elevation,
        azimuth=azimuth,
        declination=declination,
        equation_of_time=eqtime,
        air_mass=relative_air_mass(zenith),
        extra_normal=SOLAR_CONSTANT / radius_au**2,
    )


def relative_air_mass(apparent_zenith_deg: np.ndarray) -> np.ndarray:
    """Kasten & Young (1989) relative air mass.

    Returns ``inf`` where the sun is below the model's validity limit
    (apparent zenith >= 96.08 deg), which callers should treat as night.
    """
    z = np.asarray(apparent_zenith_deg, dtype=float)
    valid = z < 96.07995
    # Evaluate the fit only where it is defined; the power term has a negative
    # exponent and would produce inf/NaN outside its range.
    z_safe = np.where(valid, z, 0.0)
    denom = np.cos(np.radians(z_safe)) + 0.50572 * (96.07995 - z_safe) ** -1.6364
    return np.where(valid, 1.0 / denom, np.inf)


def pressure_from_elevation(elevation_m: float) -> float:
    """Approximate station pressure (Pa) from site elevation, ISA barometric fit."""
    return 101325.0 * (1.0 - 2.25577e-5 * elevation_m) ** 5.25588


def absolute_air_mass(relative: np.ndarray, pressure_pa: float) -> np.ndarray:
    """Scale relative air mass by station pressure to get absolute air mass."""
    return relative * (pressure_pa / 101325.0)


def angle_of_incidence(
    surface_tilt_deg: float,
    surface_azimuth_deg: float,
    solar_zenith_deg: np.ndarray,
    solar_azimuth_deg: np.ndarray,
) -> np.ndarray:
    """Angle between the sun's rays and a tilted plane's normal, in degrees.

    Args:
        surface_tilt_deg: Plane tilt from horizontal (0 = flat, 90 = vertical).
        surface_azimuth_deg: Direction the plane faces, clockwise from north.
        solar_zenith_deg: Apparent solar zenith angles.
        solar_azimuth_deg: Solar azimuths clockwise from north.

    Returns:
        Angle of incidence in degrees, 0 to 180. Values above 90 mean the sun
        is behind the plane.
    """
    tilt = np.radians(surface_tilt_deg)
    surf_az = np.radians(surface_azimuth_deg)
    zenith = np.radians(np.asarray(solar_zenith_deg, dtype=float))
    sun_az = np.radians(np.asarray(solar_azimuth_deg, dtype=float))

    cos_aoi = np.clip(
        np.cos(zenith) * np.cos(tilt)
        + np.sin(zenith) * np.sin(tilt) * np.cos(sun_az - surf_az),
        -1.0,
        1.0,
    )
    return np.degrees(np.arccos(cos_aoi))
