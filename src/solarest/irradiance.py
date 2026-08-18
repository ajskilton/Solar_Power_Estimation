"""Transposition of horizontal irradiance onto a tilted plane.

Open-Meteo supplies global horizontal (GHI), direct normal (DNI) and diffuse
horizontal (DHI) irradiance, so no decomposition model is needed -- only
transposition onto the plane of the array (POA), plus reflection losses at
the module surface.

Two sky-diffuse models are provided:

* :func:`perez_sky_diffuse` -- Perez et al. (1990), the accuracy benchmark and
  the default here and in NREL's PVWatts/SAM.
* :func:`hdkr_sky_diffuse` -- Hay-Davies-Klucher-Reindl, simpler and a useful
  cross-check (Duffie & Beckman eq. 2.16.14).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .solarpos import angle_of_incidence

# Perez (1990) "all sites composite" F-coefficients, indexed by sky clearness
# bin. Columns are F11, F12, F13, F21, F22, F23.
_PEREZ_BINS = np.array([1.065, 1.230, 1.500, 1.950, 2.800, 4.500, 6.200])
_PEREZ_F = np.array(
    [
        [-0.0083117, 0.5877285, -0.0620636, -0.0596012, 0.0721249, -0.0220216],
        [0.1299457, 0.6825954, -0.1513752, -0.0189325, 0.0659650, -0.0288748],
        [0.3296958, 0.4868735, -0.2210958, 0.0554140, -0.0639588, -0.0260542],
        [0.5682053, 0.1874525, -0.2951290, 0.1088631, -0.1519229, -0.0139754],
        [0.8730280, -0.3920403, -0.3616149, 0.2255647, -0.4620442, 0.0012448],
        [1.1326077, -1.2367284, -0.4118494, 0.2877813, -0.8230357, 0.0558651],
        [1.0601591, -1.5999137, -0.3589221, 0.2642124, -1.1272340, 0.1310694],
        [0.6777470, -0.3272588, -0.2504286, 0.1561313, -1.3765031, 0.2506212],
    ]
)

# Perez circumsolar term divides by cos(zenith); below this elevation the
# quotient is capped, as in the original paper.
_PEREZ_COS_ZENITH_FLOOR = np.cos(np.radians(85.0))


@dataclass(frozen=True)
class PlaneOfArray:
    """Irradiance components incident on the tilted plane, W/m^2.

    Attributes:
        beam: Direct beam on the plane.
        sky_diffuse: Diffuse from the sky dome.
        ground_diffuse: Diffuse reflected off the ground.
        global_: Sum of the three components, before reflection losses.
        effective: Global POA after incidence-angle (reflection) losses --
            this is what actually reaches the cells.
        aoi: Angle of incidence, degrees.
    """

    beam: np.ndarray
    sky_diffuse: np.ndarray
    ground_diffuse: np.ndarray
    global_: np.ndarray
    effective: np.ndarray
    aoi: np.ndarray


def perez_sky_diffuse(
    surface_tilt_deg: float,
    dhi: np.ndarray,
    dni: np.ndarray,
    extra_normal: np.ndarray,
    solar_zenith_deg: np.ndarray,
    aoi_deg: np.ndarray,
    air_mass: np.ndarray,
) -> np.ndarray:
    """Sky-diffuse irradiance on a tilted plane, Perez et al. (1990).

    Splits the sky dome into an isotropic background, a circumsolar disc and a
    horizon brightening band, weighted by empirical clearness and brightness
    indices.

    Args:
        surface_tilt_deg: Plane tilt from horizontal.
        dhi: Diffuse horizontal irradiance, W/m^2.
        dni: Direct normal irradiance, W/m^2.
        extra_normal: Extraterrestrial normal irradiance, W/m^2.
        solar_zenith_deg: Apparent solar zenith, degrees.
        aoi_deg: Angle of incidence on the plane, degrees.
        air_mass: Relative air mass (``inf`` at night is handled).

    Returns:
        Sky-diffuse irradiance on the plane, W/m^2.
    """
    tilt = np.radians(surface_tilt_deg)
    zenith_rad = np.radians(np.asarray(solar_zenith_deg, dtype=float))
    dhi = np.asarray(dhi, dtype=float)
    dni = np.asarray(dni, dtype=float)

    # The model is only defined where there is diffuse light to redistribute.
    lit = dhi > 0.0
    dhi_safe = np.where(lit, dhi, 1.0)

    # Sky clearness epsilon.
    kappa_z3 = 1.041 * zenith_rad**3
    epsilon = ((dhi_safe + dni) / dhi_safe + kappa_z3) / (1.0 + kappa_z3)

    # Sky brightness delta. Air mass is inf below the horizon; those samples
    # are masked out by `lit`/`usable` before the result is used.
    finite_am = np.where(np.isfinite(air_mass), air_mass, 0.0)
    delta = finite_am * dhi_safe / extra_normal

    bin_index = np.searchsorted(_PEREZ_BINS, epsilon, side="right")
    coeffs = _PEREZ_F[np.clip(bin_index, 0, len(_PEREZ_F) - 1)]
    f11, f12, f13, f21, f22, f23 = (coeffs[..., i] for i in range(6))

    f1 = np.maximum(0.0, f11 + f12 * delta + f13 * zenith_rad)
    f2 = f21 + f22 * delta + f23 * zenith_rad

    # Circumsolar geometric factor a/b, with b floored per the paper so the
    # term stays bounded at sunrise and sunset.
    a = np.maximum(0.0, np.cos(np.radians(np.asarray(aoi_deg, dtype=float))))
    b = np.maximum(_PEREZ_COS_ZENITH_FLOOR, np.cos(zenith_rad))

    sky = dhi_safe * (
        (1.0 - f1) * (1.0 + np.cos(tilt)) / 2.0 + f1 * a / b + f2 * np.sin(tilt)
    )
    usable = lit & np.isfinite(air_mass)
    return np.where(usable, np.maximum(0.0, sky), 0.0)


def hdkr_sky_diffuse(
    surface_tilt_deg: float,
    ghi: np.ndarray,
    dhi: np.ndarray,
    dni: np.ndarray,
    extra_normal: np.ndarray,
    solar_zenith_deg: np.ndarray,
    aoi_deg: np.ndarray,
) -> np.ndarray:
    """Sky-diffuse irradiance on a tilted plane, HDKR model.

    Anisotropic circumsolar weighting (Hay & Davies) plus horizon brightening
    (Klucher/Reindl). Cheaper than Perez and typically within a few percent.
    """
    tilt = np.radians(surface_tilt_deg)
    zenith_rad = np.radians(np.asarray(solar_zenith_deg, dtype=float))
    cos_zenith = np.maximum(0.0, np.cos(zenith_rad))
    ghi = np.asarray(ghi, dtype=float)
    dhi = np.asarray(dhi, dtype=float)
    dni = np.asarray(dni, dtype=float)

    lit = (dhi > 0.0) & (ghi > 0.0)
    ghi_safe = np.where(lit, ghi, 1.0)

    anisotropy = np.clip(dni / extra_normal, 0.0, 1.0)
    horizon_factor = np.sqrt(np.clip(dni * cos_zenith / ghi_safe, 0.0, 1.0))

    # Beam geometric ratio, floored the same way as Perez to stay bounded.
    cos_aoi = np.maximum(0.0, np.cos(np.radians(np.asarray(aoi_deg, dtype=float))))
    rb = cos_aoi / np.maximum(_PEREZ_COS_ZENITH_FLOOR, cos_zenith)

    isotropic = (
        (1.0 - anisotropy)
        * (1.0 + np.cos(tilt))
        / 2.0
        * (1.0 + horizon_factor * np.sin(tilt / 2.0) ** 3)
    )
    sky = dhi * (isotropic + anisotropy * rb)
    return np.where(lit, np.maximum(0.0, sky), 0.0)


def ashrae_iam(aoi_deg: np.ndarray, b0: float = 0.05) -> np.ndarray:
    """ASHRAE incidence-angle modifier for glazing reflection losses.

    Returns the fraction of irradiance transmitted relative to normal
    incidence. Zero for light arriving from behind the plane.
    """
    aoi = np.asarray(aoi_deg, dtype=float)
    front = aoi < 90.0
    cos_aoi = np.where(front, np.cos(np.radians(np.where(front, aoi, 0.0))), 1.0)
    iam = 1.0 - b0 * (1.0 / cos_aoi - 1.0)
    return np.where(front, np.clip(iam, 0.0, 1.0), 0.0)


def diffuse_effective_angles(surface_tilt_deg: float) -> tuple[float, float]:
    """Brandemuehl & Beckman effective incidence angles for diffuse light.

    Sky and ground diffuse arrive from every direction, so their reflection
    loss is evaluated at a single representative angle that depends only on
    tilt (Duffie & Beckman eq. 5.4.2).

    Returns:
        ``(sky_angle_deg, ground_angle_deg)``.
    """
    beta = float(surface_tilt_deg)
    sky = 59.7 - 0.1388 * beta + 0.001497 * beta**2
    ground = 90.0 - 0.5788 * beta + 0.002693 * beta**2
    return sky, ground


def transpose(
    surface_tilt_deg: float,
    surface_azimuth_deg: float,
    ghi: np.ndarray,
    dni: np.ndarray,
    dhi: np.ndarray,
    solar_zenith_deg: np.ndarray,
    solar_azimuth_deg: np.ndarray,
    extra_normal: np.ndarray,
    air_mass: np.ndarray,
    albedo: float = 0.20,
    model: str = "perez",
    iam_b0: float = 0.05,
) -> PlaneOfArray:
    """Project horizontal irradiance onto the array plane and apply IAM losses.

    Args:
        surface_tilt_deg: Array tilt from horizontal, degrees.
        surface_azimuth_deg: Array azimuth clockwise from north, degrees.
        ghi: Global horizontal irradiance, W/m^2.
        dni: Direct normal irradiance, W/m^2.
        dhi: Diffuse horizontal irradiance, W/m^2.
        solar_zenith_deg: Apparent solar zenith, degrees.
        solar_azimuth_deg: Solar azimuth clockwise from north, degrees.
        extra_normal: Extraterrestrial normal irradiance, W/m^2.
        air_mass: Relative air mass.
        albedo: Ground reflectance, 0-1.
        model: ``"perez"`` or ``"hdkr"``.
        iam_b0: ASHRAE incidence-angle-modifier coefficient.

    Returns:
        A :class:`PlaneOfArray` with the component breakdown.
    """
    aoi = angle_of_incidence(
        surface_tilt_deg, surface_azimuth_deg, solar_zenith_deg, solar_azimuth_deg
    )

    # Beam: only counted when the sun is above the horizon *and* in front of
    # the plane.
    sun_up = np.asarray(solar_zenith_deg, dtype=float) < 90.0
    cos_aoi = np.cos(np.radians(aoi))
    beam = np.where(sun_up & (cos_aoi > 0.0), np.asarray(dni, dtype=float) * cos_aoi, 0.0)

    if model == "perez":
        sky = perez_sky_diffuse(
            surface_tilt_deg, dhi, dni, extra_normal, solar_zenith_deg, aoi, air_mass
        )
    elif model == "hdkr":
        sky = hdkr_sky_diffuse(
            surface_tilt_deg, ghi, dhi, dni, extra_normal, solar_zenith_deg, aoi
        )
    else:
        raise ValueError(f"unknown transposition model: {model!r}")

    tilt = np.radians(surface_tilt_deg)
    ground = np.asarray(ghi, dtype=float) * albedo * (1.0 - np.cos(tilt)) / 2.0

    total = beam + sky + ground

    sky_angle, ground_angle = diffuse_effective_angles(surface_tilt_deg)
    effective = (
        beam * ashrae_iam(aoi, iam_b0)
        + sky * float(ashrae_iam(np.array(sky_angle), iam_b0))
        + ground * float(ashrae_iam(np.array(ground_angle), iam_b0))
    )

    return PlaneOfArray(
        beam=beam,
        sky_diffuse=sky,
        ground_diffuse=ground,
        global_=total,
        effective=effective,
        aoi=aoi,
    )
