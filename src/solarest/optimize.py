"""Find the array orientation that maximises output at a site.

A coarse grid over tilt and azimuth followed by a local refinement. Because
sun position and irradiance reconciliation are shared across candidates (see
:class:`~solarest.simulate.SitePrecompute`), each extra candidate costs only a
transposition and a power-chain pass.

The objective is pluggable. Today it is total annual AC energy; once household
consumption data is in play, the interesting objective becomes self-consumed
energy or bill savings, which slots in here without touching the search.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from .pvmodel import PowerChain, SystemSpec
from .simulate import SitePrecompute

Objective = Callable[[PowerChain], float]

# Mean length of a Gregorian year in hours (365.2425 days).
HOURS_PER_GREGORIAN_YEAR = 8765.82


def total_energy(chain: PowerChain) -> float:
    """Total AC energy over the modelled period, kWh. The default objective."""
    return float(chain.ac_kw.sum())


@dataclass(frozen=True)
class Candidate:
    """One evaluated orientation."""

    tilt_deg: float
    azimuth_deg: float
    score: float
    annual_kwh: float


@dataclass(frozen=True)
class OrientationResult:
    """Outcome of an orientation search.

    Attributes:
        best_tilt_deg: Tilt of the best orientation found.
        best_azimuth_deg: Azimuth of the best orientation found.
        best_annual_kwh: Mean annual AC energy at that orientation.
        baseline_annual_kwh: Mean annual AC energy at the requested
            orientation, for comparison.
        improvement_pct: Percentage gain over the requested orientation.
        surface: Coarse-grid samples, useful for plotting a sensitivity map.
    """

    best_tilt_deg: float
    best_azimuth_deg: float
    best_annual_kwh: float
    baseline_annual_kwh: float
    improvement_pct: float
    surface: list[Candidate]


def rule_of_thumb_tilt(latitude: float) -> float:
    """A reasonable fixed tilt for a latitude, before any site-specific search.

    Optimal tilt sits below latitude because summer sun is both stronger and
    longer-lasting. The fit is the widely used ``0.76*|lat| + 3.1``, clamped to
    a buildable range.

    Args:
        latitude: Degrees north (sign is ignored).

    Returns:
        Tilt in degrees, 5 to 60.
    """
    return float(np.clip(0.76 * abs(latitude) + 3.1, 5.0, 60.0))


def equator_facing_azimuth(latitude: float) -> float:
    """The azimuth an array should face: south above the equator, north below."""
    return 180.0 if latitude >= 0 else 0.0


def optimise_orientation(
    site: SitePrecompute,
    system: SystemSpec,
    objective: Objective = total_energy,
    coarse_tilt_step: float = 10.0,
    coarse_azimuth_step: float = 20.0,
    refine: bool = True,
) -> OrientationResult:
    """Search tilt and azimuth for the best score at a site.

    Args:
        site: Precomputed site geometry and irradiance.
        system: The system to vary; every field except tilt and azimuth is
            held fixed.
        objective: Scores a :class:`~solarest.pvmodel.PowerChain`; higher wins.
        coarse_tilt_step: Coarse grid spacing in tilt, degrees.
        coarse_azimuth_step: Coarse grid spacing in azimuth, degrees.
        refine: Run a fine local search around the coarse winner.

    Returns:
        An :class:`OrientationResult`.
    """
    # Normalise by elapsed time rather than by distinct calendar years: a
    # record shifted into local standard time clips a few hours off each end
    # and so touches one more calendar year than it actually spans.
    years = max(len(site.local_times) / HOURS_PER_GREGORIAN_YEAR, 1e-9)

    def evaluate(tilt: float, azimuth: float) -> Candidate:
        chain = site.run(replace(system, tilt_deg=float(tilt), azimuth_deg=float(azimuth)))
        return Candidate(
            tilt_deg=float(tilt),
            azimuth_deg=float(azimuth),
            score=objective(chain),
            annual_kwh=float(chain.ac_kw.sum()) / years,
        )

    # Search the hemisphere the sun actually occupies, plus a margin: due east
    # through due west via the equator-facing side.
    centre = equator_facing_azimuth(site.weather.latitude)
    azimuths = np.arange(centre - 100.0, centre + 100.0 + 1e-9, coarse_azimuth_step)
    tilts = np.arange(0.0, 90.0 + 1e-9, coarse_tilt_step)

    surface = [evaluate(t, np.mod(a, 360.0)) for t in tilts for a in azimuths]
    best = max(surface, key=lambda c: c.score)

    if refine:
        fine_tilts = np.clip(
            np.arange(best.tilt_deg - coarse_tilt_step, best.tilt_deg + coarse_tilt_step + 1e-9, 2.0),
            0.0,
            90.0,
        )
        fine_azimuths = np.arange(
            best.azimuth_deg - coarse_azimuth_step,
            best.azimuth_deg + coarse_azimuth_step + 1e-9,
            4.0,
        )
        refined = [
            evaluate(t, np.mod(a, 360.0))
            for t in np.unique(fine_tilts)
            for a in fine_azimuths
        ]
        best = max([best, *refined], key=lambda c: c.score)

    baseline = evaluate(system.tilt_deg, system.azimuth_deg)
    improvement = (
        (best.annual_kwh - baseline.annual_kwh) / baseline.annual_kwh * 100.0
        if baseline.annual_kwh > 0
        else 0.0
    )

    return OrientationResult(
        best_tilt_deg=best.tilt_deg,
        best_azimuth_deg=best.azimuth_deg,
        best_annual_kwh=best.annual_kwh,
        baseline_annual_kwh=baseline.annual_kwh,
        improvement_pct=improvement,
        surface=surface,
    )
