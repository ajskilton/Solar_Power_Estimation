"""Orientation search."""

from __future__ import annotations

import asyncio

import pytest

from solarest.optimize import (
    equator_facing_azimuth,
    optimise_orientation,
    rule_of_thumb_tilt,
    total_energy,
)
from solarest.pvmodel import SystemSpec
from solarest.simulate import precompute
from solarest.weather import SyntheticClearSky


def site_for(latitude: float, longitude: float = 0.0):
    weather = asyncio.run(SyntheticClearSky().fetch(latitude, longitude, [2023]))
    return precompute(weather, utc_offset_seconds=0)


@pytest.fixture(scope="module")
def northern_site():
    return site_for(51.5, -0.13)


def test_rule_of_thumb_tilt_sits_below_the_latitude():
    """Optimal tilt undershoots latitude: summer sun is stronger and longer."""
    for latitude in (30.0, 45.0, 55.0):
        assert 0 < rule_of_thumb_tilt(latitude) < latitude


def test_rule_of_thumb_tilt_is_clamped_to_buildable_angles():
    assert rule_of_thumb_tilt(0.0) == pytest.approx(5.0)
    assert rule_of_thumb_tilt(85.0) == pytest.approx(60.0)
    assert rule_of_thumb_tilt(-51.5) == rule_of_thumb_tilt(51.5)


def test_arrays_face_the_equator():
    assert equator_facing_azimuth(51.5) == 180.0
    assert equator_facing_azimuth(0.0) == 180.0
    assert equator_facing_azimuth(-33.9) == 0.0


def test_search_finds_an_equator_facing_orientation(northern_site):
    result = optimise_orientation(northern_site, SystemSpec(tilt_deg=10.0, azimuth_deg=90.0))
    assert 150.0 <= result.best_azimuth_deg <= 210.0
    assert 20.0 <= result.best_tilt_deg <= 60.0


def test_search_beats_a_deliberately_poor_starting_point(northern_site):
    poor = SystemSpec(tilt_deg=80.0, azimuth_deg=45.0)  # steep, facing north-east
    result = optimise_orientation(northern_site, poor)
    assert result.best_annual_kwh > result.baseline_annual_kwh
    assert result.improvement_pct > 10.0


def test_a_good_starting_point_leaves_little_on_the_table(northern_site):
    good = SystemSpec(tilt_deg=40.0, azimuth_deg=180.0)
    result = optimise_orientation(northern_site, good)
    assert result.improvement_pct < 3.0


def test_southern_hemisphere_arrays_are_pointed_north():
    result = optimise_orientation(site_for(-33.9, 151.2), SystemSpec(tilt_deg=20.0, azimuth_deg=180.0))
    assert result.best_azimuth_deg < 60.0 or result.best_azimuth_deg > 300.0


def test_baseline_matches_a_direct_simulation(northern_site):
    """The reported baseline must be the same number the estimator would give."""
    spec = SystemSpec(tilt_deg=35.0, azimuth_deg=180.0)
    result = optimise_orientation(northern_site, spec, refine=False)
    direct = northern_site.run(spec).ac_kw.sum()
    # One year of hourly data, normalised by elapsed years.
    assert result.baseline_annual_kwh == pytest.approx(direct / (8760 / 8765.82), rel=1e-6)


def test_the_reported_surface_covers_the_searched_grid(northern_site):
    result = optimise_orientation(
        northern_site, SystemSpec(), coarse_tilt_step=15.0, coarse_azimuth_step=25.0
    )
    tilts = {candidate.tilt_deg for candidate in result.surface}
    assert 0.0 in tilts and 90.0 in tilts
    assert all(candidate.annual_kwh >= 0 for candidate in result.surface)


def test_a_flat_array_is_indifferent_to_azimuth(northern_site):
    """A physical invariant: at zero tilt every direction is the same panel."""
    result = optimise_orientation(
        northern_site, SystemSpec(), coarse_tilt_step=30.0, refine=False
    )
    flat = [c.annual_kwh for c in result.surface if c.tilt_deg == 0.0]
    assert len(flat) > 1
    assert max(flat) == pytest.approx(min(flat), rel=1e-9)


def test_refinement_never_makes_the_answer_worse(northern_site):
    spec = SystemSpec(tilt_deg=10.0, azimuth_deg=120.0)
    coarse = optimise_orientation(northern_site, spec, refine=False)
    refined = optimise_orientation(northern_site, spec, refine=True)
    assert refined.best_annual_kwh >= coarse.best_annual_kwh - 1e-9


def test_a_custom_objective_is_honoured(northern_site):
    """Maximising winter output should tilt the array steeper than annual would."""

    def winter_energy(chain):
        # The first and last 1500 hours of the year are the winter shoulders.
        return float(chain.ac_kw[:1500].sum() + chain.ac_kw[-1500:].sum())

    annual = optimise_orientation(northern_site, SystemSpec(), objective=total_energy)
    winter = optimise_orientation(northern_site, SystemSpec(), objective=winter_energy)
    assert winter.best_tilt_deg > annual.best_tilt_deg
