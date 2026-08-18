"""Cell temperature, DC output, losses and the inverter."""

from __future__ import annotations

import numpy as np
import pytest

from solarest.pvmodel import (
    DEFAULT_LOSSES,
    MOUNTS,
    SystemSpec,
    combine_losses,
    pvwatts_ac,
    pvwatts_dc,
    sapm_cell_temperature,
)


def test_default_losses_match_the_pvwatts_stack():
    """PVWatts' headline 14.08% default, combined multiplicatively."""
    assert combine_losses(DEFAULT_LOSSES) == pytest.approx(0.1408, abs=0.0005)


def test_losses_combine_multiplicatively_not_additively():
    combined = combine_losses({"a": 0.1, "b": 0.1})
    assert combined == pytest.approx(0.19)  # not 0.20


def test_zero_losses_combine_to_zero():
    assert combine_losses({}) == 0.0
    assert combine_losses({"a": 0.0, "b": 0.0}) == pytest.approx(0.0)


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_invalid_loss_fractions_are_rejected(bad):
    with pytest.raises(ValueError, match="must be in"):
        combine_losses({"soiling": bad})


def test_dc_output_equals_nameplate_at_standard_test_conditions():
    """1000 W/m^2 on the cells at 25 degC is the definition of nameplate."""
    dc = pvwatts_dc(np.array([1000.0]), np.array([25.0]), dc_capacity_kw=5.0, gamma_pdc=-0.0035)
    assert float(dc[0]) == pytest.approx(5.0)


def test_dc_output_is_linear_in_irradiance():
    dc = pvwatts_dc(np.array([0.0, 250.0, 500.0, 1000.0]), np.full(4, 25.0), 4.0, -0.0035)
    assert dc == pytest.approx([0.0, 1.0, 2.0, 4.0])


def test_hot_cells_lose_power_at_the_stated_coefficient():
    dc = pvwatts_dc(np.array([1000.0]), np.array([65.0]), 4.0, -0.0035)
    # 40 degC above STC at -0.35%/degC is a 14% derate.
    assert float(dc[0]) == pytest.approx(4.0 * (1 - 0.0035 * 40))


def test_cold_cells_gain_power():
    cold = pvwatts_dc(np.array([1000.0]), np.array([0.0]), 4.0, -0.0035)
    assert float(cold[0]) > 4.0


def test_dc_output_never_goes_negative_when_extremely_hot():
    dc = pvwatts_dc(np.array([1000.0]), np.array([500.0]), 4.0, -0.0047)
    assert float(dc[0]) == 0.0


def test_cell_temperature_equals_air_temperature_in_the_dark():
    cell = sapm_cell_temperature(
        np.array([0.0]), np.array([12.0]), np.array([2.0]), MOUNTS["roof_mount"]
    )
    assert float(cell[0]) == pytest.approx(12.0)


def test_cell_temperature_rises_with_sun_and_falls_with_wind():
    still = sapm_cell_temperature(
        np.array([1000.0]), np.array([20.0]), np.array([0.0]), MOUNTS["roof_mount"]
    )
    breezy = sapm_cell_temperature(
        np.array([1000.0]), np.array([20.0]), np.array([8.0]), MOUNTS["roof_mount"]
    )
    assert float(still[0]) > float(breezy[0]) > 20.0
    # A close-mounted array at full sun runs tens of degrees above ambient.
    assert 55.0 < float(still[0]) < 90.0


def test_airflow_behind_the_array_keeps_it_cooler():
    """Mounting style is only about how much heat can escape the back."""
    conditions = (np.array([1000.0]), np.array([25.0]), np.array([1.0]))
    ground = sapm_cell_temperature(*conditions, MOUNTS["ground_mount"])
    roof = sapm_cell_temperature(*conditions, MOUNTS["roof_mount"])
    integrated = sapm_cell_temperature(*conditions, MOUNTS["roof_integrated"])
    assert float(ground[0]) < float(roof[0]) < float(integrated[0])


def test_inverter_delivers_its_rating_at_full_load():
    """At the nominal operating point the curve collapses to the rated efficiency."""
    eta, ac_rating = 0.96, 5.0
    dc_at_rating = ac_rating / eta
    ac = pvwatts_ac(np.array([dc_at_rating]), ac_rating, eta)
    assert float(ac[0]) == pytest.approx(ac_rating, rel=1e-6)


def test_inverter_clips_at_its_ac_rating():
    ac = pvwatts_ac(np.array([100.0]), 5.0, 0.96)
    assert float(ac[0]) == pytest.approx(5.0)


def test_unclipped_mode_exceeds_the_rating():
    clipped = pvwatts_ac(np.array([20.0]), 5.0, 0.96, clip=True)
    unclipped = pvwatts_ac(np.array([20.0]), 5.0, 0.96, clip=False)
    assert float(clipped[0]) == pytest.approx(5.0)
    assert float(unclipped[0]) > 5.0


def test_inverter_produces_nothing_from_nothing():
    ac = pvwatts_ac(np.array([0.0, 0.0]), 5.0, 0.96)
    assert np.all(ac == 0.0)


def test_inverter_is_less_efficient_at_part_load():
    """The whole point of the curve: efficiency sags at a trickle of input."""
    rating = 5.0
    dc = np.array([0.05, 0.5, 5.0])
    ac = pvwatts_ac(dc, rating, 0.96)
    efficiency = ac / dc
    assert efficiency[0] < efficiency[1] < efficiency[2]
    assert efficiency[0] < 0.9


def test_system_spec_derives_inverter_rating_and_losses():
    spec = SystemSpec(dc_capacity_kw=6.0, dc_ac_ratio=1.2)
    assert spec.ac_capacity_kw == pytest.approx(5.0)
    assert spec.total_loss_fraction == pytest.approx(0.1408, abs=0.0005)
    assert spec.gamma_pdc == pytest.approx(-0.0035)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"dc_capacity_kw": 0}, "dc_capacity_kw must be positive"),
        ({"dc_capacity_kw": -1}, "dc_capacity_kw must be positive"),
        ({"tilt_deg": 91}, "tilt_deg must be between"),
        ({"tilt_deg": -1}, "tilt_deg must be between"),
        ({"mount": "hot air balloon"}, "unknown mount"),
        ({"module_type": "unobtainium"}, "unknown module_type"),
        ({"dc_ac_ratio": 0}, "dc_ac_ratio must be positive"),
        ({"inverter_efficiency": 1.4}, "inverter_efficiency must be"),
        ({"albedo": 1.4}, "albedo must be between"),
    ],
)
def test_invalid_system_specs_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SystemSpec(**kwargs)


def test_module_types_are_ordered_by_temperature_sensitivity():
    standard = SystemSpec(module_type="standard").gamma_pdc
    premium = SystemSpec(module_type="premium").gamma_pdc
    thin_film = SystemSpec(module_type="thin_film").gamma_pdc
    assert standard < premium < thin_film < 0
