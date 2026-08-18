"""Battery dispatch: the energy balance, and the physics it must not break."""

from __future__ import annotations

import numpy as np
import pytest

from solarest.battery import (
    BatterySpec,
    dispatch,
    suggest_capacity,
    sweep_sizes,
)


def daily(generation_by_hour, load_by_hour, days: int = 365):
    """Tile one representative day into a year of both series."""
    return (
        np.tile(np.asarray(generation_by_hour, dtype=float), days),
        np.tile(np.asarray(load_by_hour, dtype=float), days),
    )


def midday_sun(peak: float = 3.0) -> np.ndarray:
    """A crude bell of generation centred on noon."""
    hours = np.arange(24)
    return np.where(
        (hours >= 8) & (hours <= 16), peak * np.cos((hours - 12) / 5.0), 0.0
    ).clip(0.0)


def evening_load(total: float = 12.0) -> np.ndarray:
    """A household that uses everything after dark."""
    shape = np.zeros(24)
    shape[18:23] = 1.0
    return shape * total / shape.sum()


# --------------------------------------------------------- the balance law


def test_energy_is_conserved():
    """Nothing may be created or destroyed except by round-trip losses."""
    generation, load = daily(midday_sun(), evening_load())
    outcome = dispatch(generation, load, BatterySpec(usable_capacity_kwh=6.0))

    retained = float(outcome.soc_kwh[-1]) - outcome.initial_soc_kwh
    assert outcome.generation_kwh + outcome.total_import_kwh == pytest.approx(
        outcome.load_kwh
        + outcome.total_export_kwh
        + outcome.storage_loss_kwh
        + retained
    )


def test_losses_match_the_round_trip_rating():
    generation, load = daily(midday_sun(), evening_load())
    outcome = dispatch(
        generation, load, BatterySpec(usable_capacity_kwh=6.0, round_trip_efficiency=0.9)
    )
    throughput = float(outcome.charge_kwh.sum())
    assert outcome.storage_loss_kwh == pytest.approx(throughput * 0.1, rel=0.02)


def test_a_perfect_battery_loses_nothing():
    generation, load = daily(midday_sun(), evening_load())
    outcome = dispatch(
        generation, load, BatterySpec(usable_capacity_kwh=6.0, round_trip_efficiency=1.0)
    )
    assert outcome.storage_loss_kwh == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------- invariants


def test_the_battery_never_exceeds_its_capacity_or_goes_negative():
    generation, load = daily(midday_sun(peak=8.0), evening_load(total=2.0))
    outcome = dispatch(generation, load, BatterySpec(usable_capacity_kwh=5.0))
    assert outcome.soc_kwh.max() <= 5.0 + 1e-9
    assert outcome.soc_kwh.min() >= -1e-9


def test_nothing_is_ever_imported_and_exported_in_the_same_hour():
    generation, load = daily(midday_sun(), evening_load())
    outcome = dispatch(generation, load, BatterySpec(usable_capacity_kwh=4.0))
    assert not np.any((outcome.import_kwh > 1e-9) & (outcome.export_kwh > 1e-9))


def test_all_the_flows_are_non_negative():
    generation, load = daily(midday_sun(), evening_load())
    outcome = dispatch(generation, load, BatterySpec(usable_capacity_kwh=4.0))
    for series in (
        outcome.direct_kwh, outcome.charge_kwh, outcome.discharge_kwh,
        outcome.import_kwh, outcome.export_kwh,
    ):
        assert series.min() >= -1e-12


def test_mismatched_series_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        dispatch(np.ones(10), np.ones(11))


# --------------------------------------------------------- the no-battery case


def test_no_battery_means_direct_use_only():
    generation, load = daily(midday_sun(), evening_load())
    bare = dispatch(generation, load, None)
    zero = dispatch(generation, load, BatterySpec(usable_capacity_kwh=0.0))

    assert bare.avoided_import_kwh == pytest.approx(zero.avoided_import_kwh)
    assert bare.discharge_kwh.sum() == 0.0
    assert bare.avoided_import_kwh == pytest.approx(
        np.minimum(generation, load).sum()
    )


def test_a_household_that_uses_everything_at_night_gets_nothing_from_pv_alone():
    generation, load = daily(midday_sun(), evening_load())
    bare = dispatch(generation, load, None)
    assert bare.avoided_import_kwh == pytest.approx(0.0)
    assert bare.self_sufficiency_pct == pytest.approx(0.0)


def test_a_battery_rescues_exactly_that_household():
    """The case the whole exercise exists to answer."""
    generation, load = daily(midday_sun(), evening_load())
    stored = dispatch(generation, load, BatterySpec(usable_capacity_kwh=10.0))
    assert stored.self_sufficiency_pct > 50.0


def test_a_battery_adds_nothing_when_generation_already_lands_on_the_load():
    """No surplus to store means no work for a battery, however large."""
    shape = midday_sun()
    generation, load = daily(shape, shape)
    bare = dispatch(generation, load, None)
    stored = dispatch(generation, load, BatterySpec(usable_capacity_kwh=20.0))
    assert stored.avoided_import_kwh == pytest.approx(bare.avoided_import_kwh)
    assert stored.equivalent_full_cycles == pytest.approx(0.0)


# ------------------------------------------------------------ power limits


def test_a_slow_battery_cannot_absorb_a_sharp_peak():
    generation, load = daily(midday_sun(peak=8.0), evening_load())
    fast = dispatch(
        generation, load, BatterySpec(usable_capacity_kwh=10.0, max_charge_kw=10.0)
    )
    slow = dispatch(
        generation, load, BatterySpec(usable_capacity_kwh=10.0, max_charge_kw=0.5)
    )
    assert slow.charge_kwh.max() <= 0.5 + 1e-9
    assert slow.avoided_import_kwh < fast.avoided_import_kwh


def test_discharge_respects_its_own_limit():
    generation, load = daily(midday_sun(peak=8.0), evening_load())
    outcome = dispatch(
        generation, load, BatterySpec(usable_capacity_kwh=10.0, max_discharge_kw=0.8)
    )
    assert outcome.discharge_kwh.max() <= 0.8 + 1e-9


def test_power_limits_default_to_a_sensible_c_rate():
    spec = BatterySpec(usable_capacity_kwh=6.0)
    assert spec.charge_limit_kw == pytest.approx(3.0)
    assert spec.discharge_limit_kw == pytest.approx(3.0)


def test_a_battery_that_starts_full_can_discharge_before_it_has_charged():
    """Isolated by giving the year no sun at all: the only energy is the head start."""
    _, load = daily(midday_sun(), evening_load())
    generation = np.zeros_like(load)

    started_full = dispatch(
        generation, load, BatterySpec(usable_capacity_kwh=6.0, initial_soc_frac=1.0)
    )
    started_empty = dispatch(generation, load, BatterySpec(usable_capacity_kwh=6.0))

    assert started_full.avoided_import_kwh > 0.0
    assert started_empty.avoided_import_kwh == pytest.approx(0.0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"usable_capacity_kwh": -1.0}, "cannot be negative"),
        ({"round_trip_efficiency": 0.0}, "round_trip_efficiency"),
        ({"round_trip_efficiency": 1.2}, "round_trip_efficiency"),
        ({"initial_soc_frac": 1.5}, "initial_soc_frac"),
        ({"max_charge_kw": 0.0}, "max_charge_kw"),
        ({"max_discharge_kw": -2.0}, "max_discharge_kw"),
    ],
)
def test_impossible_batteries_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        BatterySpec(**kwargs)


# ---------------------------------------------------------------- metrics


def test_self_sufficiency_and_self_consumption_stay_in_range():
    generation, load = daily(midday_sun(), evening_load())
    outcome = dispatch(generation, load, BatterySpec(usable_capacity_kwh=5.0))
    assert 0.0 <= outcome.self_sufficiency_pct <= 100.0
    assert 0.0 <= outcome.self_consumption_pct <= 100.0


def test_self_consumption_counts_everything_that_did_not_reach_the_grid():
    generation, load = daily(midday_sun(), evening_load())
    outcome = dispatch(generation, load, BatterySpec(usable_capacity_kwh=5.0))
    expected = 100.0 * (
        outcome.generation_kwh - outcome.total_export_kwh
    ) / outcome.generation_kwh
    assert outcome.self_consumption_pct == pytest.approx(expected)


def test_cycle_count_reflects_throughput():
    """An evening household with plenty of sun should cycle roughly daily."""
    generation, load = daily(midday_sun(peak=6.0), evening_load(total=10.0))
    outcome = dispatch(generation, load, BatterySpec(usable_capacity_kwh=5.0))
    assert 200 < outcome.equivalent_full_cycles < 400


def test_an_empty_year_does_not_divide_by_zero():
    outcome = dispatch(np.zeros(24), np.zeros(24), BatterySpec(usable_capacity_kwh=5.0))
    assert outcome.self_sufficiency_pct == 0.0
    assert outcome.self_consumption_pct == 0.0


# ------------------------------------------------------------- the curve


def test_bigger_batteries_never_avoid_less():
    generation, load = daily(midday_sun(), evening_load())
    curve = sweep_sizes(generation, load, [0.0, 2.0, 5.0, 10.0, 20.0])
    avoided = [entry.avoided_import_kwh for entry in curve]
    assert avoided == sorted(avoided)


def test_the_curve_shows_diminishing_returns():
    """Each extra kWh of capacity must buy no more than the one before it.

    Early steps can tie exactly -- while the battery is small enough that every
    kWh of it is used every day, capacity is the only binding constraint -- so
    this is non-increasing rather than strictly decreasing.
    """
    generation, load = daily(midday_sun(), evening_load())
    curve = sweep_sizes(generation, load, [0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0])
    marginals = [entry.marginal_kwh_per_kwh for entry in curve[1:]]

    for earlier, later in zip(marginals, marginals[1:]):
        assert later <= earlier + 1e-9
    assert marginals[-1] < marginals[0]  # and it really does fall off


def test_the_curve_is_returned_in_ascending_order():
    generation, load = daily(midday_sun(), evening_load())
    curve = sweep_sizes(generation, load, [10.0, 2.0, 5.0])
    assert [entry.capacity_kwh for entry in curve] == [2.0, 5.0, 10.0]


def test_the_first_size_has_no_marginal_to_report():
    generation, load = daily(midday_sun(), evening_load())
    assert sweep_sizes(generation, load, [0.0, 5.0])[0].marginal_kwh_per_kwh == 0.0


def test_the_suggestion_stops_where_capacity_stops_paying():
    generation, load = daily(midday_sun(), evening_load())
    curve = sweep_sizes(generation, load, [0.0, 2.0, 4.0, 6.0, 10.0, 20.0])
    suggested = suggest_capacity(curve, marginal_threshold_kwh=60.0)

    assert suggested in {entry.capacity_kwh for entry in curve}
    beyond = [e for e in curve if e.capacity_kwh > suggested]
    if beyond:
        assert beyond[0].marginal_kwh_per_kwh < 60.0


def test_no_battery_is_suggested_when_none_would_earn_its_keep():
    """Generation already meets the load, so storage has nothing to do."""
    shape = midday_sun()
    generation, load = daily(shape, shape)
    curve = sweep_sizes(generation, load, [0.0, 5.0, 10.0])
    assert suggest_capacity(curve) == 0.0


def test_the_sweep_holds_explicit_power_limits_across_sizes():
    generation, load = daily(midday_sun(peak=8.0), evening_load())
    template = BatterySpec(usable_capacity_kwh=1.0, max_charge_kw=1.0)
    for entry in sweep_sizes(generation, load, [5.0, 15.0], template=template):
        assert entry.capacity_kwh in (5.0, 15.0)
    # With charging pinned at 1 kW, a much larger pack cannot fill any faster.
    pinned = sweep_sizes(generation, load, [5.0, 40.0], template=template)
    assert pinned[1].avoided_import_kwh < pinned[0].avoided_import_kwh * 2
