"""Battery dispatch: how much grid import a given battery actually avoids.

The model is a greedy self-consumption controller, which is what a domestic
hybrid inverter does out of the box: use generation on site the instant it
arrives, store what is left over, and discharge to cover demand when the sun is
down. Nothing is charged from the grid and nothing is dispatched against a
price signal.

Sequential state makes this the one part of the chain that cannot be
vectorised -- the state of charge at hour *n* depends on hour *n-1* -- so it is
a plain loop over 8760 hours. That costs a few milliseconds, which is cheap
enough that sweeping twenty battery sizes stays interactive.

Deliberately not modelled, and each would move the answer:

* **Grid charging on an off-peak tariff.** This can raise grid import while
  lowering the bill, so it belongs with a tariff model rather than here.
* **DC coupling.** A DC-coupled battery can capture energy the inverter would
  otherwise clip. Assuming AC coupling is the conservative choice.
* **Degradation.** Capacity is treated as constant; a real pack loses a few
  percent per year.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Default charge and discharge power as a multiple of usable capacity. Domestic
# packs are typically specified around 0.5C -- a 5 kWh battery with a 2.5 kW
# limit -- which is rarely the binding constraint on a domestic array.
DEFAULT_C_RATE = 0.5


@dataclass(frozen=True)
class BatterySpec:
    """A battery, as far as an energy balance is concerned.

    Attributes:
        usable_capacity_kwh: Usable capacity, kWh. This is not the nameplate
            figure -- packs reserve headroom at both ends, so usable is
            typically 90-95% of nominal. Zero models no battery at all.
        max_charge_kw: Charge power limit at the AC side. ``None`` derives one
            from :data:`DEFAULT_C_RATE`.
        max_discharge_kw: Discharge power limit at the AC side. ``None``
            derives one from :data:`DEFAULT_C_RATE`.
        round_trip_efficiency: AC-to-AC efficiency of a full store-and-return
            cycle, split evenly between the two directions.
        initial_soc_frac: State of charge at the first hour, as a fraction of
            usable capacity.
    """

    usable_capacity_kwh: float = 5.0
    max_charge_kw: float | None = None
    max_discharge_kw: float | None = None
    round_trip_efficiency: float = 0.90
    initial_soc_frac: float = 0.0

    def __post_init__(self) -> None:
        if self.usable_capacity_kwh < 0:
            raise ValueError("usable_capacity_kwh cannot be negative")
        if not 0.0 < self.round_trip_efficiency <= 1.0:
            raise ValueError("round_trip_efficiency must be in (0, 1]")
        if not 0.0 <= self.initial_soc_frac <= 1.0:
            raise ValueError("initial_soc_frac must be between 0 and 1")
        for name in ("max_charge_kw", "max_discharge_kw"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when given")

    @property
    def charge_limit_kw(self) -> float:
        """Resolved charge power limit, kW."""
        if self.max_charge_kw is not None:
            return self.max_charge_kw
        return self.usable_capacity_kwh * DEFAULT_C_RATE

    @property
    def discharge_limit_kw(self) -> float:
        """Resolved discharge power limit, kW."""
        if self.max_discharge_kw is not None:
            return self.max_discharge_kw
        return self.usable_capacity_kwh * DEFAULT_C_RATE

    @property
    def one_way_efficiency(self) -> float:
        """Efficiency of a single direction, so that two of them round-trip."""
        return float(np.sqrt(self.round_trip_efficiency))


@dataclass(frozen=True)
class DispatchResult:
    """The energy balance over a year, hour by hour and in total.

    All series are in kWh per hour and aligned with the inputs.

    Attributes:
        direct_kwh: Generation consumed on site the moment it was made.
        charge_kwh: Generation diverted into the battery, measured at the AC
            side before charging losses.
        discharge_kwh: Energy delivered from the battery to the load, after
            discharging losses.
        import_kwh: Energy drawn from the grid.
        export_kwh: Generation sent to the grid.
        soc_kwh: State of charge at the end of each hour.
        generation_kwh: Annual generation, kWh.
        load_kwh: Annual consumption, kWh.
        battery_kwh: Usable capacity modelled, kWh.
        initial_soc_kwh: State of charge before the first hour, kWh.
    """

    direct_kwh: np.ndarray
    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    import_kwh: np.ndarray
    export_kwh: np.ndarray
    soc_kwh: np.ndarray

    generation_kwh: float
    load_kwh: float
    battery_kwh: float
    initial_soc_kwh: float = 0.0

    @property
    def avoided_import_kwh(self) -> float:
        """Grid energy the system displaced: direct use plus battery discharge.

        This is the headline number -- what the household does not have to buy.
        """
        return float(self.direct_kwh.sum() + self.discharge_kwh.sum())

    @property
    def total_import_kwh(self) -> float:
        """Grid energy still required over the year."""
        return float(self.import_kwh.sum())

    @property
    def total_export_kwh(self) -> float:
        """Generation sent to the grid over the year."""
        return float(self.export_kwh.sum())

    @property
    def self_sufficiency_pct(self) -> float:
        """Share of consumption met without the grid."""
        if self.load_kwh <= 0:
            return 0.0
        return 100.0 * self.avoided_import_kwh / self.load_kwh

    @property
    def self_consumption_pct(self) -> float:
        """Share of generation kept on site rather than exported.

        Round-trip storage losses count as consumed here, which is the
        convention in the literature: the energy did not reach the grid.
        """
        if self.generation_kwh <= 0:
            return 0.0
        return 100.0 * (self.generation_kwh - self.total_export_kwh) / self.generation_kwh

    @property
    def storage_loss_kwh(self) -> float:
        """Energy lost to round-trip inefficiency over the year.

        Everything charged that was neither discharged nor is still sitting in
        the pack at the end of the year.
        """
        if self.soc_kwh.size == 0:
            return 0.0
        retained = float(self.soc_kwh[-1]) - self.initial_soc_kwh
        return float(self.charge_kwh.sum() - self.discharge_kwh.sum() - retained)

    @property
    def equivalent_full_cycles(self) -> float:
        """Battery throughput expressed as full charge-discharge cycles."""
        if self.battery_kwh <= 0:
            return 0.0
        return float(self.charge_kwh.sum() / self.battery_kwh)


def dispatch(
    generation_kwh: np.ndarray,
    load_kwh: np.ndarray,
    battery: BatterySpec | None = None,
) -> DispatchResult:
    """Run a year of generation and demand through a self-consumption battery.

    Within each hour the order is: meet demand directly from generation, store
    any surplus, then cover any remaining demand from the battery. Hourly
    resolution slightly overstates direct self-consumption, because mismatch
    inside the hour averages out; with a battery in the loop the effect is
    small, since the battery would have absorbed that flicker in reality too.

    Args:
        generation_kwh: Hourly AC generation, kWh.
        load_kwh: Hourly consumption, kWh. Must be the same length.
        battery: The battery. ``None`` or zero capacity models PV alone.

    Returns:
        A :class:`DispatchResult`.

    Raises:
        ValueError: If the two series differ in length.
    """
    generation = np.asarray(generation_kwh, dtype=float)
    demand = np.asarray(load_kwh, dtype=float)
    if generation.shape != demand.shape:
        raise ValueError(
            f"generation and load must be the same length, "
            f"got {generation.size} and {demand.size}"
        )

    battery = battery or BatterySpec(usable_capacity_kwh=0.0)
    capacity = battery.usable_capacity_kwh

    direct = np.minimum(generation, demand)
    surplus = generation - direct
    deficit = demand - direct

    if capacity <= 0:
        zeros = np.zeros_like(generation)
        return DispatchResult(
            direct_kwh=direct,
            charge_kwh=zeros,
            discharge_kwh=zeros,
            import_kwh=deficit,
            export_kwh=surplus,
            soc_kwh=zeros,
            generation_kwh=float(generation.sum()),
            load_kwh=float(demand.sum()),
            battery_kwh=0.0,
        )

    eta = battery.one_way_efficiency
    charge_limit = battery.charge_limit_kw
    discharge_limit = battery.discharge_limit_kw
    initial_soc = capacity * battery.initial_soc_frac

    charge = np.zeros_like(generation)
    discharge = np.zeros_like(generation)
    soc_series = np.zeros_like(generation)
    soc = initial_soc

    for i in range(generation.size):
        spare = surplus[i]
        if spare > 0.0:
            # Room is in stored terms; converting back through the charging
            # efficiency gives the AC energy that would fill it.
            room = (capacity - soc) / eta
            taken = min(spare, charge_limit, room)
            if taken > 0.0:
                soc += taken * eta
                charge[i] = taken

        short = deficit[i]
        if short > 0.0:
            available = soc * eta
            given = min(short, discharge_limit, available)
            if given > 0.0:
                soc -= given / eta
                discharge[i] = given

        soc_series[i] = soc

    return DispatchResult(
        direct_kwh=direct,
        charge_kwh=charge,
        discharge_kwh=discharge,
        import_kwh=deficit - discharge,
        export_kwh=surplus - charge,
        soc_kwh=soc_series,
        generation_kwh=float(generation.sum()),
        load_kwh=float(demand.sum()),
        battery_kwh=capacity,
        initial_soc_kwh=initial_soc,
    )


@dataclass(frozen=True)
class SizeResult:
    """One battery size on the sizing curve.

    Attributes:
        capacity_kwh: Usable capacity modelled.
        avoided_import_kwh: Annual grid energy displaced.
        self_sufficiency_pct: Share of demand met on site.
        self_consumption_pct: Share of generation kept on site.
        export_kwh: Annual generation sent to the grid.
        equivalent_full_cycles: Annual battery throughput, in full cycles.
        marginal_kwh_per_kwh: Extra grid energy displaced per extra kWh of
            capacity, against the previous size on the curve. This is where
            diminishing returns show up, and it is the number to size on.
    """

    capacity_kwh: float
    avoided_import_kwh: float
    self_sufficiency_pct: float
    self_consumption_pct: float
    export_kwh: float
    equivalent_full_cycles: float
    marginal_kwh_per_kwh: float


def sweep_sizes(
    generation_kwh: np.ndarray,
    load_kwh: np.ndarray,
    capacities_kwh: Sequence[float] | np.ndarray,
    template: BatterySpec | None = None,
) -> list[SizeResult]:
    """Dispatch a range of battery sizes to expose diminishing returns.

    The curve from zero upwards is the actual sizing aid: avoided import rises
    steeply for the first couple of kWh, then flattens once the battery is
    routinely reaching morning with charge to spare. The knee is the point past
    which extra capacity mostly sits idle.

    Args:
        generation_kwh: Hourly AC generation, kWh.
        load_kwh: Hourly consumption, kWh.
        capacities_kwh: Usable capacities to model. Sorted ascending on output.
        template: Battery whose efficiency and power limits apply to every
            size. Its own capacity is ignored. Power limits given explicitly on
            the template are held fixed across sizes; leaving them ``None``
            scales them with capacity, which is the realistic case.

    Returns:
        One :class:`SizeResult` per capacity, ascending.
    """
    template = template or BatterySpec()
    sizes = sorted(float(c) for c in capacities_kwh)

    results: list[SizeResult] = []
    previous: tuple[float, float] | None = None

    for capacity in sizes:
        outcome = dispatch(
            generation_kwh,
            load_kwh,
            BatterySpec(
                usable_capacity_kwh=capacity,
                max_charge_kw=template.max_charge_kw,
                max_discharge_kw=template.max_discharge_kw,
                round_trip_efficiency=template.round_trip_efficiency,
                initial_soc_frac=template.initial_soc_frac,
            ),
        )
        avoided = outcome.avoided_import_kwh

        marginal = 0.0
        if previous is not None:
            last_capacity, last_avoided = previous
            step = capacity - last_capacity
            if step > 0:
                marginal = (avoided - last_avoided) / step

        results.append(
            SizeResult(
                capacity_kwh=capacity,
                avoided_import_kwh=avoided,
                self_sufficiency_pct=outcome.self_sufficiency_pct,
                self_consumption_pct=outcome.self_consumption_pct,
                export_kwh=outcome.total_export_kwh,
                equivalent_full_cycles=outcome.equivalent_full_cycles,
                marginal_kwh_per_kwh=marginal,
            )
        )
        previous = (capacity, avoided)

    return results


def suggest_capacity(
    curve: Sequence[SizeResult], marginal_threshold_kwh: float = 60.0
) -> float:
    """Pick the size past which extra capacity stops earning its keep.

    Walks up the curve and stops at the last size still displacing more than
    ``marginal_threshold_kwh`` of grid import per kWh of capacity added. The
    default is a rule of thumb, not a financial calculation: at roughly 25p per
    kWh, 60 kWh a year is about £15 of annual saving per kWh of battery, which
    is where domestic packs stop paying for themselves over a sensible life.

    Args:
        curve: Output of :func:`sweep_sizes`, ascending.
        marginal_threshold_kwh: Cut-off for annual kWh displaced per kWh added.

    Returns:
        The suggested usable capacity, kWh. Zero if even the first step is not
        worth taking.
    """
    suggestion = 0.0
    for entry in curve:
        if entry.capacity_kwh <= 0:
            continue
        if entry.marginal_kwh_per_kwh >= marginal_threshold_kwh:
            suggestion = entry.capacity_kwh
        else:
            break
    return suggestion
