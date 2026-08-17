"""PV array performance model: irradiance and weather in, AC power out.

The chain follows NREL's PVWatts v5 (Dobos, NREL/TP-6A20-62641), which is the
de-facto reference for system-level annual yield estimates:

    plane-of-array irradiance
      -> cell temperature (Sandia array performance model)
      -> DC power (linear in irradiance, linear temperature derate)
      -> system losses (soiling, mismatch, wiring, ...)
      -> inverter efficiency curve and AC clipping

Nameplate DC capacity is given in kW at standard test conditions (1000 W/m^2,
25 degC cell), so no module area or count is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .irradiance import PlaneOfArray


@dataclass(frozen=True)
class MountThermal:
    """Sandia array performance model thermal coefficients.

    ``a`` and ``b`` set the module temperature rise and its wind sensitivity;
    ``delta_t`` is the extra cell-above-back-surface rise at 1000 W/m^2.
    """

    a: float
    b: float
    delta_t: float
    label: str


# Sandia coefficients by mounting style (King et al. 2004, table 4). Airflow
# behind the modules is what separates them: a ground-mounted rack runs
# coolest, a roof-integrated array runs hottest.
MOUNTS: dict[str, MountThermal] = {
    "ground_mount": MountThermal(-3.56, -0.0750, 3.0, "Open rack (ground mount)"),
    "roof_mount": MountThermal(-2.98, -0.0471, 1.0, "Roof mount, standoff"),
    "roof_integrated": MountThermal(-2.81, -0.0455, 0.0, "Roof-integrated / no gap"),
}

# PVWatts default loss breakdown, as fractions. Shading here means generic
# near-field shading; a real horizon or obstruction survey would replace it.
DEFAULT_LOSSES: dict[str, float] = {
    "soiling": 0.02,
    "shading": 0.03,
    "snow": 0.0,
    "mismatch": 0.02,
    "wiring": 0.02,
    "connections": 0.005,
    "light_induced_degradation": 0.015,
    "nameplate_rating": 0.01,
    "availability": 0.03,
}

# Module technology presets: temperature coefficient of maximum power, per degC.
MODULE_TYPES: dict[str, float] = {
    "standard": -0.0047,
    "premium": -0.0035,
    "thin_film": -0.0020,
}

# Reference efficiency the PVWatts inverter curve was fitted at.
_INVERTER_ETA_REF = 0.9637


def combine_losses(losses: dict[str, float]) -> float:
    """Combine individual loss fractions multiplicatively.

    PVWatts states its 14.08% default as a stack of independent derates, not a
    sum, so ``0.02`` soiling and ``0.03`` shading give ``1 - 0.98*0.97``.

    Args:
        losses: Mapping of loss name to fraction in [0, 1).

    Returns:
        The combined loss as a single fraction in [0, 1).
    """
    remaining = 1.0
    for name, value in losses.items():
        if not 0.0 <= value < 1.0:
            raise ValueError(f"loss {name!r} must be in [0, 1), got {value}")
        remaining *= 1.0 - value
    return 1.0 - remaining


def sapm_cell_temperature(
    poa_global: np.ndarray,
    air_temp_c: np.ndarray,
    wind_speed_ms: np.ndarray,
    mount: MountThermal,
) -> np.ndarray:
    """Cell temperature from the Sandia array performance model, degC.

    Args:
        poa_global: Global plane-of-array irradiance, W/m^2.
        air_temp_c: Ambient air temperature at 2 m, degC.
        wind_speed_ms: Wind speed at 10 m, m/s.
        mount: Thermal coefficients for the mounting style.

    Returns:
        Cell temperature, degC.
    """
    poa = np.asarray(poa_global, dtype=float)
    back_surface = poa * np.exp(mount.a + mount.b * np.asarray(wind_speed_ms, float))
    return back_surface + np.asarray(air_temp_c, float) + (poa / 1000.0) * mount.delta_t


def pvwatts_dc(
    poa_effective: np.ndarray,
    cell_temp_c: np.ndarray,
    dc_capacity_kw: float,
    gamma_pdc: float,
) -> np.ndarray:
    """DC power from the PVWatts module model, kW.

    Power scales linearly with effective irradiance and derates linearly with
    cell temperature above 25 degC.

    Args:
        poa_effective: POA irradiance after reflection losses, W/m^2.
        cell_temp_c: Cell temperature, degC.
        dc_capacity_kw: Nameplate DC capacity at STC, kW.
        gamma_pdc: Temperature coefficient of P_max, per degC (negative).

    Returns:
        DC power, kW (non-negative).
    """
    scaled = np.asarray(poa_effective, dtype=float) / 1000.0
    derate = 1.0 + gamma_pdc * (np.asarray(cell_temp_c, dtype=float) - 25.0)
    return np.maximum(0.0, scaled * dc_capacity_kw * derate)


def pvwatts_ac(
    dc_power_kw: np.ndarray,
    ac_capacity_kw: float,
    eta_nominal: float = 0.96,
    clip: bool = True,
) -> np.ndarray:
    """AC power from the PVWatts inverter model, kW.

    Applies a part-load efficiency curve -- inverters are inefficient at very
    low load -- and, by default, clips output at the inverter's AC rating.

    Args:
        dc_power_kw: DC power into the inverter, kW.
        ac_capacity_kw: Inverter AC rating, kW.
        eta_nominal: Inverter nominal (weighted) efficiency.
        clip: When False, skip the AC ceiling. Used to price clipping losses
            by difference against the clipped result.

    Returns:
        AC power, kW.
    """
    dc = np.asarray(dc_power_kw, dtype=float)
    dc_rating = ac_capacity_kw / eta_nominal  # inverter's DC input rating

    # The curve has a 1/zeta term, so evaluate it only where the array is
    # actually producing and substitute zero elsewhere.
    producing = dc > 0.0
    zeta = np.where(producing, dc / dc_rating, 1.0)
    efficiency = (eta_nominal / _INVERTER_ETA_REF) * (
        -0.0162 * zeta - 0.0059 / zeta + 0.9858
    )
    ac = np.where(producing, efficiency * dc, 0.0)
    ceiling = ac_capacity_kw if clip else np.inf
    return np.clip(ac, 0.0, ceiling)


@dataclass(frozen=True)
class SystemSpec:
    """Everything about the PV system that the model needs.

    Attributes:
        dc_capacity_kw: Nameplate DC capacity at STC, kW.
        tilt_deg: Array tilt from horizontal, degrees.
        azimuth_deg: Array azimuth clockwise from north, degrees.
        mount: Key into :data:`MOUNTS`.
        module_type: Key into :data:`MODULE_TYPES`.
        dc_ac_ratio: DC nameplate divided by inverter AC rating.
        inverter_efficiency: Inverter nominal efficiency, 0-1.
        albedo: Ground reflectance, 0-1.
        losses: Loss-fraction breakdown; defaults to the PVWatts stack.
        transposition_model: ``"perez"`` or ``"hdkr"``.
    """

    dc_capacity_kw: float = 4.0
    tilt_deg: float = 35.0
    azimuth_deg: float = 180.0
    mount: str = "roof_mount"
    module_type: str = "premium"
    dc_ac_ratio: float = 1.15
    inverter_efficiency: float = 0.96
    albedo: float = 0.20
    losses: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LOSSES))
    transposition_model: str = "perez"

    def __post_init__(self) -> None:
        if self.dc_capacity_kw <= 0:
            raise ValueError("dc_capacity_kw must be positive")
        if not 0.0 <= self.tilt_deg <= 90.0:
            raise ValueError("tilt_deg must be between 0 and 90")
        if self.mount not in MOUNTS:
            raise ValueError(f"unknown mount {self.mount!r}; expected one of {list(MOUNTS)}")
        if self.module_type not in MODULE_TYPES:
            raise ValueError(f"unknown module_type {self.module_type!r}")
        if self.dc_ac_ratio <= 0:
            raise ValueError("dc_ac_ratio must be positive")
        if not 0.0 < self.inverter_efficiency <= 1.0:
            raise ValueError("inverter_efficiency must be in (0, 1]")
        if not 0.0 <= self.albedo <= 1.0:
            raise ValueError("albedo must be between 0 and 1")

    @property
    def gamma_pdc(self) -> float:
        """Temperature coefficient of P_max for the chosen module type, per degC."""
        return MODULE_TYPES[self.module_type]

    @property
    def ac_capacity_kw(self) -> float:
        """Inverter AC rating implied by the DC capacity and DC/AC ratio."""
        return self.dc_capacity_kw / self.dc_ac_ratio

    @property
    def total_loss_fraction(self) -> float:
        """Combined system loss fraction applied to DC power."""
        return combine_losses(self.losses)


@dataclass(frozen=True)
class PowerChain:
    """Hourly output of the PV model, aligned with the input weather series.

    Attributes:
        poa: The plane-of-array irradiance breakdown.
        cell_temp_c: Cell temperature, degC.
        dc_kw: DC power after system losses, kW.
        ac_kw: AC power at the meter, kW.
        clipped_kw: DC-to-AC power lost to inverter clipping, kW.
    """

    poa: PlaneOfArray
    cell_temp_c: np.ndarray
    dc_kw: np.ndarray
    ac_kw: np.ndarray
    clipped_kw: np.ndarray


def run_power_chain(
    poa: PlaneOfArray,
    air_temp_c: np.ndarray,
    wind_speed_ms: np.ndarray,
    system: SystemSpec,
) -> PowerChain:
    """Convert plane-of-array irradiance and weather into AC power.

    Args:
        poa: Transposed irradiance from :func:`solarest.irradiance.transpose`.
        air_temp_c: Ambient air temperature, degC.
        wind_speed_ms: Wind speed at 10 m, m/s.
        system: The system being modelled.

    Returns:
        A :class:`PowerChain` of hourly series.
    """
    mount = MOUNTS[system.mount]
    cell_temp = sapm_cell_temperature(poa.global_, air_temp_c, wind_speed_ms, mount)

    dc_gross = pvwatts_dc(
        poa.effective, cell_temp, system.dc_capacity_kw, system.gamma_pdc
    )
    dc = dc_gross * (1.0 - system.total_loss_fraction)

    ac = pvwatts_ac(dc, system.ac_capacity_kw, system.inverter_efficiency)

    # What clipping alone costs: the inverter's own conversion loss is not
    # clipping, so compare against the same curve with the ceiling removed.
    unclipped = pvwatts_ac(
        dc, system.ac_capacity_kw, system.inverter_efficiency, clip=False
    )
    clipped = np.maximum(0.0, unclipped - ac)

    return PowerChain(
        poa=poa, cell_temp_c=cell_temp, dc_kw=dc, ac_kw=ac, clipped_kw=clipped
    )
