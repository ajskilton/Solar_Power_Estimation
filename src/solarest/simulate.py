"""End-to-end simulation: a site's weather plus a system spec, hour by hour."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .irradiance import transpose
from .pvmodel import PowerChain, SystemSpec, run_power_chain
from .solarpos import SolarPosition, solar_position
from .weather import WeatherSeries, split_beam


@dataclass(frozen=True)
class SitePrecompute:
    """Everything about a site that does not depend on array orientation.

    Sun position and irradiance reconciliation are the expensive part of a
    run and are identical for every system at the same site, so they are
    computed once and reused -- which is what makes the orientation search
    affordable.

    Attributes:
        weather: The weather series this was built from.
        position: Sun position at each averaging interval's midpoint.
        ghi: Global horizontal irradiance, W/m^2.
        dni: Direct normal irradiance after reconciliation, W/m^2.
        dhi: Diffuse horizontal irradiance after reconciliation, W/m^2.
        local_times: Timestamps in the site's local standard time.
    """

    weather: WeatherSeries
    position: SolarPosition
    ghi: np.ndarray
    dni: np.ndarray
    dhi: np.ndarray
    local_times: np.ndarray

    def run(self, system: SystemSpec) -> PowerChain:
        """Model one system orientation and rating against this site."""
        poa = transpose(
            surface_tilt_deg=system.tilt_deg,
            surface_azimuth_deg=system.azimuth_deg,
            ghi=self.ghi,
            dni=self.dni,
            dhi=self.dhi,
            solar_zenith_deg=self.position.zenith,
            solar_azimuth_deg=self.position.azimuth,
            extra_normal=self.position.extra_normal,
            air_mass=self.position.air_mass,
            albedo=system.albedo,
            model=system.transposition_model,
        )
        return run_power_chain(
            poa, self.weather.temp_c, self.weather.wind_ms, system
        )

    def simulate(self, system: SystemSpec) -> "Simulation":
        """Model a system and wrap the result with its site context."""
        return Simulation(
            weather=self.weather,
            system=system,
            position=self.position,
            chain=self.run(system),
            dni=self.dni,
            dhi=self.dhi,
            local_times=self.local_times,
        )


@dataclass(frozen=True)
class Simulation:
    """Hourly simulation output aligned with the input weather series.

    Attributes:
        weather: The weather the simulation was driven with.
        system: The system that was modelled.
        position: Sun position at each averaging interval's midpoint.
        chain: Irradiance, temperature and power series.
        dni: Direct normal irradiance derived during reconciliation, W/m^2.
        dhi: Diffuse horizontal irradiance after reconciliation, W/m^2.
        local_times: Timestamps in the site's local *standard* time.
    """

    weather: WeatherSeries
    system: SystemSpec
    position: SolarPosition
    chain: PowerChain
    dni: np.ndarray
    dhi: np.ndarray
    local_times: np.ndarray

    @property
    def ac_kw(self) -> np.ndarray:
        """AC power at the meter, kW, one value per hour."""
        return self.chain.ac_kw

    @property
    def energy_kwh(self) -> np.ndarray:
        """Energy per hourly step, kWh. Numerically equal to mean power in kW."""
        return self.chain.ac_kw

    def with_system(self, system: SystemSpec) -> "Simulation":
        """Re-run at the same site with a different system spec."""
        return replace(self, system=system, chain=self._precompute().run(system))

    def _precompute(self) -> SitePrecompute:
        return SitePrecompute(
            weather=self.weather,
            position=self.position,
            ghi=self.weather.ghi,
            dni=self.dni,
            dhi=self.dhi,
            local_times=self.local_times,
        )


def precompute(
    weather: WeatherSeries, utc_offset_seconds: int | None = None
) -> SitePrecompute:
    """Compute the orientation-independent part of a site simulation.

    Sun position is evaluated at the **midpoint** of each hour-ending
    averaging interval, which is the consistent pairing for hour-averaged
    irradiance and matters most at sunrise and sunset.

    Args:
        weather: Hourly weather for the site.
        utc_offset_seconds: Site standard-time offset used for the reported
            local timestamps. Defaults to the weather series' own offset.

    Returns:
        A reusable :class:`SitePrecompute`.
    """
    midpoints = weather.interval_midpoints()
    position = solar_position(midpoints, weather.latitude, weather.longitude)

    ghi, dni, dhi = split_beam(
        weather.ghi,
        weather.bhi,
        weather.dhi,
        position.cos_zenith,
        position.extra_normal,
    )

    offset = (
        weather.utc_offset_seconds if utc_offset_seconds is None else utc_offset_seconds
    )
    return SitePrecompute(
        weather=weather,
        position=position,
        ghi=ghi,
        dni=dni,
        dhi=dhi,
        local_times=weather.times + np.timedelta64(int(offset), "s"),
    )


def simulate(
    weather: WeatherSeries,
    system: SystemSpec,
    utc_offset_seconds: int | None = None,
) -> Simulation:
    """Run the full model chain over a weather series.

    Args:
        weather: Hourly weather for the site.
        system: The PV system to model.
        utc_offset_seconds: Site standard-time offset for local timestamps.

    Returns:
        A :class:`Simulation` of hourly series.
    """
    return precompute(weather, utc_offset_seconds).simulate(system)
