"""Estimate the solar PV energy a location can generate over a year.

The public surface, roughly in the order a caller uses it:

* :mod:`solarest.weather` -- fetch hourly reanalysis weather from Open-Meteo.
* :mod:`solarest.simulate` -- run the model chain over that weather.
* :mod:`solarest.results` -- aggregate hours into annual, monthly, diurnal and
  typical-year views.
* :mod:`solarest.optimize` -- search for the best array orientation.
* :mod:`solarest.load` -- household demand, from meter data or from bills.
* :mod:`solarest.battery` -- dispatch a battery against generation and demand.
* :mod:`solarest.sizing` -- put those together and say what the answer is worth.
* :mod:`solarest.api` -- the HTTP service and the website it serves.
"""

from .battery import BatterySpec, DispatchResult, dispatch, suggest_capacity, sweep_sizes
from .irradiance import PlaneOfArray, transpose
from .load import HouseholdShape, LoadProfile, synthesise
from .optimize import equator_facing_azimuth, optimise_orientation, rule_of_thumb_tilt
from .pvmodel import MODULE_TYPES, MOUNTS, PowerChain, SystemSpec
from .results import EstimateResult, TypicalYear, summarise
from .simulate import Simulation, SitePrecompute, precompute, simulate
from .sizing import SizingResult, size_for_household
from .solarpos import SolarPosition, solar_position
from .weather import (
    OpenMeteoArchive,
    SiteInfo,
    SyntheticClearSky,
    WeatherSeries,
    default_year_range,
)

__version__ = "0.1.0"

__all__ = [
    "BatterySpec",
    "DispatchResult",
    "EstimateResult",
    "HouseholdShape",
    "LoadProfile",
    "MODULE_TYPES",
    "MOUNTS",
    "OpenMeteoArchive",
    "PlaneOfArray",
    "PowerChain",
    "SiteInfo",
    "Simulation",
    "SitePrecompute",
    "SizingResult",
    "SolarPosition",
    "SyntheticClearSky",
    "SystemSpec",
    "TypicalYear",
    "WeatherSeries",
    "default_year_range",
    "dispatch",
    "equator_facing_azimuth",
    "optimise_orientation",
    "precompute",
    "rule_of_thumb_tilt",
    "simulate",
    "size_for_household",
    "solar_position",
    "suggest_capacity",
    "summarise",
    "sweep_sizes",
    "synthesise",
    "transpose",
    "__version__",
]
