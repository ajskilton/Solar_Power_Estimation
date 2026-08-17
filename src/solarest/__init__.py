"""Estimate the solar PV energy a location can generate over a year.

The public surface, roughly in the order a caller uses it:

* :mod:`solarest.weather` -- fetch hourly reanalysis weather from Open-Meteo.
* :mod:`solarest.simulate` -- run the model chain over that weather.
* :mod:`solarest.results` -- aggregate hours into annual, monthly, diurnal and
  typical-year views.
* :mod:`solarest.optimize` -- search for the best array orientation.
* :mod:`solarest.api` -- the HTTP service and the website it serves.
"""

from .irradiance import PlaneOfArray, transpose
from .optimize import equator_facing_azimuth, optimise_orientation, rule_of_thumb_tilt
from .pvmodel import MODULE_TYPES, MOUNTS, PowerChain, SystemSpec
from .results import EstimateResult, TypicalYear, summarise
from .simulate import Simulation, SitePrecompute, precompute, simulate
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
    "EstimateResult",
    "MODULE_TYPES",
    "MOUNTS",
    "OpenMeteoArchive",
    "PlaneOfArray",
    "PowerChain",
    "SiteInfo",
    "Simulation",
    "SitePrecompute",
    "SolarPosition",
    "SyntheticClearSky",
    "SystemSpec",
    "TypicalYear",
    "WeatherSeries",
    "default_year_range",
    "equator_facing_azimuth",
    "optimise_orientation",
    "precompute",
    "rule_of_thumb_tilt",
    "simulate",
    "solar_position",
    "summarise",
    "transpose",
    "__version__",
]
