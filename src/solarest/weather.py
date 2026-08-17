"""Weather input: the Open-Meteo historical reanalysis archive.

Annual yield is estimated from *observed* weather rather than a forecast, so
the data source is Open-Meteo's Historical Weather API, which serves the
ECMWF ERA5 / ERA5-Land reanalysis (hourly, back to 1940, ~25 km grid, about
five days behind real time).

Several whole calendar years are pulled so the result carries a real
distribution rather than a single lucky or unlucky year. Each year is fetched
and cached separately, so a repeat visit or a second system configuration at
the same site costs no API calls.

A :class:`SyntheticClearSky` provider implements the same interface for
offline development and tests. Data it returns is clearly marked as synthetic
and must never be presented as a real estimate.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import httpx
import numpy as np

from .solarpos import SOLAR_CONSTANT, solar_position

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Open-Meteo variable names -> our field names.
#
# Beam is taken on the *horizontal* (`direct_radiation`) rather than as direct
# normal irradiance. ERA5's primitive quantities are global and beam-on-
# horizontal; Open-Meteo's DNI is those divided by its own cos(zenith). Taking
# the primitive and applying our own solar geometry avoids a divide-and-undo
# round trip that blows up at sunrise and sunset.
_HOURLY_VARIABLES = {
    "shortwave_radiation": "ghi",
    "direct_radiation": "bhi",
    "diffuse_radiation": "dhi",
    "temperature_2m": "temp_c",
    "wind_speed_10m": "wind_ms",
}

# ERA5 is published with roughly a five-day lag, so the most recent complete
# calendar year is only safe to use once we are clear of that window.
_ARCHIVE_LAG_DAYS = 7

DEFAULT_YEARS = 10
MAX_YEARS = 20
EARLIEST_YEAR = 1950


@dataclass(frozen=True)
class WeatherSeries:
    """Hourly weather for one location over one or more whole years.

    Timestamps are UTC and **hour-ending**: Open-Meteo reports irradiance as
    the mean over the preceding hour, so the value stamped 12:00 covers
    11:00-12:00. Use :meth:`interval_midpoints` when computing sun position.

    Attributes:
        times: ``datetime64[s]`` UTC timestamps, one per hour.
        ghi: Global horizontal irradiance, W/m^2.
        bhi: Beam (direct) irradiance on the horizontal, W/m^2.
        dhi: Diffuse horizontal irradiance, W/m^2.
        temp_c: Air temperature at 2 m, degC.
        wind_ms: Wind speed at 10 m, m/s.
        latitude: Latitude of the grid cell actually used, degrees north.
        longitude: Longitude of the grid cell actually used, degrees east.
        elevation_m: Grid cell elevation, metres.
        timezone: IANA timezone name for the site.
        utc_offset_seconds: Site's standard UTC offset, seconds.
        source: Human-readable provenance string.
        synthetic: True when the data is modelled, not observed.
    """

    times: np.ndarray
    ghi: np.ndarray
    bhi: np.ndarray
    dhi: np.ndarray
    temp_c: np.ndarray
    wind_ms: np.ndarray
    latitude: float
    longitude: float
    elevation_m: float
    timezone: str
    utc_offset_seconds: int
    source: str
    synthetic: bool = False

    def __len__(self) -> int:
        return int(self.times.size)

    @property
    def years(self) -> list[int]:
        """Sorted list of distinct calendar years present in the series."""
        return sorted({int(y) for y in self.times.astype("datetime64[Y]").astype(int) + 1970})

    def interval_midpoints(self) -> np.ndarray:
        """Timestamps at the centre of each hour-ending averaging interval."""
        return self.times - np.timedelta64(30, "m")

    def local_times(self) -> np.ndarray:
        """Timestamps shifted into the site's standard local time."""
        return self.times + np.timedelta64(self.utc_offset_seconds, "s")


class WeatherProvider(Protocol):
    """Something that can supply weather and metadata for a site."""

    async def site_info(
        self, latitude: float, longitude: float
    ) -> "SiteInfo":  # pragma: no cover - protocol definition
        ...

    async def fetch(
        self, latitude: float, longitude: float, years: Sequence[int]
    ) -> WeatherSeries:  # pragma: no cover - protocol definition
        ...


def default_year_range(count: int = DEFAULT_YEARS, today: date | None = None) -> list[int]:
    """The most recent ``count`` complete calendar years available in ERA5.

    Args:
        count: How many years to include.
        today: Override for the current date, for deterministic tests.

    Returns:
        Ascending list of calendar years.
    """
    if not 1 <= count <= MAX_YEARS:
        raise ValueError(f"years must be between 1 and {MAX_YEARS}, got {count}")
    today = today or date.today()
    # Early in January the previous year may not be fully published yet.
    last_complete = today.year - 1
    if (today - date(today.year, 1, 1)).days < _ARCHIVE_LAG_DAYS:
        last_complete -= 1
    first = max(EARLIEST_YEAR, last_complete - count + 1)
    return list(range(first, last_complete + 1))


class ArchiveError(RuntimeError):
    """Raised when the Open-Meteo archive cannot supply usable data."""


@dataclass(frozen=True)
class SiteInfo:
    """Static facts about a grid cell, resolved once per location.

    Attributes:
        latitude: Latitude of the reanalysis cell, degrees north.
        longitude: Longitude of the reanalysis cell, degrees east.
        elevation_m: Cell elevation, metres.
        timezone: IANA timezone name.
        utc_offset_seconds: The site's **standard** time offset from UTC.
    """

    latitude: float
    longitude: float
    elevation_m: float
    timezone: str
    utc_offset_seconds: int


def standard_utc_offset(timezone_name: str) -> int:
    """The site's non-daylight-saving offset from UTC, in seconds.

    Results are reported in local standard time all year, the same convention
    TMY files and PVGIS use: it keeps every day 24 hours long, so hour-of-day
    profiles stay comparable across the seasons instead of jumping by an hour
    twice a year.

    Args:
        timezone_name: IANA timezone name, e.g. ``"Europe/London"``.

    Returns:
        Offset in seconds; 0 if the timezone is unknown to the system.
    """
    try:
        from zoneinfo import ZoneInfo

        zone = ZoneInfo(timezone_name)
    except Exception:  # unknown zone, or no tz database available
        log.warning("unknown timezone %r; falling back to UTC", timezone_name)
        return 0

    # Standard time is the smallest offset the zone takes across a year;
    # daylight saving only ever moves clocks forward.
    from datetime import datetime

    offsets = []
    for month in range(1, 13):
        moment = datetime(2023, month, 15, 12, 0, tzinfo=zone)
        delta = moment.utcoffset()
        if delta is not None:
            offsets.append(int(delta.total_seconds()))
    return min(offsets) if offsets else 0


class OpenMeteoArchive:
    """Fetches ERA5 reanalysis weather from Open-Meteo, with an on-disk cache.

    Args:
        cache_dir: Where to store cached years. ``None`` disables caching.
        timeout: Per-request timeout, seconds.
        max_concurrency: Simultaneous in-flight requests to Open-Meteo.
        client: Optional pre-built ``httpx.AsyncClient`` (mainly for tests).
        cache_precision: Decimal places coordinates are rounded to for cache
            keys. Three places is ~110 m, far finer than the ~25 km
            reanalysis grid, so it never changes which cell is returned.
    """

    def __init__(
        self,
        cache_dir: Path | str | None = ".cache/weather",
        timeout: float = 60.0,
        max_concurrency: int = 4,
        client: httpx.AsyncClient | None = None,
        cache_precision: int = 3,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.timeout = timeout
        self.max_concurrency = max_concurrency
        self._client = client
        self.cache_precision = cache_precision
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, latitude: float, longitude: float, year: int) -> Path | None:
        if self.cache_dir is None:
            return None
        key = (
            f"{round(latitude, self.cache_precision)}_"
            f"{round(longitude, self.cache_precision)}_{year}_"
            f"{','.join(sorted(_HOURLY_VARIABLES))}"
        )
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{year}_{digest}.npz"

    async def site_info(self, latitude: float, longitude: float) -> SiteInfo:
        """Resolve a site's grid cell, elevation and standard-time offset.

        The bulk weather is requested in UTC so that every day is a clean 24
        hours; this one-day probe with ``timezone=auto`` is what tells us the
        site's actual timezone so results can be reported in local time.

        Args:
            latitude: Degrees north.
            longitude: Degrees east.

        Returns:
            A :class:`SiteInfo` for the location.

        Raises:
            ArchiveError: If Open-Meteo cannot be reached or rejects the query.
        """
        _validate_coordinates(latitude, longitude)
        cached = self._read_site_cache(latitude, longitude)
        if cached is not None:
            return cached

        probe_year = default_year_range(1)[0]
        params = {
            "latitude": f"{latitude:.4f}",
            "longitude": f"{longitude:.4f}",
            "start_date": f"{probe_year}-01-01",
            "end_date": f"{probe_year}-01-01",
            "hourly": "temperature_2m",
            "timezone": "auto",
        }
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(
            timeout=self.timeout,
            headers={"User-Agent": "solarest/0.1 (+solar yield estimator)"},
        )
        try:
            response = await client.get(ARCHIVE_URL, params=params)
        except httpx.HTTPError as exc:
            raise ArchiveError(f"could not reach the Open-Meteo archive: {exc}") from exc
        finally:
            if owns_client:
                await client.aclose()

        if response.status_code != 200:
            raise ArchiveError(
                f"Open-Meteo returned {response.status_code}: {_error_detail(response)}"
            )

        payload = response.json()
        timezone_name = str(payload.get("timezone") or "UTC")
        info = SiteInfo(
            latitude=float(payload.get("latitude", latitude)),
            longitude=float(payload.get("longitude", longitude)),
            elevation_m=float(payload.get("elevation", 0.0) or 0.0),
            timezone=timezone_name,
            utc_offset_seconds=standard_utc_offset(timezone_name),
        )
        self._write_site_cache(latitude, longitude, info)
        return info

    def _site_cache_path(self, latitude: float, longitude: float) -> Path | None:
        if self.cache_dir is None:
            return None
        key = f"site_{round(latitude, self.cache_precision)}_{round(longitude, self.cache_precision)}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"site_{digest}.json"

    def _read_site_cache(self, latitude: float, longitude: float) -> SiteInfo | None:
        path = self._site_cache_path(latitude, longitude)
        if path is None or not path.exists():
            return None
        try:
            return SiteInfo(**json.loads(path.read_text()))
        except (OSError, ValueError, TypeError) as exc:
            log.warning("discarding unreadable site cache %s: %s", path, exc)
            return None

    def _write_site_cache(self, latitude: float, longitude: float, info: SiteInfo) -> None:
        path = self._site_cache_path(latitude, longitude)
        if path is None:
            return
        try:
            path.write_text(json.dumps(info.__dict__))
        except OSError as exc:
            log.warning("could not write site cache %s: %s", path, exc)

    async def fetch(
        self, latitude: float, longitude: float, years: Sequence[int]
    ) -> WeatherSeries:
        """Fetch and concatenate whole years of hourly weather for a site.

        Years already on disk are read from the cache; the rest are requested
        concurrently.

        Args:
            latitude: Degrees north.
            longitude: Degrees east.
            years: Calendar years to fetch, in any order.

        Returns:
            One :class:`WeatherSeries` covering all requested years, sorted by
            time.

        Raises:
            ArchiveError: If Open-Meteo rejects the request or returns no
                usable data.
        """
        _validate_coordinates(latitude, longitude)
        wanted = sorted(set(int(y) for y in years))
        if not wanted:
            raise ValueError("at least one year must be requested")

        cached: dict[int, WeatherSeries] = {}
        missing: list[int] = []
        for year in wanted:
            hit = self._read_cache(latitude, longitude, year)
            if hit is None:
                missing.append(year)
            else:
                cached[year] = hit

        if missing:
            fetched = await self._fetch_years(latitude, longitude, missing)
            for year, series in fetched.items():
                self._write_cache(latitude, longitude, year, series)
                cached[year] = series

        return concat_series([cached[y] for y in wanted])

    async def _fetch_years(
        self, latitude: float, longitude: float, years: Iterable[int]
    ) -> dict[int, WeatherSeries]:
        semaphore = asyncio.Semaphore(self.max_concurrency)
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(
            timeout=self.timeout, headers={"User-Agent": "solarest/0.1 (+solar yield estimator)"}
        )
        try:

            async def one(year: int) -> tuple[int, WeatherSeries]:
                async with semaphore:
                    return year, await self._fetch_year(client, latitude, longitude, year)

            results = await asyncio.gather(*(one(y) for y in years))
        finally:
            if owns_client:
                await client.aclose()
        return dict(results)

    async def _fetch_year(
        self,
        client: httpx.AsyncClient,
        latitude: float,
        longitude: float,
        year: int,
    ) -> WeatherSeries:
        params = {
            "latitude": f"{latitude:.4f}",
            "longitude": f"{longitude:.4f}",
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "hourly": ",".join(_HOURLY_VARIABLES),
            "timezone": "GMT",
            "wind_speed_unit": "ms",
            "temperature_unit": "celsius",
        }
        try:
            response = await client.get(ARCHIVE_URL, params=params)
        except httpx.HTTPError as exc:
            raise ArchiveError(f"could not reach the Open-Meteo archive: {exc}") from exc

        if response.status_code != 200:
            detail = _error_detail(response)
            raise ArchiveError(f"Open-Meteo returned {response.status_code} for {year}: {detail}")

        return _parse_archive_response(response.json(), year)

    def _read_cache(
        self, latitude: float, longitude: float, year: int
    ) -> WeatherSeries | None:
        path = self._cache_path(latitude, longitude, year)
        if path is None or not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data["meta"]))
                return WeatherSeries(
                    times=data["times"].astype("datetime64[s]"),
                    ghi=data["ghi"],
                    bhi=data["bhi"],
                    dhi=data["dhi"],
                    temp_c=data["temp_c"],
                    wind_ms=data["wind_ms"],
                    **meta,
                )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # A truncated or stale-schema cache file should never be fatal.
            log.warning("discarding unreadable weather cache %s: %s", path, exc)
            try:
                path.unlink()
            except OSError:
                pass
            return None

    def _write_cache(
        self, latitude: float, longitude: float, year: int, series: WeatherSeries
    ) -> None:
        path = self._cache_path(latitude, longitude, year)
        if path is None:
            return
        meta = {
            "latitude": series.latitude,
            "longitude": series.longitude,
            "elevation_m": series.elevation_m,
            "timezone": series.timezone,
            "utc_offset_seconds": series.utc_offset_seconds,
            "source": series.source,
            "synthetic": series.synthetic,
        }
        tmp = path.with_suffix(".tmp.npz")
        try:
            np.savez_compressed(
                tmp,
                times=series.times.astype("datetime64[s]").astype("int64"),
                ghi=series.ghi,
                bhi=series.bhi,
                dhi=series.dhi,
                temp_c=series.temp_c,
                wind_ms=series.wind_ms,
                meta=json.dumps(meta),
            )
            tmp.replace(path)
        except OSError as exc:
            log.warning("could not write weather cache %s: %s", path, exc)
            tmp.unlink(missing_ok=True)


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:200]
    return str(payload.get("reason", payload))[:200]


def _validate_coordinates(latitude: float, longitude: float) -> None:
    if not -90.0 <= latitude <= 90.0:
        raise ValueError(f"latitude must be between -90 and 90, got {latitude}")
    if not -180.0 <= longitude <= 180.0:
        raise ValueError(f"longitude must be between -180 and 180, got {longitude}")


def _parse_archive_response(payload: dict, year: int) -> WeatherSeries:
    """Turn one Open-Meteo archive JSON payload into a :class:`WeatherSeries`."""
    hourly = payload.get("hourly")
    if not hourly or not hourly.get("time"):
        raise ArchiveError(f"Open-Meteo returned no hourly data for {year}")

    times = np.array(hourly["time"], dtype="datetime64[s]")
    units = payload.get("hourly_units", {})

    fields: dict[str, np.ndarray] = {}
    for api_name, field_name in _HOURLY_VARIABLES.items():
        raw = hourly.get(api_name)
        if raw is None:
            raise ArchiveError(f"Open-Meteo response for {year} is missing {api_name}")
        values = np.array([np.nan if v is None else v for v in raw], dtype=float)
        fields[field_name] = values

    # Defend against a unit default changing under us: the API is asked for
    # m/s and degC, but honour whatever it says it sent.
    if str(units.get("wind_speed_10m", "ms")).replace("/", "") in {"kmh", "kmh-1"}:
        fields["wind_ms"] = fields["wind_ms"] / 3.6
    if str(units.get("temperature_2m", "°C")).endswith("F"):
        fields["temp_c"] = (fields["temp_c"] - 32.0) * 5.0 / 9.0

    # Gaps: irradiance gaps are genuinely dark-or-unknown and are safest at
    # zero; temperature and wind are interpolated so the thermal model stays
    # continuous.
    for name in ("ghi", "bhi", "dhi"):
        fields[name] = np.nan_to_num(fields[name], nan=0.0, posinf=0.0, neginf=0.0)
        fields[name] = np.maximum(0.0, fields[name])
    fields["temp_c"] = _interpolate_gaps(fields["temp_c"], fallback=15.0)
    fields["wind_ms"] = np.maximum(0.0, _interpolate_gaps(fields["wind_ms"], fallback=1.0))

    return WeatherSeries(
        times=times,
        latitude=float(payload.get("latitude", np.nan)),
        longitude=float(payload.get("longitude", np.nan)),
        elevation_m=float(payload.get("elevation", 0.0) or 0.0),
        timezone=str(payload.get("timezone", "GMT")),
        utc_offset_seconds=int(payload.get("utc_offset_seconds", 0) or 0),
        source="Open-Meteo Historical Weather API (ECMWF ERA5 reanalysis)",
        **fields,
    )


def _interpolate_gaps(values: np.ndarray, fallback: float) -> np.ndarray:
    """Fill NaNs by linear interpolation; use ``fallback`` if all are NaN."""
    missing = np.isnan(values)
    if not missing.any():
        return values
    if missing.all():
        return np.full_like(values, fallback)
    index = np.arange(values.size)
    filled = values.copy()
    filled[missing] = np.interp(index[missing], index[~missing], values[~missing])
    return filled


def concat_series(parts: Sequence[WeatherSeries]) -> WeatherSeries:
    """Join per-year series into one, sorted by time and de-duplicated.

    Args:
        parts: Series for the same site, in any order.

    Returns:
        A single :class:`WeatherSeries`. Metadata is taken from the first part.

    Raises:
        ValueError: If ``parts`` is empty.
    """
    if not parts:
        raise ValueError("nothing to concatenate")
    if len(parts) == 1:
        return parts[0]

    times = np.concatenate([p.times for p in parts])
    order = np.argsort(times, kind="stable")
    times = times[order]
    unique = np.concatenate(([True], times[1:] != times[:-1]))

    def stitch(name: str) -> np.ndarray:
        joined = np.concatenate([getattr(p, name) for p in parts])[order]
        return joined[unique]

    head = parts[0]
    return replace(
        head,
        times=times[unique],
        ghi=stitch("ghi"),
        bhi=stitch("bhi"),
        dhi=stitch("dhi"),
        temp_c=stitch("temp_c"),
        wind_ms=stitch("wind_ms"),
    )


class SyntheticClearSky:
    """Offline provider generating physically plausible clear-sky weather.

    Uses the Ineichen-Perez clear-sky model with a fixed Linke turbidity, plus
    a smooth seasonal/diurnal air temperature cycle. Output is deterministic,
    which makes it useful for tests and for developing without network access,
    but it is **not** a real resource estimate: it has no clouds, so it
    overstates yield substantially in most climates.

    Args:
        linke_turbidity: Atmospheric turbidity; 2 is very clear, 5 is hazy.
        elevation_m: Site elevation used by the clear-sky model.
    """

    def __init__(self, linke_turbidity: float = 3.0, elevation_m: float = 0.0) -> None:
        self.linke_turbidity = linke_turbidity
        self.elevation_m = elevation_m

    async def site_info(self, latitude: float, longitude: float) -> SiteInfo:
        """Approximate site metadata, using nautical time zones."""
        _validate_coordinates(latitude, longitude)
        offset_hours = int(round(longitude / 15.0))
        return SiteInfo(
            latitude=latitude,
            longitude=longitude,
            elevation_m=self.elevation_m,
            timezone=f"Etc/GMT{-offset_hours:+d}",
            utc_offset_seconds=offset_hours * 3600,
        )

    async def fetch(
        self, latitude: float, longitude: float, years: Sequence[int]
    ) -> WeatherSeries:
        """Generate hourly clear-sky weather for the requested years."""
        _validate_coordinates(latitude, longitude)
        wanted = sorted(set(int(y) for y in years))
        if not wanted:
            raise ValueError("at least one year must be requested")

        blocks = [self._year(latitude, longitude, y) for y in wanted]
        return concat_series(blocks)

    def _year(self, latitude: float, longitude: float, year: int) -> WeatherSeries:
        start = np.datetime64(f"{year}-01-01T01:00:00", "s")
        end = np.datetime64(f"{year + 1}-01-01T00:00:00", "s")
        times = np.arange(start, end + np.timedelta64(1, "s"), np.timedelta64(1, "h"))

        midpoints = times - np.timedelta64(30, "m")
        pos = solar_position(midpoints, latitude, longitude)
        ghi, bhi, dhi = ineichen_clear_sky(
            pos.zenith, pos.air_mass, pos.extra_normal, self.elevation_m, self.linke_turbidity
        )

        day_of_year = (
            (times.astype("datetime64[D]") - np.datetime64(f"{year}-01-01")).astype(int) + 1
        )
        hour_of_day = (times - times.astype("datetime64[D]")).astype("timedelta64[m]").astype(
            float
        ) / 60.0

        # Seasonal swing scaled by latitude, plus a diurnal cycle lagging the
        # sun by a couple of hours.
        seasonal_phase = 2 * np.pi * (day_of_year - 196) / 365.25
        hemisphere = np.sign(latitude) if latitude != 0 else 1.0
        mean_temp = 26.0 - 0.35 * abs(latitude)
        seasonal = 0.28 * abs(latitude) * np.cos(seasonal_phase) * hemisphere
        solar_hour = np.mod(hour_of_day + longitude / 15.0, 24.0)
        diurnal = 5.0 * np.cos(2 * np.pi * (solar_hour - 15.0) / 24.0)

        return WeatherSeries(
            times=times,
            ghi=ghi,
            bhi=bhi,
            dhi=dhi,
            temp_c=mean_temp + seasonal + diurnal,
            wind_ms=np.full(times.shape, 2.5),
            latitude=latitude,
            longitude=longitude,
            elevation_m=self.elevation_m,
            timezone="GMT",
            utc_offset_seconds=int(round(longitude / 15.0) * 3600),
            source=f"Synthetic Ineichen-Perez clear sky (TL={self.linke_turbidity})",
            synthetic=True,
        )


def ineichen_clear_sky(
    apparent_zenith_deg: np.ndarray,
    air_mass: np.ndarray,
    extra_normal: np.ndarray,
    elevation_m: float,
    linke_turbidity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ineichen & Perez (2002) clear-sky irradiance.

    Args:
        apparent_zenith_deg: Apparent solar zenith, degrees.
        air_mass: Relative air mass (``inf`` below the horizon).
        extra_normal: Extraterrestrial normal irradiance, W/m^2.
        elevation_m: Site elevation, metres.
        linke_turbidity: Linke turbidity factor.

    Returns:
        ``(ghi, bhi, dhi)`` in W/m^2 -- global, beam-on-horizontal and diffuse
        horizontal -- satisfying ``ghi = bhi + dhi`` exactly.
    """
    cos_zenith = np.maximum(0.0, np.cos(np.radians(np.asarray(apparent_zenith_deg, float))))
    day = np.isfinite(air_mass) & (cos_zenith > 0.0)
    am = np.where(day, air_mass, 1.0)

    fh1 = np.exp(-elevation_m / 8000.0)
    fh2 = np.exp(-elevation_m / 1250.0)
    a1 = 5.09e-5 * elevation_m + 0.868
    a2 = 3.92e-5 * elevation_m + 0.0387

    ghi = (
        a1
        * extra_normal
        * cos_zenith
        * np.exp(-a2 * am * (fh1 + fh2 * (linke_turbidity - 1.0)))
        * np.exp(0.01 * am**1.8)
    )
    ghi = np.where(day, np.maximum(0.0, ghi), 0.0)

    b = 0.664 + 0.163 / fh1
    dni = b * extra_normal * np.exp(-0.09 * am * (linke_turbidity - 1.0))
    # Beam on the horizontal cannot exceed the global the model allows through.
    bhi = np.where(day, np.clip(dni * cos_zenith, 0.0, ghi), 0.0)

    dhi = np.maximum(0.0, ghi - bhi)
    return ghi, bhi, dhi


def clear_sky_reference(
    times: np.ndarray, latitude: float, longitude: float, elevation_m: float = 0.0
) -> np.ndarray:
    """Clear-sky GHI for a timestamp series, used as a sanity ceiling.

    Returns:
        Clear-sky global horizontal irradiance in W/m^2.
    """
    pos = solar_position(times, latitude, longitude)
    ghi, _, _ = ineichen_clear_sky(
        pos.zenith, pos.air_mass, pos.extra_normal, elevation_m, 3.0
    )
    return np.minimum(ghi, SOLAR_CONSTANT)


def split_beam(
    ghi: np.ndarray,
    bhi: np.ndarray,
    dhi: np.ndarray,
    cos_zenith: np.ndarray,
    extra_normal: np.ndarray,
    min_cos_zenith: float = 0.06976,  # cos(86 deg)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconcile the three irradiance components and derive DNI.

    Reanalysis irradiance arrives as hour-averages, so at sunrise and sunset
    the beam component and the instantaneous sun position disagree: dividing
    beam-on-horizontal by a near-zero cosine produces absurd direct normal
    values. This enforces ``ghi = bhi + dhi``, derives DNI only while the sun
    is comfortably up, and reassigns the rest to diffuse -- which is where
    that energy physically goes at very low sun angles anyway.

    Args:
        ghi: Global horizontal irradiance, W/m^2.
        bhi: Beam irradiance on the horizontal, W/m^2.
        dhi: Diffuse horizontal irradiance, W/m^2.
        cos_zenith: Cosine of the apparent solar zenith, floored at 0.
        extra_normal: Extraterrestrial normal irradiance, W/m^2 -- the
            physical ceiling for DNI.
        min_cos_zenith: Below this cosine, all irradiance is treated as
            diffuse.

    Returns:
        ``(ghi, dni, dhi)`` in W/m^2, mutually consistent.
    """
    ghi = np.maximum(0.0, np.asarray(ghi, dtype=float))
    bhi = np.clip(np.asarray(bhi, dtype=float), 0.0, ghi)
    dhi = np.asarray(dhi, dtype=float)

    # Trust global and beam; diffuse is the remainder. ERA5 defines it that
    # way, and it keeps the energy balance exact even when a value is patched.
    dhi = np.where(np.isfinite(dhi), np.clip(dhi, 0.0, ghi), ghi - bhi)
    total = bhi + dhi
    scale = np.where(total > 0.0, ghi / np.where(total > 0.0, total, 1.0), 0.0)
    bhi = bhi * scale
    dhi = ghi - bhi

    usable = np.asarray(cos_zenith, dtype=float) >= min_cos_zenith
    safe_cos = np.where(usable, cos_zenith, 1.0)
    dni = np.where(usable, np.clip(bhi / safe_cos, 0.0, extra_normal), 0.0)

    # Whatever beam we could not attribute to a credible sun direction stays
    # in the plane's energy budget as diffuse.
    dhi = ghi - dni * np.asarray(cos_zenith, dtype=float)
    return ghi, dni, np.maximum(0.0, dhi)
