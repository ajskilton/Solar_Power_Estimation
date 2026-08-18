"""Running the estimator inside a browser, with no server behind it.

The model chain needs nothing but numpy, so it runs unchanged under Pyodide.
The only thing that cannot cross into the browser is the HTTP client: there
are no sockets in a WebAssembly sandbox, and requests have to go through the
host page's ``fetch``.

So this module supplies a :class:`~solarest.weather.WeatherProvider` that gets
its JSON from an injected coroutine instead of from httpx. In the browser that
coroutine wraps ``pyodide.http.pyfetch``; in tests it is a plain function
returning a dict, which is how the parsing and error handling below are
covered without a browser anywhere in sight.

Everything downstream -- the transposition, the PV model, the TMY assembly,
the battery dispatch -- is the same code the test suite and the CLI exercise.
That is the point of doing it this way rather than porting the model to
JavaScript: there is one implementation, and it is the validated one.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from urllib.parse import urlencode

from .geocode import GEOCODING_URL, GeocodingError, Place
from .weather import (
    ARCHIVE_URL,
    ArchiveError,
    SiteInfo,
    WeatherSeries,
    _HOURLY_VARIABLES,
    _parse_archive_response,
    _validate_coordinates,
    concat_series,
    default_year_range,
    standard_utc_offset,
)

# A coroutine taking a fully-formed URL and returning decoded JSON.
FetchJson = Callable[[str], Awaitable[dict]]


async def pyodide_fetch_json(url: str) -> dict:
    """Fetch JSON through the browser, via Pyodide.

    Imported lazily so this module stays importable -- and testable -- outside
    a browser.

    Args:
        url: Absolute URL to request.

    Returns:
        The decoded JSON body.

    Raises:
        ArchiveError: If the request fails or the response is not JSON.
    """
    try:
        from pyodide.http import pyfetch
    except ModuleNotFoundError as exc:  # pragma: no cover - browser only
        raise ArchiveError(
            "browser fetching needs Pyodide; pass fetch_json= when running elsewhere"
        ) from exc

    try:
        response = await pyfetch(url)
    except Exception as exc:  # pragma: no cover - network failure in browser
        raise ArchiveError(f"could not reach {_host_of(url)}: {exc}") from exc

    if response.status != 200:  # pragma: no cover - browser only
        raise ArchiveError(f"{_host_of(url)} returned {response.status}")
    return await response.json()


def _host_of(url: str) -> str:
    """The hostname part of a URL, for error messages."""
    return url.split("//", 1)[-1].split("/", 1)[0] or url


def _url(base: str, params: dict) -> str:
    return f"{base}?{urlencode(params)}"


class BrowserArchive:
    """Open-Meteo archive client that fetches through the host page.

    Mirrors :class:`~solarest.weather.OpenMeteoArchive` minus the on-disk
    cache -- a browser tab has no filesystem worth the name, and the site
    keeps results in memory for the session instead.

    Args:
        fetch_json: Coroutine returning decoded JSON for a URL. Defaults to
            the Pyodide implementation; tests inject their own.
    """

    def __init__(self, fetch_json: FetchJson | None = None) -> None:
        self._fetch_json = fetch_json or pyodide_fetch_json
        self._site_cache: dict[tuple[float, float], SiteInfo] = {}
        self._year_cache: dict[tuple[float, float, int], WeatherSeries] = {}

    async def site_info(self, latitude: float, longitude: float) -> SiteInfo:
        """Resolve the grid cell, elevation and standard-time offset.

        Args:
            latitude: Degrees north.
            longitude: Degrees east.

        Returns:
            A :class:`~solarest.weather.SiteInfo`.

        Raises:
            ArchiveError: If Open-Meteo cannot be reached or rejects the query.
        """
        _validate_coordinates(latitude, longitude)
        key = (round(latitude, 3), round(longitude, 3))
        if key in self._site_cache:
            return self._site_cache[key]

        probe_year = default_year_range(1)[0]
        payload = await self._fetch_json(
            _url(
                ARCHIVE_URL,
                {
                    "latitude": f"{latitude:.4f}",
                    "longitude": f"{longitude:.4f}",
                    "start_date": f"{probe_year}-01-01",
                    "end_date": f"{probe_year}-01-01",
                    "hourly": "temperature_2m",
                    "timezone": "auto",
                },
            )
        )
        timezone_name = str(payload.get("timezone") or "UTC")
        info = SiteInfo(
            latitude=float(payload.get("latitude", latitude)),
            longitude=float(payload.get("longitude", longitude)),
            elevation_m=float(payload.get("elevation", 0.0) or 0.0),
            timezone=timezone_name,
            utc_offset_seconds=standard_utc_offset(timezone_name),
        )
        self._site_cache[key] = info
        return info

    async def fetch(
        self, latitude: float, longitude: float, years: Sequence[int]
    ) -> WeatherSeries:
        """Fetch whole years of hourly weather and join them.

        Years are requested one at a time rather than concurrently. A decade
        of ERA5 is ten sizeable responses, and a browser tab hitting a free
        public API is not the place to open ten sockets at once; the loop also
        keeps the progress reporting honest.

        Args:
            latitude: Degrees north.
            longitude: Degrees east.
            years: Calendar years, in any order.

        Returns:
            One :class:`~solarest.weather.WeatherSeries` covering them all.

        Raises:
            ArchiveError: If Open-Meteo rejects a request or returns no data.
            ValueError: If no years were requested.
        """
        _validate_coordinates(latitude, longitude)
        wanted = sorted({int(y) for y in years})
        if not wanted:
            raise ValueError("at least one year must be requested")

        parts = []
        for year in wanted:
            key = (round(latitude, 3), round(longitude, 3), year)
            if key not in self._year_cache:
                payload = await self._fetch_json(
                    _url(
                        ARCHIVE_URL,
                        {
                            "latitude": f"{latitude:.4f}",
                            "longitude": f"{longitude:.4f}",
                            "start_date": f"{year}-01-01",
                            "end_date": f"{year}-12-31",
                            "hourly": ",".join(_HOURLY_VARIABLES),
                            "timezone": "GMT",
                            "wind_speed_unit": "ms",
                            "temperature_unit": "celsius",
                        },
                    )
                )
                self._year_cache[key] = _parse_archive_response(payload, year)
            parts.append(self._year_cache[key])

        return concat_series(parts)


async def browser_geocode(
    name: str,
    count: int = 8,
    language: str = "en",
    fetch_json: FetchJson | None = None,
) -> list[Place]:
    """Search for a place by name, through the host page's fetch.

    Args:
        name: Query string; at least two characters.
        count: Maximum results.
        language: Preferred language for names.
        fetch_json: Coroutine returning decoded JSON for a URL.

    Returns:
        Matching places, best first.

    Raises:
        ValueError: If the query is too short.
        GeocodingError: If the service cannot be reached.
    """
    query = name.strip()
    if len(query) < 2:
        raise ValueError("search text must be at least two characters")

    fetch = fetch_json or pyodide_fetch_json
    try:
        payload = await fetch(
            _url(
                GEOCODING_URL,
                {"name": query, "count": count, "language": language, "format": "json"},
            )
        )
    except ArchiveError as exc:
        raise GeocodingError(str(exc)) from exc

    return [
        Place(
            name=str(entry.get("name", "")),
            latitude=float(entry["latitude"]),
            longitude=float(entry["longitude"]),
            country=entry.get("country"),
            country_code=entry.get("country_code"),
            admin1=entry.get("admin1"),
            elevation_m=entry.get("elevation"),
            timezone=entry.get("timezone"),
            population=entry.get("population"),
        )
        for entry in payload.get("results") or []
        if entry.get("latitude") is not None and entry.get("longitude") is not None
    ]
