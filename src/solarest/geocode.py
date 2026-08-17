"""Place-name search via the Open-Meteo geocoding API."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


@dataclass(frozen=True)
class Place:
    """A geocoded location.

    Attributes:
        name: Settlement name.
        latitude: Degrees north.
        longitude: Degrees east.
        country: Country name, if known.
        country_code: ISO 3166-1 alpha-2 code, if known.
        admin1: First-level administrative area (state, region), if known.
        elevation_m: Elevation in metres, if known.
        timezone: IANA timezone name, if known.
        population: Population, if known.
    """

    name: str
    latitude: float
    longitude: float
    country: str | None = None
    country_code: str | None = None
    admin1: str | None = None
    elevation_m: float | None = None
    timezone: str | None = None
    population: int | None = None

    @property
    def label(self) -> str:
        """A single-line human-readable description of the place."""
        parts = [self.name]
        if self.admin1 and self.admin1 != self.name:
            parts.append(self.admin1)
        if self.country:
            parts.append(self.country)
        return ", ".join(parts)

    def to_dict(self) -> dict:
        """Serialise to a JSON-friendly dict, including the display label."""
        return {**asdict(self), "label": self.label}


class GeocodingError(RuntimeError):
    """Raised when the geocoding service cannot be reached or rejects a query."""


async def search(
    name: str,
    count: int = 8,
    language: str = "en",
    timeout: float = 15.0,
    client: httpx.AsyncClient | None = None,
) -> list[Place]:
    """Look up places matching a search string.

    Args:
        name: Free-text place name, at least two characters.
        count: Maximum number of results.
        language: Preferred result language.
        timeout: Request timeout in seconds.
        client: Optional pre-built client (mainly for tests).

    Returns:
        Matching places, best match first. Empty when nothing matches.

    Raises:
        ValueError: If ``name`` is too short.
        GeocodingError: If the service is unreachable or returns an error.
    """
    query = name.strip()
    if len(query) < 2:
        raise ValueError("search text must be at least two characters")

    params = {
        "name": query,
        "count": max(1, min(int(count), 20)),
        "language": language,
        "format": "json",
    }

    owns_client = client is None
    client = client or httpx.AsyncClient(
        timeout=timeout, headers={"User-Agent": "solarest/0.1 (+solar yield estimator)"}
    )
    try:
        response = await client.get(GEOCODING_URL, params=params)
    except httpx.HTTPError as exc:
        raise GeocodingError(f"could not reach the geocoding service: {exc}") from exc
    finally:
        if owns_client:
            await client.aclose()

    if response.status_code != 200:
        raise GeocodingError(f"geocoding service returned {response.status_code}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise GeocodingError("geocoding service returned malformed JSON") from exc

    # Open-Meteo omits "results" entirely when nothing matches.
    return [_to_place(item) for item in payload.get("results") or []]


def _to_place(item: dict) -> Place:
    elevation = item.get("elevation")
    population = item.get("population")
    return Place(
        name=str(item.get("name", "")),
        latitude=float(item["latitude"]),
        longitude=float(item["longitude"]),
        country=item.get("country"),
        country_code=item.get("country_code"),
        admin1=item.get("admin1"),
        elevation_m=float(elevation) if elevation is not None else None,
        timezone=item.get("timezone"),
        population=int(population) if population is not None else None,
    )
