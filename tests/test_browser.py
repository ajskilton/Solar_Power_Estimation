"""The browser build: the injected-fetch archive, and the JS-facing bridge.

No browser here. The point of injecting the fetch is that everything except
the ``pyfetch`` call itself can be exercised in ordinary CPython, which is
where the rest of the suite already runs.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from solarest import webapp
from solarest.browser import BrowserArchive, browser_geocode
from solarest.geocode import GeocodingError
from solarest.weather import ArchiveError


def archive_payload(year: int, hours: int = 48) -> dict:
    base = np.datetime64(f"{year}-01-01T00:00", "h")
    return {
        "latitude": 51.5, "longitude": -0.125, "elevation": 25.0,
        "timezone": "GMT", "utc_offset_seconds": 0,
        "hourly_units": {"wind_speed_10m": "m/s", "temperature_2m": "°C"},
        "hourly": {
            "time": [str(base + np.timedelta64(i, "h")) for i in range(hours)],
            "shortwave_radiation": [120.0] * hours,
            "direct_radiation": [80.0] * hours,
            "diffuse_radiation": [40.0] * hours,
            "temperature_2m": [11.0] * hours,
            "wind_speed_10m": [3.5] * hours,
        },
    }


def recording_fetch(handler):
    """Wrap a handler so the test can see every URL requested."""
    calls: list[str] = []

    async def fetch(url: str) -> dict:
        calls.append(url)
        return handler(url)

    return fetch, calls


# --------------------------------------------------------------- the archive


def test_the_archive_asks_open_meteo_for_what_it_needs():
    fetch, calls = recording_fetch(lambda url: archive_payload(2023))
    archive = BrowserArchive(fetch_json=fetch)

    series = asyncio.run(archive.fetch(51.5, -0.125, [2023]))

    assert len(series) == 48
    url = calls[0]
    assert url.startswith("https://archive-api.open-meteo.com/v1/archive?")
    for expected in ("start_date=2023-01-01", "end_date=2023-12-31",
                     "timezone=GMT", "wind_speed_unit=ms", "temperature_unit=celsius"):
        assert expected in url
    for variable in ("shortwave_radiation", "direct_radiation", "diffuse_radiation",
                     "temperature_2m", "wind_speed_10m"):
        assert variable in url


def test_years_are_joined_and_requested_once_each():
    fetch, calls = recording_fetch(lambda url: archive_payload(int(url.split("start_date=")[1][:4])))
    archive = BrowserArchive(fetch_json=fetch)

    series = asyncio.run(archive.fetch(51.5, -0.125, [2022, 2023]))
    assert len(series) == 96
    assert len(calls) == 2


def test_a_second_run_at_the_same_site_costs_no_requests():
    """The in-memory cache is what makes changing the tilt feel instant."""
    fetch, calls = recording_fetch(lambda url: archive_payload(2023))
    archive = BrowserArchive(fetch_json=fetch)

    first = asyncio.run(archive.fetch(51.5, -0.125, [2023]))
    again = asyncio.run(archive.fetch(51.5, -0.125, [2023]))

    assert len(calls) == 1
    assert again.ghi == pytest.approx(first.ghi)


def test_the_site_probe_resolves_the_timezone_in_standard_time():
    def handler(url: str) -> dict:
        assert "timezone=auto" in url
        payload = archive_payload(2024, hours=24)
        payload["timezone"] = "Europe/London"
        return payload

    fetch, _ = recording_fetch(handler)
    info = asyncio.run(BrowserArchive(fetch_json=fetch).site_info(51.5, -0.125))

    assert info.timezone == "Europe/London"
    assert info.utc_offset_seconds == 0  # GMT, not BST
    assert info.elevation_m == 25.0


def test_the_site_probe_is_only_made_once():
    fetch, calls = recording_fetch(lambda url: archive_payload(2024, hours=24))
    archive = BrowserArchive(fetch_json=fetch)

    asyncio.run(archive.site_info(51.5, -0.125))
    asyncio.run(archive.site_info(51.5, -0.125))
    assert len(calls) == 1


def test_a_failing_fetch_surfaces_as_an_archive_error():
    async def broken(url: str) -> dict:
        raise ArchiveError("could not reach archive-api.open-meteo.com: NetworkError")

    with pytest.raises(ArchiveError, match="could not reach"):
        asyncio.run(BrowserArchive(fetch_json=broken).fetch(51.5, -0.125, [2023]))


def test_an_empty_response_is_rejected():
    async def empty(url: str) -> dict:
        return {"hourly": {"time": []}}

    with pytest.raises(ArchiveError, match="no hourly data"):
        asyncio.run(BrowserArchive(fetch_json=empty).fetch(51.5, -0.125, [2023]))


@pytest.mark.parametrize(
    "latitude, longitude", [(95.0, 0.0), (0.0, 200.0)]
)
def test_impossible_coordinates_are_rejected(latitude, longitude):
    fetch, _ = recording_fetch(lambda url: archive_payload(2023))
    with pytest.raises(ValueError):
        asyncio.run(BrowserArchive(fetch_json=fetch).fetch(latitude, longitude, [2023]))


def test_no_years_is_rejected():
    fetch, _ = recording_fetch(lambda url: archive_payload(2023))
    with pytest.raises(ValueError, match="at least one year"):
        asyncio.run(BrowserArchive(fetch_json=fetch).fetch(51.5, -0.125, []))


# -------------------------------------------------------------- geocoding


def test_geocoding_maps_results_onto_places():
    async def fetch(url: str) -> dict:
        assert url.startswith("https://geocoding-api.open-meteo.com/v1/search?")
        assert "name=Bristol" in url
        return {"results": [{
            "name": "Bristol", "latitude": 51.45, "longitude": -2.58,
            "country": "United Kingdom", "admin1": "England", "elevation": 30.0,
        }]}

    places = asyncio.run(browser_geocode("Bristol", fetch_json=fetch))
    assert places[0].label == "Bristol, England, United Kingdom"
    assert places[0].latitude == 51.45


def test_geocoding_skips_entries_without_coordinates():
    async def fetch(url: str) -> dict:
        return {"results": [{"name": "Nowhere"}, {"name": "Bath", "latitude": 51.4, "longitude": -2.4}]}

    places = asyncio.run(browser_geocode("Bath", fetch_json=fetch))
    assert [p.name for p in places] == ["Bath"]


def test_geocoding_handles_no_matches():
    async def fetch(url: str) -> dict:
        return {}

    assert asyncio.run(browser_geocode("Xyzzy", fetch_json=fetch)) == []


def test_a_short_query_is_rejected_before_any_request():
    async def fetch(url: str) -> dict:  # pragma: no cover - must not be called
        raise AssertionError("should not have been requested")

    with pytest.raises(ValueError, match="at least two characters"):
        asyncio.run(browser_geocode("a", fetch_json=fetch))


def test_geocoding_failures_are_reported_as_geocoding_errors():
    async def broken(url: str) -> dict:
        raise ArchiveError("could not reach geocoding-api.open-meteo.com")

    with pytest.raises(GeocodingError, match="could not reach"):
        asyncio.run(browser_geocode("Bristol", fetch_json=broken))


# ----------------------------------------------------------- the JS bridge


def offline(request: dict) -> dict:
    """A request the bridge can serve without a network."""
    return {"latitude": 51.5, "longitude": -0.13, "years": 1, "synthetic": True, **request}


def test_the_bridge_options_match_what_the_page_expects():
    payload = webapp.options()
    assert {m["value"] for m in payload["mounts"]} == {
        "ground_mount", "roof_mount", "roof_integrated"
    }
    assert [a["value"] for a in payload["household_archetypes"]][0] == "nine_to_five"
    assert payload["reference_year"] == 2023


def test_the_bridge_estimates():
    payload = asyncio.run(webapp.estimate(offline({"system": {"dc_capacity_kw": 4.0}})))

    assert payload["annual"]["energy_kwh_mean"] > 0
    assert len(payload["monthly"]) == 12
    assert payload["weather"]["synthetic"] is True
    assert "ac_kw" not in payload["typical_year"]


def test_the_bridge_fills_in_the_same_defaults_as_the_http_api():
    """One definition of "a default system", or the two builds would drift."""
    payload = asyncio.run(webapp.estimate(offline({"system": {"dc_capacity_kw": 3.0}})))
    assert payload["system"]["azimuth_deg"] == 180.0
    assert 30.0 < payload["system"]["tilt_deg"] < 50.0


def test_the_bridge_can_include_the_hourly_series_and_the_orientation_search():
    payload = asyncio.run(
        webapp.estimate(offline({"include_hourly": True, "optimise": True}))
    )
    assert len(payload["typical_year"]["ac_kw"]) == 8760
    assert payload["orientation"]["best_tilt_deg"] >= 0


def test_the_bridge_sizes_a_battery():
    payload = asyncio.run(webapp.sizing(offline({
        "consumption": {"annual_kwh": 3500.0, "household": {"archetype": "nine_to_five"}},
        "battery": {"usable_capacity_kwh": 5.0},
    })))

    assert payload["balance"]["avoided_import_kwh"] > 0
    assert len(payload["sizing_curve"]) > 5
    assert len(payload["diurnal_balance"]["by_year"]["generation"]) == 24
    assert payload["confidence"]["band"] is not None


def test_the_bridge_accepts_metered_data():
    payload = asyncio.run(webapp.sizing(offline({
        "consumption": {"hourly_kwh": [0.4] * 8760},
        "battery": {"usable_capacity_kwh": 5.0},
    })))
    assert payload["consumption"]["measured"] is True
    assert payload["confidence"]["band"] is None


def test_the_bridge_rejects_a_nonsense_system():
    with pytest.raises(ValueError, match="tilt_deg"):
        asyncio.run(webapp.estimate(offline({"system": {"tilt_deg": 120.0}})))


def test_the_bridge_rejects_ambiguous_consumption():
    with pytest.raises(ValueError, match="exactly one"):
        asyncio.run(webapp.sizing(offline({
            "consumption": {"annual_kwh": 3500.0, "monthly_kwh": [300.0] * 12},
        })))


def test_the_bridge_payload_matches_the_http_payload_key_for_key():
    """The front end cannot tell the two backends apart, and must not have to."""
    from fastapi.testclient import TestClient

    from solarest import api
    from solarest.weather import SyntheticClearSky

    api.provider = SyntheticClearSky()
    body = {
        "latitude": 51.5, "longitude": -0.13, "years": 1, "include_hourly": False,
        "system": {"dc_capacity_kw": 4.0, "tilt_deg": 35, "azimuth_deg": 180},
        "consumption": {"annual_kwh": 3500.0},
        "battery": {"usable_capacity_kwh": 5.0},
    }
    over_http = TestClient(api.app).post("/api/sizing", json=body).json()
    in_browser = asyncio.run(webapp.sizing({**body, "synthetic": True}))

    assert set(over_http) == set(in_browser)
    assert set(over_http["balance"]) == set(in_browser["balance"])
    assert over_http["balance"]["avoided_import_kwh"] == pytest.approx(
        in_browser["balance"]["avoided_import_kwh"], rel=1e-6
    )
