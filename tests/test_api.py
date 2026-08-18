"""HTTP surface and the service layer beneath it."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from solarest import api
from solarest.service import EstimateRequest, SystemRequest, outcome_to_dict, run_estimate
from solarest.weather import ArchiveError, OpenMeteoArchive, SyntheticClearSky


@pytest.fixture
def client(monkeypatch):
    """A test client backed by the offline provider, never the network."""
    monkeypatch.setattr(api, "provider", SyntheticClearSky())
    return TestClient(api.app)


def estimate_body(**overrides) -> dict:
    body = {
        "latitude": 51.5,
        "longitude": -0.13,
        "years": 2,
        "include_hourly": False,
        "system": {"dc_capacity_kw": 4.0, "tilt_deg": 35, "azimuth_deg": 180},
    }
    body.update(overrides)
    return body


# -------------------------------------------------------------- endpoints


def test_health_reports_the_data_source(client):
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["synthetic"] is True
    assert len(payload["available_years"]) == 20


def test_options_describes_the_choices_the_ui_offers(client):
    payload = client.get("/api/options").json()
    assert {m["value"] for m in payload["mounts"]} == {
        "ground_mount", "roof_mount", "roof_integrated",
    }
    assert {m["value"] for m in payload["module_types"]} == {
        "standard", "premium", "thin_film",
    }
    assert payload["default_losses_pct"]["combined_total"] == pytest.approx(14.08, abs=0.01)


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Solar Power Estimation" in response.text


def test_static_assets_are_served(client):
    for asset in ("app.js", "charts.js", "styles.css"):
        assert client.get(f"/static/{asset}").status_code == 200


def test_estimate_returns_a_complete_payload(client):
    payload = client.post("/api/estimate", json=estimate_body()).json()

    assert payload["annual"]["energy_kwh_mean"] > 0
    assert len(payload["monthly"]) == 12
    assert len(payload["diurnal"]) == 12
    assert payload["weather"]["synthetic"] is True
    assert payload["system"]["dc_capacity_kw"] == 4.0
    assert payload["system"]["tilt_deg"] == 35.0
    assert payload["site"]["timezone"]


def test_estimate_omits_the_hourly_series_unless_asked(client):
    lean = client.post("/api/estimate", json=estimate_body()).json()
    assert "ac_kw" not in lean["typical_year"]

    full = client.post("/api/estimate", json=estimate_body(include_hourly=True)).json()
    assert len(full["typical_year"]["ac_kw"]) == 8760


def test_estimate_can_include_the_orientation_search(client):
    payload = client.post("/api/estimate", json=estimate_body(optimise=True)).json()
    assert "orientation" in payload
    assert payload["orientation"]["best_tilt_deg"] >= 0
    assert len(payload["orientation"]["surface"]) > 10


def test_orientation_is_absent_by_default(client):
    assert "orientation" not in client.post("/api/estimate", json=estimate_body()).json()


def test_defaults_fill_in_tilt_and_azimuth_from_latitude(client):
    body = estimate_body(system={"dc_capacity_kw": 3.0})
    payload = client.post("/api/estimate", json=body).json()
    assert payload["system"]["azimuth_deg"] == 180.0  # northern hemisphere
    assert 30.0 < payload["system"]["tilt_deg"] < 50.0


def test_southern_hemisphere_defaults_face_north(client):
    body = estimate_body(latitude=-33.9, longitude=151.2, system={"dc_capacity_kw": 3.0})
    payload = client.post("/api/estimate", json=body).json()
    assert payload["system"]["azimuth_deg"] == 0.0


def test_output_scales_linearly_with_array_size(client):
    """The DC/AC ratio is held fixed, so the inverter grows with the array.

    Nothing in the chain then depends on absolute size, and four times the
    panels must give exactly four times the energy.
    """
    small = client.post("/api/estimate", json=estimate_body(
        system={"dc_capacity_kw": 2.0, "tilt_deg": 35, "azimuth_deg": 180})).json()
    large = client.post("/api/estimate", json=estimate_body(
        system={"dc_capacity_kw": 8.0, "tilt_deg": 35, "azimuth_deg": 180})).json()
    ratio = large["annual"]["energy_kwh_mean"] / small["annual"]["energy_kwh_mean"]
    assert ratio == pytest.approx(4.0, rel=1e-5)


def test_a_fixed_inverter_caps_the_gain_from_more_panels(client):
    """Raising DC/AC instead of the inverter runs into clipping."""
    matched = client.post("/api/estimate", json=estimate_body(
        system={"dc_capacity_kw": 4.0, "dc_ac_ratio": 1.0})).json()
    oversized = client.post("/api/estimate", json=estimate_body(
        system={"dc_capacity_kw": 8.0, "dc_ac_ratio": 2.0})).json()
    # Same inverter, twice the panels: well short of twice the energy.
    ratio = oversized["annual"]["energy_kwh_mean"] / matched["annual"]["energy_kwh_mean"]
    assert 1.0 < ratio < 2.0
    assert oversized["losses"]["clipping_pct"] > matched["losses"]["clipping_pct"]


def test_shading_reduces_output(client):
    clear = client.post("/api/estimate", json=estimate_body(
        system={"dc_capacity_kw": 4.0, "shading_pct": 0})).json()
    shaded = client.post("/api/estimate", json=estimate_body(
        system={"dc_capacity_kw": 4.0, "shading_pct": 40})).json()
    assert shaded["annual"]["energy_kwh_mean"] < clear["annual"]["energy_kwh_mean"] * 0.7


def test_explicit_total_losses_override_the_itemised_stack(client):
    payload = client.post("/api/estimate", json=estimate_body(
        system={"dc_capacity_kw": 4.0, "system_losses_pct": 5.0})).json()
    assert payload["system"]["system_losses_pct"] == pytest.approx(5.0)


# ------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "body",
    [
        {"latitude": 95.0, "longitude": 0.0},
        {"latitude": 0.0, "longitude": 200.0},
        {"latitude": 51.5, "longitude": -0.13, "years": 0},
        {"latitude": 51.5, "longitude": -0.13, "years": 50},
        {"latitude": 51.5, "longitude": -0.13, "system": {"dc_capacity_kw": -1}},
        {"latitude": 51.5, "longitude": -0.13, "system": {"tilt_deg": 120}},
        {"latitude": 51.5, "longitude": -0.13, "system": {"mount": "balloon"}},
        {"latitude": 51.5, "longitude": -0.13, "system": {"dc_ac_ratio": 0}},
    ],
)
def test_invalid_requests_are_rejected(client, body):
    assert client.post("/api/estimate", json=body).status_code == 422


def test_missing_coordinates_are_rejected(client):
    assert client.post("/api/estimate", json={}).status_code == 422


def test_geocode_requires_a_usable_query(client):
    assert client.get("/api/geocode?q=a").status_code == 422


# ---------------------------------------------------------------- exports


def test_csv_export_has_a_row_for_every_hour(client):
    response = client.post("/api/estimate.csv", json=estimate_body())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]

    lines = response.text.strip().splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("timestamp"))
    assert len(lines) - header - 1 == 8760
    assert lines[header] == "timestamp,ac_kw,energy_kwh,poa_w_m2,air_temp_c"
    # Provenance is carried in the comment block.
    assert any("Open-Meteo" in line or "Synthetic" in line for line in lines[:header])


def test_csv_values_line_up_with_the_json_series(client):
    body = estimate_body(include_hourly=True)
    payload = client.post("/api/estimate", json=body).json()
    csv_text = client.post("/api/estimate.csv", json=body).text

    lines = csv_text.strip().splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("timestamp"))
    first = lines[header + 1].split(",")
    noon = lines[header + 12].split(",")
    assert float(first[1]) == pytest.approx(payload["typical_year"]["ac_kw"][0], abs=1e-3)
    assert float(noon[1]) == pytest.approx(payload["typical_year"]["ac_kw"][11], abs=1e-3)


# ------------------------------------------------- upstream failure paths


def test_archive_failures_surface_as_bad_gateway(client, monkeypatch):
    class Broken:
        async def site_info(self, latitude, longitude):
            raise ArchiveError("Open-Meteo returned 429: rate limited")

        async def fetch(self, latitude, longitude, years):
            raise ArchiveError("Open-Meteo returned 429: rate limited")

    monkeypatch.setattr(api, "provider", Broken())
    response = client.post("/api/estimate", json=estimate_body())
    assert response.status_code == 502
    assert "rate limited" in response.json()["detail"]


def test_geocode_failures_surface_as_bad_gateway(client, monkeypatch):
    async def broken(*args, **kwargs):
        from solarest.geocode import GeocodingError

        raise GeocodingError("service unreachable")

    monkeypatch.setattr(api.geocode, "search", broken)
    response = client.get("/api/geocode?q=Bristol")
    assert response.status_code == 502


def test_geocode_results_are_passed_through(client, monkeypatch):
    from solarest.geocode import Place

    async def fake(*args, **kwargs):
        return [Place(name="Bristol", latitude=51.45, longitude=-2.58,
                      country="United Kingdom", admin1="England")]

    monkeypatch.setattr(api.geocode, "search", fake)
    results = client.get("/api/geocode?q=Bristol").json()["results"]
    assert results[0]["label"] == "Bristol, England, United Kingdom"
    assert results[0]["latitude"] == 51.45


# ------------------------------------- the real archive client, mocked out


def build_archive_payload(year: int, hours: int) -> dict:
    times = []
    stamp = f"{year}-01-01T00:00"
    import numpy as np

    base = np.datetime64(f"{year}-01-01T00:00", "h")
    times = [str(base + np.timedelta64(i, "h")) for i in range(hours)]
    return {
        "latitude": 51.5, "longitude": -0.125, "elevation": 25.0,
        "timezone": "GMT", "utc_offset_seconds": 0,
        "hourly_units": {"wind_speed_10m": "m/s", "temperature_2m": "°C"},
        "hourly": {
            "time": times,
            "shortwave_radiation": [120.0] * hours,
            "direct_radiation": [80.0] * hours,
            "diffuse_radiation": [40.0] * hours,
            "temperature_2m": [11.0] * hours,
            "wind_speed_10m": [3.5] * hours,
        },
    }


def test_archive_client_fetches_caches_and_reuses(tmp_path):
    """Exercises the real HTTP path against a stub transport."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        start = request.url.params["start_date"]
        year = int(start[:4])
        # The metadata probe asks for a single day.
        hours = 24 if request.url.params["hourly"] == "temperature_2m" else 48
        payload = build_archive_payload(year, hours)
        if request.url.params.get("timezone") == "auto":
            payload["timezone"] = "Europe/London"
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    archive = OpenMeteoArchive(
        cache_dir=tmp_path, client=httpx.AsyncClient(transport=transport)
    )

    series = asyncio.run(archive.fetch(51.5, -0.125, [2022, 2023]))
    assert len(series) == 96  # two stub years of 48 hours
    assert len(calls) == 2

    # Second call must be served entirely from disk.
    again = asyncio.run(archive.fetch(51.5, -0.125, [2022, 2023]))
    assert len(calls) == 2
    assert again.ghi == pytest.approx(series.ghi)


def test_archive_site_probe_resolves_the_timezone(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        payload = build_archive_payload(2024, 24)
        payload["timezone"] = "Europe/London"
        return httpx.Response(200, json=payload)

    archive = OpenMeteoArchive(
        cache_dir=tmp_path, client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    info = asyncio.run(archive.site_info(51.5, -0.125))
    assert info.timezone == "Europe/London"
    assert info.utc_offset_seconds == 0  # standard time, not BST
    assert info.elevation_m == 25.0


def test_archive_reports_upstream_errors(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"reason": "start_date is out of range"})

    archive = OpenMeteoArchive(
        cache_dir=tmp_path, client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(ArchiveError, match="out of range"):
        asyncio.run(archive.fetch(51.5, -0.125, [1800]))


def test_archive_reports_connection_failures(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("name resolution failed")

    archive = OpenMeteoArchive(
        cache_dir=tmp_path, client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(ArchiveError, match="could not reach"):
        asyncio.run(archive.fetch(51.5, -0.125, [2023]))


# ----------------------------------------------------------- service layer


def test_run_estimate_is_usable_without_http():
    request = EstimateRequest(
        latitude=51.5, longitude=-0.13, years=1,
        system=SystemRequest(dc_capacity_kw=4.0, tilt_deg=35, azimuth_deg=180),
    )
    outcome = asyncio.run(run_estimate(request, SyntheticClearSky()))
    assert outcome.result.annual.energy_kwh_mean > 0
    assert outcome.spec.dc_capacity_kw == 4.0

    payload = outcome_to_dict(outcome, include_hourly=False)
    assert json.dumps(payload)  # serialises cleanly


def test_blank_site_names_are_normalised():
    request = EstimateRequest(latitude=0.0, longitude=0.0, name="   ")
    assert request.name is None
