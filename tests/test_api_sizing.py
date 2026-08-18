"""The sizing endpoint, and the request model beneath it."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from solarest import api
from solarest.service import (
    ConsumptionRequest,
    SizingRequest,
    run_sizing,
    sizing_to_dict,
)
from solarest.weather import SyntheticClearSky


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "provider", SyntheticClearSky())
    return TestClient(api.app)


def sizing_body(**overrides) -> dict:
    body = {
        "latitude": 51.5,
        "longitude": -0.13,
        "years": 1,
        "include_hourly": False,
        "system": {"dc_capacity_kw": 4.0, "tilt_deg": 35, "azimuth_deg": 180},
        "consumption": {"annual_kwh": 3500.0},
        "battery": {"usable_capacity_kwh": 5.0},
    }
    body.update(overrides)
    return body


# ------------------------------------------------------------- the payload


def test_sizing_returns_a_complete_payload(client):
    payload = client.post("/api/sizing", json=sizing_body()).json()

    assert payload["balance"]["avoided_import_kwh"] > 0
    assert payload["consumption"]["annual_kwh"] == pytest.approx(3500.0, abs=1.0)
    assert len(payload["monthly_balance"]) == 12
    assert len(payload["sizing_curve"]) > 5
    # It still carries everything /api/estimate would have returned.
    assert payload["annual"]["energy_kwh_mean"] > 0
    assert payload["site"]["timezone"]


def test_the_balance_closes_in_the_payload(client):
    balance = client.post("/api/sizing", json=sizing_body()).json()["balance"]
    assert balance["avoided_import_kwh"] == pytest.approx(
        balance["direct_use_kwh"] + balance["from_battery_kwh"], abs=0.2
    )
    assert balance["load_kwh"] == pytest.approx(
        balance["avoided_import_kwh"] + balance["import_kwh"], abs=0.2
    )


def test_monthly_balance_sums_to_the_annual_figures(client):
    payload = client.post("/api/sizing", json=sizing_body()).json()
    months = payload["monthly_balance"]
    assert sum(m["import_kwh"] for m in months) == pytest.approx(
        payload["balance"]["import_kwh"], abs=1.0
    )
    assert sum(m["load_kwh"] for m in months) == pytest.approx(
        payload["balance"]["load_kwh"], abs=1.0
    )


def test_the_options_endpoint_describes_the_archetypes(client):
    archetypes = client.get("/api/options").json()["household_archetypes"]
    by_name = {a["value"]: a for a in archetypes}

    assert "nine_to_five" in by_name
    assert by_name["nine_to_five"]["daytime_window"] == "09:00-17:00"
    assert (
        by_name["nine_to_five"]["daytime_fraction"]
        < by_name["home_all_day"]["daytime_fraction"]
    )


# --------------------------------------------------------- the three ways in


def test_monthly_bills_are_accepted(client):
    bills = [420, 380, 340, 280, 240, 210, 205, 220, 260, 310, 380, 430]
    payload = client.post(
        "/api/sizing", json=sizing_body(consumption={"monthly_kwh": bills})
    ).json()
    assert payload["consumption"]["annual_kwh"] == pytest.approx(sum(bills), abs=1.0)
    assert not payload["consumption"]["measured"]


def test_metered_data_is_accepted_and_marked_as_measured(client):
    hourly = list(np.full(8760, 0.4))
    payload = client.post(
        "/api/sizing", json=sizing_body(consumption={"hourly_kwh": hourly})
    ).json()

    assert payload["consumption"]["measured"]
    assert payload["consumption"]["annual_kwh"] == pytest.approx(3504.0, abs=1.0)
    assert payload["confidence"]["band"] is None
    assert "metered" in payload["confidence"]["note"].lower()


def test_half_hourly_data_is_summed_down_to_hours(client):
    half_hourly = list(np.full(17520, 0.2))
    payload = client.post(
        "/api/sizing", json=sizing_body(consumption={"hourly_kwh": half_hourly})
    ).json()
    assert payload["consumption"]["annual_kwh"] == pytest.approx(3504.0, abs=1.0)


def test_the_daytime_fraction_changes_the_answer(client):
    def avoided(fraction: float) -> float:
        payload = client.post(
            "/api/sizing",
            json=sizing_body(
                consumption={
                    "annual_kwh": 3500.0,
                    "household": {"daytime_fraction": fraction},
                },
                battery={"usable_capacity_kwh": 0.0},
            ),
        ).json()
        return payload["balance"]["avoided_import_kwh"]

    assert avoided(0.45) > avoided(0.15)


# ------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "consumption",
    [
        {},                                            # nothing given
        {"annual_kwh": 3500, "monthly_kwh": [1] * 12},  # two given
        {"annual_kwh": -5},
        {"monthly_kwh": [100] * 11},                    # wrong length
        {"annual_kwh": 3500, "household": {"daytime_fraction": 1.5}},
        {"annual_kwh": 3500, "household": {"archetype": "houseboat"}},
        {"hourly_kwh": [1.0] * 500},                    # not a year
    ],
)
def test_bad_consumption_is_rejected(client, consumption):
    response = client.post("/api/sizing", json=sizing_body(consumption=consumption))
    assert response.status_code == 422


@pytest.mark.parametrize(
    "battery",
    [
        {"usable_capacity_kwh": -1},
        {"round_trip_efficiency": 0.0},
        {"round_trip_efficiency": 1.5},
        {"initial_soc_frac": 2.0},
        {"max_charge_kw": 0.0},
    ],
)
def test_bad_batteries_are_rejected(client, battery):
    assert client.post("/api/sizing", json=sizing_body(battery=battery)).status_code == 422


def test_a_ragged_meter_series_is_reported_as_a_bad_request(client):
    """8761 readings is a whole year plus one, which nothing can align."""
    body = sizing_body(consumption={"hourly_kwh": [0.4] * 8761})
    assert client.post("/api/sizing", json=body).status_code == 400


def test_consumption_requires_exactly_one_source():
    with pytest.raises(ValueError, match="exactly one"):
        ConsumptionRequest()
    with pytest.raises(ValueError, match="exactly one"):
        ConsumptionRequest(annual_kwh=3000.0, monthly_kwh=[250.0] * 12)


# ------------------------------------------------------------ service layer


def test_run_sizing_is_usable_without_http():
    request = SizingRequest(
        latitude=51.5,
        longitude=-0.13,
        years=1,
        consumption=ConsumptionRequest(annual_kwh=3500.0),
    )
    outcome = asyncio.run(run_sizing(request, SyntheticClearSky()))

    assert outcome.sizing.result.avoided_import_kwh > 0
    assert json.dumps(sizing_to_dict(outcome))  # serialises cleanly


def test_the_note_explains_what_a_bills_based_answer_is_worth():
    request = SizingRequest(
        latitude=51.5,
        longitude=-0.13,
        years=1,
        consumption=ConsumptionRequest(annual_kwh=3500.0),
    )
    outcome = asyncio.run(run_sizing(request, SyntheticClearSky()))
    confidence = sizing_to_dict(outcome)["confidence"]

    assert confidence["band"]["spread_pct"] < confidence["band_pv_only"]["spread_pct"]
    assert "assumption" in confidence["note"]


# --------------------------------------------------- the typical-day payload


def test_the_payload_carries_a_day_profile_for_the_chart(client):
    diurnal = client.post("/api/sizing", json=sizing_body()).json()["diurnal_balance"]
    expected = {"generation", "direct", "from_battery", "to_battery", "import", "export"}

    assert set(diurnal["by_month"]) == expected
    assert set(diurnal["by_year"]) == expected
    for grid in diurnal["by_month"].values():
        assert len(grid) == 12 and all(len(row) == 24 for row in grid)
    for row in diurnal["by_year"].values():
        assert len(row) == 24


def test_the_serialised_day_still_balances(client):
    """Rounding for the wire must not break the identity the chart draws."""
    diurnal = client.post("/api/sizing", json=sizing_body()).json()["diurnal_balance"]
    year = diurnal["by_year"]

    for hour in range(24):
        assert year["generation"][hour] == pytest.approx(
            year["direct"][hour] + year["to_battery"][hour] + year["export"][hour],
            abs=1e-3,
        )


def test_the_day_profile_shows_the_battery_working(client):
    year = client.post("/api/sizing", json=sizing_body()).json()["diurnal_balance"]["by_year"]
    charging = sum(year["to_battery"][9:16])
    discharging = sum(year["from_battery"][18:23])

    assert charging > sum(year["to_battery"][18:23])
    assert discharging > sum(year["from_battery"][9:16])


def test_no_battery_leaves_the_day_profile_free_of_storage(client):
    body = sizing_body(battery={"usable_capacity_kwh": 0.0})
    year = client.post("/api/sizing", json=body).json()["diurnal_balance"]["by_year"]

    assert sum(year["to_battery"]) == pytest.approx(0.0)
    assert sum(year["from_battery"]) == pytest.approx(0.0)
