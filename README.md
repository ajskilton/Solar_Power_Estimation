# Solar Power Estimation

Tool designed to help households decide whether to invest in solar power.

Give it a location and a rough description of a PV system, and it estimates how
much electricity that system would generate over a year — hour by hour, from a
decade of real historical weather rather than a rule of thumb.

```bash
pip install -e ".[dev]"
solarest serve            # then open http://127.0.0.1:8000
```

---

## What it does

The estimator pulls **hourly historical weather** for the site from the
[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
— the ECMWF **ERA5 reanalysis**, the same dataset used for climate research —
and runs every hour of the last ten years through a physical model of the array:

| Stage | Model |
|---|---|
| Sun position | NOAA solar calculator (condensed Meeus) |
| Irradiance on the array plane | Perez et al. (1990), with HDKR as an alternative |
| Reflection off the glass | ASHRAE incidence-angle modifier |
| Cell temperature | Sandia array performance model, by mounting style |
| DC output | PVWatts v5, linear in irradiance with a temperature derate |
| Losses | PVWatts default stack (soiling, shading, mismatch, wiring, …) |
| Inverter | PVWatts v5 part-load efficiency curve, with AC clipping |

Because it runs ten separate years, the answer comes with the spread between a
good year and a poor one — not a single number pretending to be certain.

### What you get back

- **Expected annual generation**, with P10/P50/P90 and the best and worst years.
- **Specific yield, performance ratio, capacity factor** — the numbers you can
  compare against an installer's quotation.
- **Monthly generation**, with the range across the record.
- **A diurnal profile** — average output by hour and month, in local standard time.
- **A loss breakdown** — where the sunlight goes between the panel and the meter.
- **An 8760-hour typical year**, downloadable as CSV.
- Optionally, the **best tilt and azimuth** for the site.

---

## The typical year, and why it is not an average

The 8760-hour series is built by the **Sandia TMY method**: for each calendar
month, the estimator picks the one real month from the record whose daily
distributions of irradiance and temperature best match the long-run
distribution (lowest weighted Finkelstein–Schafer statistic), and splices that
whole month in.

It would have been easier to average the ten years together hour by hour. That
would be wrong for the next stage of this project. Averaging smooths away cloudy
runs — the four grey days in a row that decide how big a battery has to be — and
would quietly make storage look better than it is. Keeping real months preserves
that variability.

The typical year's annual total therefore lands *near* the ten-year mean, not
exactly on it. The headline figures come from the full record; the typical year
is for shape.

Timestamps are in **local standard time, with no daylight saving**, the same
convention TMY files and PVGIS use. Every day is 24 hours long, so hour-of-day
profiles stay comparable across the seasons.

---

## Using it

### The website

```bash
solarest serve --port 8000
```

Search for a place, describe the system, press estimate. Nothing is stored; the
only state on disk is a weather cache.

### The command line

```bash
solarest estimate 51.4545 -2.5879 --kwp 4 --tilt 35 --azimuth 180
solarest estimate 51.4545 -2.5879 --optimise --years 15
solarest estimate 51.4545 -2.5879 --synthetic     # offline, no network
```

### As a library

```python
import asyncio
from solarest import OpenMeteoArchive, SystemSpec, precompute, summarise, default_year_range

archive = OpenMeteoArchive()
weather = asyncio.run(archive.fetch(51.4545, -2.5879, default_year_range(10)))
site = precompute(weather)

result = summarise(site.simulate(SystemSpec(dc_capacity_kw=4.0, tilt_deg=35, azimuth_deg=180)))
print(result.annual.energy_kwh_mean, "kWh/year")
print(result.typical_year.ac_kw)  # 8760 hourly values, kW
```

`precompute` does the site-wide work once — sun position and irradiance
reconciliation — so trying many system configurations at the same site is cheap.
That is what makes the orientation search interactive.

### HTTP API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Liveness, and which data source is configured |
| `GET /api/options` | Mount types, module types, default losses |
| `GET /api/geocode?q=` | Place-name search |
| `POST /api/estimate` | The estimate, as JSON |
| `POST /api/estimate.csv` | The typical year as an hourly CSV |

```bash
curl -X POST localhost:8000/api/estimate -H 'content-type: application/json' \
  -d '{"latitude":51.4545,"longitude":-2.5879,"system":{"dc_capacity_kw":4}}'
```

Interactive documentation is at `/docs`.

---

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SOLAREST_CACHE_DIR` | `.cache/weather` | Where fetched years are cached |
| `SOLAREST_SYNTHETIC` | unset | `1` serves modelled clear-sky data instead of real weather |

Each year of weather is cached separately as a compressed `.npz`, so a repeat
visit or a second system configuration at the same site costs no API calls.
Open-Meteo is free for non-commercial use with generous but real rate limits;
the cache exists to stay well inside them.

`SOLAREST_SYNTHETIC=1` swaps in an Ineichen–Perez clear-sky generator for
offline development. It is clearly labelled in the API response and behind a
banner in the UI: it has no clouds, so it overstates yield substantially — for
London it returns roughly 1,700 kWh/kWp against a real figure nearer 950.

---

## Accuracy, and what this does not model

Cross-checked against [pvlib](https://pvlib-python.readthedocs.io/), an
independent implementation of the same published models, the full chain agrees
to within **0.008%** on annual AC energy (`tests/test_against_pvlib.py`). The
residual is dominated by the different solar-position algorithm, not the PV
models.

That is agreement between implementations, not agreement with reality. The real
error budget is dominated by the inputs:

- **ERA5 is a ~25 km grid.** It cannot see your valley, your hill, or your
  microclimate. Expect a few percent of bias in complex terrain, more in coastal
  and mountain locations.
- **Shading is a single user-supplied number.** There is no horizon profile and
  no near-field obstruction modelling, so a chimney or a neighbouring roof is
  yours to estimate. This is usually the largest error for a real rooftop.
- **No soiling dynamics, no snow cover model, no degradation over the system's
  life** — the PVWatts stack treats these as flat annual derates.
- **No spectral correction**, and no bifacial or tracking systems.

Treat the output as a planning guide with a realistic uncertainty of roughly
±10% for a well-characterised site, and worse where shading is significant.

---

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                       # 182 tests
solarest serve --reload
```

Layout:

```
src/solarest/
  solarpos.py    sun position, air mass, extraterrestrial irradiance
  irradiance.py  Perez / HDKR transposition, incidence-angle modifiers
  pvmodel.py     cell temperature, DC model, losses, inverter
  weather.py     Open-Meteo archive client, cache, offline clear-sky provider
  simulate.py    the model chain; SitePrecompute for cheap re-runs
  results.py     annual / monthly / diurnal aggregation, TMY assembly
  optimize.py    tilt and azimuth search
  service.py     request models and the estimate use case
  api.py         FastAPI app
  cli.py         `solarest serve` and `solarest estimate`
  web/           the single-page frontend (no build step, no dependencies)
```

The frontend is plain HTML, CSS and ES modules with hand-rolled SVG charts —
no bundler, no framework, nothing fetched from a CDN.

---

## Where this is going

The next stage is sizing: take a household's historical energy bills or
half-hourly meter data, line it up against the 8760-hour typical year, and
recommend a panel and battery size.

The pieces are already shaped for it:

- The typical year keeps real day-to-day variability, which is what battery
  sizing depends on.
- It is in local standard time with no DST gaps, so it lines up cleanly with
  metered consumption.
- `optimise_orientation` takes a pluggable objective. Today it maximises total
  generation; give it a load profile and the same search maximises
  self-consumption or bill savings instead.

---

## Credits

Weather from [Open-Meteo](https://open-meteo.com/) (ECMWF ERA5 reanalysis),
free for non-commercial use. PV model after NREL's
[PVWatts v5](https://www.nrel.gov/docs/fy14osti/62641.pdf). Sky-diffuse
transposition after Perez et al. (1990); cell temperature after King et al.
(2004); typical-year selection after the Sandia TMY method.
