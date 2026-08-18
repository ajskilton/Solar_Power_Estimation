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

## Battery sizing

Give it what the household uses as well as what the roof makes, and it works
out how much electricity never has to be bought.

```bash
solarest size 51.45 -2.59 --annual-kwh 3500 --battery-kwh 5
solarest size 51.45 -2.59 --monthly-kwh 420,380,340,280,240,210,205,220,260,310,380,430
solarest size 51.45 -2.59 --hourly-csv meter-export.csv        # the good one
```

The dispatch model is a greedy self-consumption controller, which is what a
domestic hybrid inverter does out of the box: use generation the instant it
arrives, store the surplus, discharge after dark. It respects usable capacity,
charge and discharge power limits, and round-trip efficiency.

### Three ways to describe the demand, and what each is worth

| Input | Flag | What it gives you |
|---|---|---|
| Metered interval data | `--hourly-csv` | A firm answer. The load shape is known. |
| Twelve monthly bills | `--monthly-kwh` | A good answer, plus an assumption. |
| One annual figure | `--annual-kwh` | The same, with the months guessed too. |

Interval data is accepted at any regular resolution — hourly, half-hourly,
quarter-hourly — and summed down to hours. A leap year is fine; 29 February is
dropped. Timestamp columns, headers, comments and byte-order marks are ignored,
because supplier exports contain all of them.

### The daytime fraction

Where there is no metered data, the missing information is *when* the household
uses electricity, and it is captured by a single number: the share of a
weekday's consumption falling between 09:00 and 17:00.

| Archetype | Daytime fraction | |
|---|---|---|
| `nine_to_five` | 25% | Out at work. **The default.** |
| `working_from_home` | 35% | No commute, lunchtime bump. |
| `home_all_day` | 39% | Retired, shift work, small children. |
| `flat` | 33% | A diagnostic baseline, not a household. |

Override it directly with `--daytime-fraction 0.31` when neither fits. Weekends
default to somebody being home, because a household that is out on Tuesday is
usually in on Sunday, and weekend days carry slightly more energy than weekdays.

### Why the answer comes with a range

Two households with identical bills can get very different value from the same
battery, so quoting one number from billing data alone would be false precision.
The estimator re-runs the dispatch either side of the daytime assumption and
reports the spread — **twice**, with and without the battery:

```
Varying that assumption moves the battery result by 6%
but the no-battery result by 20%.
```

That gap is the most useful thing here. A battery absorbs the mismatch between
generation and demand, which is exactly what the load-shape assumption governs,
so it also absorbs the error in that assumption. **Sizing a battery from monthly
bills is defensible. Estimating no-battery self-consumption from the same data
is not nearly as sound.**

### The sizing curve

Every run sweeps a range of capacities, because the shape of the curve is the
actual decision aid. Illustrative, for a 4 kWp array making 3,600 kWh against
3,500 kWh of consumption:

```
  kWh   avoided  self-suff  per extra kWh
  0.0     1,328      37.9%            0
  2.0     2,009      57.4%          334
  4.0     2,585      73.9%          263
  5.0     2,729      78.0%          144
  6.5     2,797      79.9%           45      <- knee
  10.0    2,804      80.1%            1
```

The last column is what each extra kWh of capacity buys. It falls off a cliff
once the battery routinely reaches morning with charge to spare, and that knee
is where to stop. `suggested_capacity_kwh` picks it automatically.

### Not modelled

Grid charging on an off-peak tariff (it can *raise* import while *lowering* the
bill, so it belongs with a tariff model); DC coupling, which would capture some
clipped energy; and battery degradation. Hourly resolution also slightly
overstates direct self-consumption — with a battery in the loop the effect is
small, since the battery absorbs that flicker in reality too.

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

Sizing a battery against that year:

```python
from solarest import BatterySpec, HouseholdShape, size_for_household, synthesise

load = synthesise(3500.0, household=HouseholdShape(archetype="nine_to_five"))
sized = size_for_household(result.typical_year, load, BatterySpec(usable_capacity_kwh=5.0))

print(sized.result.avoided_import_kwh, "kWh not bought from the grid")
print(sized.result.self_sufficiency_pct, "% self-sufficient")
print(sized.suggested_capacity_kwh, "kWh suggested")
```

Or from a smart-meter export, which needs no assumptions at all:

```python
from solarest import LoadProfile

load = LoadProfile.from_hourly(readings)   # any regular interval
```

### HTTP API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Liveness, and which data source is configured |
| `GET /api/options` | Mount types, module types, default losses |
| `GET /api/geocode?q=` | Place-name search |
| `POST /api/estimate` | The estimate, as JSON |
| `POST /api/estimate.csv` | The typical year as an hourly CSV |
| `POST /api/sizing` | Battery sizing for a household's consumption |

```bash
curl -X POST localhost:8000/api/estimate -H 'content-type: application/json' \
  -d '{"latitude":51.4545,"longitude":-2.5879,"system":{"dc_capacity_kw":4}}'

curl -X POST localhost:8000/api/sizing -H 'content-type: application/json' \
  -d '{"latitude":51.4545,"longitude":-2.5879,
       "consumption":{"annual_kwh":3500,"household":{"archetype":"nine_to_five"}},
       "battery":{"usable_capacity_kwh":5}}'
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

This matters doubly for battery sizing. Clear-sky weather has no cloudy runs,
so the battery never has to ride out four grey days — exactly the case the
typical year exists to preserve. Offline runs will therefore show optimistic
self-sufficiency *and* place the knee of the sizing curve at a smaller battery
than real weather would. Use `--synthetic` to check the plumbing, never to size
hardware.

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
pytest                       # 321 tests
solarest serve --reload
```

On Windows PowerShell the activation step is `.\.venv\Scripts\Activate.ps1`,
which needs `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` first. The
`solarest` command only exists inside the activated environment; without it,
`python -m solarest.cli` does the same job.

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
  load.py        household demand, from meter data or from bills
  battery.py     dispatch, the sizing curve, diminishing returns
  sizing.py      demand against generation, and what the answer is worth
  service.py     request models and the estimate and sizing use cases
  api.py         FastAPI app
  cli.py         `solarest serve`, `estimate` and `size`
  web/           the single-page frontend (no build step, no dependencies)
```

The frontend is plain HTML, CSS and ES modules with hand-rolled SVG charts —
no bundler, no framework, nothing fetched from a CDN.

---

## Where this is going

Battery sizing is in, as a prototype: model layer, HTTP endpoint and CLI, with
the web UI still to come. What is missing is money and panels.

- **Tariffs.** Everything here is in kWh. Turning that into pounds needs import
  and export rates, and standing charges — at which point off-peak grid
  charging becomes worth modelling, since it can raise import while lowering
  the bill.
- **Panel sizing.** `optimise_orientation` already takes a pluggable objective.
  Today it maximises total generation; handing it a `LoadProfile` makes the
  same search maximise self-consumption or bill savings instead, and sweeping
  array size alongside battery size gives a surface rather than a curve.
- **Payback.** Once tariffs and capital costs are in, the sizing curve becomes
  a net-present-value curve, and the knee moves.

---

## Credits

Weather from [Open-Meteo](https://open-meteo.com/) (ECMWF ERA5 reanalysis),
free for non-commercial use. PV model after NREL's
[PVWatts v5](https://www.nrel.gov/docs/fy14osti/62641.pdf). Sky-diffuse
transposition after Perez et al. (1990); cell temperature after King et al.
(2004); typical-year selection after the Sandia TMY method.
