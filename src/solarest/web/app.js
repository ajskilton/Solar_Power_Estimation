/**
 * Page controller: reads the form, calls the API, renders the results.
 */

import { barChart, curveChart, dayProfile, heatmap, stackedBar, legend, table } from "./charts.js";

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
  "August", "September", "October", "November", "December"];
const COMPASS = [
  [0, "North"], [22.5, "NNE"], [45, "North-east"], [67.5, "ENE"], [90, "East"],
  [112.5, "ESE"], [135, "South-east"], [157.5, "SSE"], [180, "South"],
  [202.5, "SSW"], [225, "South-west"], [247.5, "WSW"], [270, "West"],
  [292.5, "WNW"], [315, "North-west"], [337.5, "NNW"], [360, "North"],
];

const $ = (id) => document.getElementById(id);
const number = (value, digits = 0) =>
  value.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });

/** State that outlives a single render. */
const state = {
  lastRequest: null, lastResult: null, searchTimer: null,
  archetypes: [], meterReadings: null, dayMonth: "year",
};

/* ------------------------------------------------------------- helpers */

function compassName(degrees) {
  const normalised = ((degrees % 360) + 360) % 360;
  let best = COMPASS[0];
  for (const entry of COMPASS) {
    if (Math.abs(entry[0] - normalised) < Math.abs(best[0] - normalised)) best = entry;
  }
  return best[1];
}

/** Ask the server what mount types, module types and defaults exist. */
async function loadOptions() {
  try {
    const response = await fetch("/api/options");
    if (!response.ok) return;
    const data = await response.json();
    for (const item of data.mounts) {
      $("mount").add(new Option(item.label, item.value));
    }
    $("mount").value = "roof_mount";
    for (const item of data.module_types) {
      $("module").add(
        new Option(`${item.label} (${item.temperature_coefficient_pct_per_c}%/°C)`, item.value)
      );
    }
    $("module").value = "premium";
    state.archetypes = data.household_archetypes || [];
    for (const item of state.archetypes) {
      $("archetype").add(new Option(item.label, item.value));
    }
    $("archetype").value = "nine_to_five";
    applyArchetypeDefault();

    $("years").max = String(data.max_years);
    $("years").value = String(data.default_years);
    $("years-value").textContent = String(data.default_years);
  } catch {
    // Options are cosmetic; the form still works with whatever is selected.
  }
}

/* -------------------------------------------------------- place search */

function renderPlaces(places) {
  const list = $("place-results");
  if (!places.length) {
    list.hidden = true;
    return;
  }
  list.replaceChildren(
    ...places.map((place) => {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.innerHTML =
        `${place.label}<span class="muted">${place.latitude.toFixed(3)}, ` +
        `${place.longitude.toFixed(3)}${place.elevation_m != null ? ` · ${Math.round(place.elevation_m)} m` : ""}</span>`;
      button.addEventListener("click", () => {
        $("latitude").value = place.latitude.toFixed(4);
        $("longitude").value = place.longitude.toFixed(4);
        $("site-label").textContent = place.label;
        $("place").value = place.label;
        list.hidden = true;
        applyLatitudeDefaults(place.latitude);
      });
      item.appendChild(button);
      return item;
    })
  );
  list.hidden = false;
}

async function searchPlaces(query) {
  if (query.trim().length < 2) {
    $("place-results").hidden = true;
    return;
  }
  try {
    const response = await fetch(`/api/geocode?q=${encodeURIComponent(query)}`);
    if (!response.ok) throw new Error("search failed");
    renderPlaces((await response.json()).results);
  } catch {
    $("place-results").hidden = true;
  }
}

/** Nudge tilt and direction to sensible values for a newly picked latitude. */
function applyLatitudeDefaults(latitude) {
  const tilt = Math.round(Math.min(60, Math.max(5, 0.76 * Math.abs(latitude) + 3.1)));
  $("tilt").value = String(tilt);
  $("tilt-value").textContent = `${tilt}°`;
  const azimuth = latitude >= 0 ? 180 : 0;
  $("azimuth").value = String(azimuth);
  $("azimuth-value").textContent = compassName(azimuth);
}

/* -------------------------------------------------------------- request */

function buildRequest(includeHourly) {
  return {
    latitude: Number($("latitude").value),
    longitude: Number($("longitude").value),
    name: $("site-label").textContent || null,
    years: Number($("years").value),
    include_hourly: includeHourly,
    optimise: $("optimise").checked,
    system: {
      dc_capacity_kw: Number($("capacity").value),
      tilt_deg: Number($("tilt").value),
      azimuth_deg: Number($("azimuth").value),
      mount: $("mount").value,
      module_type: $("module").value,
      dc_ac_ratio: Number($("dcac").value),
      albedo: Number($("albedo").value),
      shading_pct: Number($("shading").value),
    },
  };
}

/** The demand and battery half of a sizing request. */
function buildSizingRequest(includeHourly) {
  const mode = document.querySelector('input[name="demand-mode"]:checked').value;
  const consumption = { household: { archetype: $("archetype").value } };

  if (mode === "meter") {
    consumption.hourly_kwh = state.meterReadings;
  } else if (mode === "monthly") {
    consumption.monthly_kwh = monthInputs().map((input) => Number(input.value) || 0);
  } else {
    consumption.annual_kwh = Number($("annual-kwh").value);
  }
  // Only meaningful for a synthesised profile; harmless but noisy otherwise.
  if (mode !== "meter") {
    consumption.household.daytime_fraction = Number($("daytime").value) / 100;
  }

  return {
    ...buildRequest(includeHourly),
    consumption,
    battery: { usable_capacity_kwh: Number($("battery-kwh").value) },
  };
}

async function runEstimate(event) {
  event.preventDefault();
  const sizing = $("enable-sizing").checked;

  let request;
  try {
    request = sizing ? buildSizingRequest(false) : buildRequest(false);
  } catch (error) {
    $("form-error").textContent = String(error.message || error);
    $("form-error").hidden = false;
    return;
  }
  state.lastRequest = request;

  $("form-error").hidden = true;
  $("empty").hidden = true;
  $("output").hidden = true;
  $("loading").hidden = false;
  $("run").disabled = true;
  $("loading-text").textContent = $("optimise").checked
    ? `Fetching ${request.years} years of weather, then searching orientations…`
    : `Fetching ${request.years} years of hourly weather…`;

  try {
    const response = await fetch(sizing ? "/api/sizing" : "/api/estimate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `request failed (${response.status})`);
    state.lastResult = payload;
    render(payload);
    $("output").hidden = false;
  } catch (error) {
    $("form-error").textContent = String(error.message || error);
    $("form-error").hidden = false;
    $("empty").hidden = false;
  } finally {
    $("loading").hidden = true;
    $("run").disabled = false;
  }
}

/* --------------------------------------------------------------- render */

function render(data) {
  $("synthetic-banner").hidden = !data.weather.synthetic;

  renderHeadline(data);
  renderMonthly(data);
  renderDiurnal(data);
  renderLosses(data);
  renderAnnual(data);
  renderOrientation(data);
  renderSizing(data);
}

function renderHeadline(data) {
  const annual = data.annual;
  $("hero-kwh").textContent = number(annual.energy_kwh_mean);
  $("hero-range").textContent =
    `Typical spread ${number(annual.energy_kwh_p10)} – ${number(annual.energy_kwh_p90)} kWh ` +
    `(worst year in the record ${number(annual.energy_kwh_min)}, best ${number(annual.energy_kwh_max)})`;

  const stats = [
    ["Specific yield", `${number(annual.specific_yield_kwh_per_kwp)}`, "kWh per kWp per year"],
    ["Performance ratio", annual.performance_ratio.toFixed(2), "delivered vs. ideal"],
    ["Capacity factor", `${(annual.capacity_factor * 100).toFixed(1)}%`, "of nameplate, all year"],
    ["Plane irradiation", `${number(annual.poa_kwh_m2)}`, `kWh/m² (horizontal ${number(annual.ghi_kwh_m2)})`],
    ["Peak output", `${annual.peak_ac_kw.toFixed(2)} kW`, `inverter ${data.system.inverter_ac_kw} kW AC`],
    ["Year-to-year swing", `±${annual.variability_pct.toFixed(1)}%`, `over ${annual.years_used.length} years`],
  ];
  $("stat-grid").replaceChildren(
    ...stats.map(([label, value, note]) => {
      const wrapper = document.createElement("div");
      wrapper.className = "stat";
      wrapper.innerHTML =
        `<dt>${label}</dt><dd>${value}<span class="stat-note">${note}</span></dd>`;
      return wrapper;
    })
  );

  const years = data.annual.years_used;
  $("provenance").textContent =
    `${data.system.dc_capacity_kw} kWp at ${data.system.tilt_deg}° tilt facing ` +
    `${compassName(data.system.azimuth_deg)} (${data.system.azimuth_deg}°), ` +
    `${data.system.mount_label.toLowerCase()}. ` +
    `Modelled hour by hour over ${years.length} years (${years[0]}–${years[years.length - 1]}) ` +
    `of ${data.weather.source}. Site cell ${data.site.latitude}, ${data.site.longitude} ` +
    `at ${Math.round(data.site.elevation_m)} m, ${data.site.timezone}.`;
}

function renderMonthly(data) {
  const values = data.monthly.map((m) => m.energy_kwh);
  const ranges = data.monthly.map((m) => [m.energy_kwh_min, m.energy_kwh_max]);
  const best = data.monthly.reduce((a, b) => (b.energy_kwh > a.energy_kwh ? b : a));
  const worst = data.monthly.reduce((a, b) => (b.energy_kwh < a.energy_kwh ? b : a));

  $("monthly-sub").textContent =
    `${MONTH_NAMES[best.month - 1]} is the strongest month at ${number(best.energy_kwh)} kWh; ` +
    `${MONTH_NAMES[worst.month - 1]} the weakest at ${number(worst.energy_kwh)} kWh — ` +
    `a factor of ${(best.energy_kwh / Math.max(worst.energy_kwh, 0.01)).toFixed(1)}. ` +
    `Whiskers show the best and worst occurrence of each month in the record.`;

  barChart($("monthly-chart"), {
    labels: MONTHS,
    values,
    ranges,
    unit: "kWh",
    rangeLabel: "best–worst year",
    format: (v) => number(v),
  });

  table(
    $("monthly-table"),
    ["Month", "Mean kWh", "Min kWh", "Max kWh", "Daily mean kWh", "POA kWh/m²", "Air °C", "Cell °C"],
    data.monthly.map((m) => [
      MONTH_NAMES[m.month - 1],
      number(m.energy_kwh), number(m.energy_kwh_min), number(m.energy_kwh_max),
      m.daily_mean_kwh.toFixed(1), number(m.poa_kwh_m2),
      m.mean_temp_c.toFixed(1), m.mean_cell_temp_c.toFixed(1),
    ])
  );
}

function renderDiurnal(data) {
  const colLabels = Array.from({ length: 24 }, (_, h) => (h % 3 === 0 ? `${h}` : ""));
  heatmap($("diurnal-chart"), {
    grid: data.diurnal,
    rowLabels: MONTHS,
    colLabels,
    unit: "kW",
    format: (v) => v.toFixed(2),
    describe: (r, c) =>
      `<strong>${MONTH_NAMES[r]}, ${String(c).padStart(2, "0")}:00–${String((c + 1) % 24).padStart(2, "0")}:00</strong>` +
      `<br>${data.diurnal[r][c].toFixed(2)} kW average` +
      `<br><span class="tip-muted">${(data.diurnal[r][c] * 30).toFixed(1)} kWh over the month</span>`,
  });

  table(
    $("diurnal-table"),
    ["Month", ...Array.from({ length: 24 }, (_, h) => `${h}h`)],
    data.diurnal.map((row, r) => [MONTH_NAMES[r], ...row.map((v) => v.toFixed(2))])
  );
}

function renderLosses(data) {
  const style = getComputedStyle(document.body);
  const slot = (n) => style.getPropertyValue(`--series-${n}`).trim();
  const losses = data.losses;

  // Fixed categorical order: delivered first, then each loss stage.
  const segments = [
    { label: "Delivered as AC", value: losses.delivered_pct, color: slot(1) },
    { label: "Reflection off the glass", value: losses.reflection_pct, color: slot(2) },
    { label: "Heat (cells above 25 °C)", value: losses.temperature_pct, color: slot(3) },
    { label: "System losses", value: losses.system_losses_pct, color: slot(4) },
    { label: "Inverter conversion", value: losses.inverter_pct, color: slot(5) },
    { label: "Inverter clipping", value: losses.clipping_pct, color: slot(6) },
  ].filter((s) => s.value > 0.01);

  $("losses-sub").textContent =
    `Of the ${number(losses.poa_kwh_m2)} kWh/m² landing on the panels each year, ` +
    `${losses.delivered_pct.toFixed(0)}% reaches the meter as electricity. ` +
    `Percentages are shares of the energy hitting the array.`;

  stackedBar($("loss-chart"), segments, "%");
  legend($("loss-legend"), segments.map((s) => ({
    label: s.label, color: s.color, value: `${s.value.toFixed(1)}%`,
  })));
}

function renderAnnual(data) {
  const years = data.annual.per_year.filter((y) => y.complete);
  $("annual-sub").textContent =
    `Each bar is one real year of weather. The spread is the honest uncertainty in ` +
    `any single year's output — the long-run average is the number to plan around.`;

  barChart($("annual-chart"), {
    labels: years.map((y) => String(y.year)),
    values: years.map((y) => y.energy_kwh),
    mean: data.annual.energy_kwh_mean,
    unit: "kWh",
    format: (v) => number(v),
  });

  table(
    $("annual-table"),
    ["Year", "kWh", "vs. mean", "POA kWh/m²", "GHI kWh/m²", "Mean air °C"],
    years.map((y) => [
      y.year, number(y.energy_kwh),
      `${((y.energy_kwh / data.annual.energy_kwh_mean - 1) * 100).toFixed(1)}%`,
      number(y.poa_kwh_m2), number(y.ghi_kwh_m2), y.mean_temp_c.toFixed(1),
    ])
  );
}

function renderOrientation(data) {
  const panel = $("orientation-panel");
  if (!data.orientation) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  const best = data.orientation;

  $("orientation-sub").textContent =
    best.improvement_pct > 0.5
      ? `Facing ${compassName(best.best_azimuth_deg)} at ${best.best_tilt_deg}° would yield ` +
        `${number(best.best_annual_kwh)} kWh — ${best.improvement_pct.toFixed(1)}% more than your ` +
        `current ${data.system.tilt_deg}° / ${compassName(data.system.azimuth_deg)}.`
      : `Your ${data.system.tilt_deg}° / ${compassName(data.system.azimuth_deg)} setup is within ` +
        `${Math.max(0, best.improvement_pct).toFixed(1)}% of the best possible ` +
        `(${best.best_tilt_deg}° facing ${compassName(best.best_azimuth_deg)}). Orientation is not worth changing.`;

  // Pivot the sampled surface into a tilt x azimuth grid.
  const tilts = [...new Set(best.surface.map((c) => c.tilt_deg))].sort((a, b) => a - b);
  const azimuths = [...new Set(best.surface.map((c) => c.azimuth_deg))].sort((a, b) => {
    const centre = data.site.latitude >= 0 ? 180 : 0;
    const rel = (v) => ((v - centre + 540) % 360) - 180;
    return rel(a) - rel(b);
  });
  const lookup = new Map(best.surface.map((c) => [`${c.tilt_deg}|${c.azimuth_deg}`, c.annual_kwh]));
  const grid = tilts.map((t) => azimuths.map((a) => lookup.get(`${t}|${a}`) ?? 0));

  heatmap($("orientation-chart"), {
    grid,
    rowLabels: tilts.map((t) => `${t}°`),
    // Degrees rather than compass names: at 20° spacing two adjacent columns
    // round to the same name, and the name is in the tooltip anyway.
    colLabels: azimuths.map((a) => `${a}°`),
    unit: "kWh",
    format: (v) => number(v),
    // Every orientation here generates a lot; the story is the difference
    // between them, so span the ramp across the sampled range.
    baseline: "range",
    describe: (r, c) =>
      `<strong>${tilts[r]}° tilt, facing ${compassName(azimuths[c])}</strong>` +
      `<br>${number(grid[r][c])} kWh per year` +
      `<br><span class="tip-muted">${((grid[r][c] / best.best_annual_kwh - 1) * 100).toFixed(1)}% vs. best</span>`,
  });

  $("apply-orientation").onclick = () => {
    $("tilt").value = String(Math.round(best.best_tilt_deg));
    $("tilt-value").textContent = `${Math.round(best.best_tilt_deg)}°`;
    $("azimuth").value = String(Math.round(best.best_azimuth_deg / 5) * 5);
    $("azimuth-value").textContent = compassName(best.best_azimuth_deg);
    $("optimise").checked = false;
    $("run").click();
  };
}

/* -------------------------------------------------------- battery sizing */

/** Read a semantic flow colour off the document. */
const flow = (name) =>
  getComputedStyle(document.body).getPropertyValue(`--flow-${name}`).trim();

/** The twelve monthly consumption inputs, in calendar order. */
const monthInputs = () => Array.from(document.querySelectorAll("#month-inputs input"));

/** Move the daytime slider to whatever the chosen archetype implies. */
function applyArchetypeDefault() {
  const chosen = state.archetypes.find((a) => a.value === $("archetype").value);
  if (!chosen) return;
  const percent = Math.round(chosen.daytime_fraction * 100);
  $("daytime").value = String(percent);
  $("daytime-value").textContent = `${percent}%`;
}

/** Build the twelve month boxes, pre-filled with a plausible split. */
function buildMonthInputs() {
  const seasonal = [1.24, 1.14, 1.05, 0.94, 0.85, 0.79, 0.79, 0.82, 0.88, 1.00, 1.13, 1.25];
  const total = seasonal.reduce((a, b) => a + b, 0);
  $("month-inputs").replaceChildren(
    ...MONTHS.map((label, index) => {
      const wrapper = document.createElement("div");
      const id = `month-kwh-${index}`;
      wrapper.innerHTML =
        `<label for="${id}">${label}</label>` +
        `<input type="number" id="${id}" min="0" max="20000" step="1" ` +
        `value="${Math.round((3500 * seasonal[index]) / total)}">`;
      return wrapper;
    })
  );
}

/**
 * Parse a smart-meter export in the browser.
 *
 * Deliberately forgiving, because supplier exports are: blank lines, comments
 * and a header row are skipped, and where a row has several fields the last
 * numeric one is taken — which handles `timestamp,kwh` without being told
 * which column is which. Mirrors the CLI's reader.
 */
function parseMeterCsv(text) {
  const readings = [];
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim().replace(/^﻿/, "");
    if (!line || line.startsWith("#")) continue;
    for (const field of line.split(",").reverse()) {
      const value = Number(field.trim());
      if (field.trim() !== "" && Number.isFinite(value)) {
        readings.push(value);
        break;
      }
    }
  }
  return readings;
}

async function loadMeterFile(file) {
  const status = $("meter-status");
  if (!file) return;
  try {
    const readings = parseMeterCsv(await file.text());
    const perHour = readings.length / 8760;
    const known = { 1: "hourly", 2: "half-hourly", 4: "quarter-hourly", 60: "minute" };
    const leap = readings.length % (366 * 24) === 0;

    if (!readings.length) throw new Error("no numeric readings found");
    if (readings.length % 8760 !== 0 && !leap) {
      throw new Error(
        `${readings.length.toLocaleString()} readings is not a whole year at a ` +
        `fixed interval — 8,760 (hourly) or 17,520 (half-hourly) are the usual shapes`
      );
    }
    state.meterReadings = readings;
    const total = readings.reduce((a, b) => a + b, 0);
    status.textContent =
      `${readings.length.toLocaleString()} readings` +
      `${known[perHour] ? ` (${known[perHour]})` : ""}, ` +
      `${number(total)} kWh over the year. No assumptions needed.`;
    status.classList.remove("error");
  } catch (error) {
    state.meterReadings = null;
    status.textContent = `Could not read that file: ${error.message}`;
    status.classList.add("error");
  }
}

/** Which demand pane is showing, and whether the household knobs matter. */
function applyDemandMode() {
  const mode = document.querySelector('input[name="demand-mode"]:checked').value;
  $("demand-annual").hidden = mode !== "annual";
  $("demand-monthly").hidden = mode !== "monthly";
  $("demand-meter").hidden = mode !== "meter";
  // Metered data already says when electricity is used; nothing to assume.
  $("household-fields").hidden = mode === "meter";
}

/** Pull one day's worth of every balance series out of the payload. */
function daySeries(data, selection) {
  const source = data.diurnal_balance;
  const pick = (name) =>
    selection === "year" ? source.by_year[name] : source.by_month[name][selection];
  return {
    generation: pick("generation"),
    direct: pick("direct"),
    fromBattery: pick("from_battery"),
    toBattery: pick("to_battery"),
    imported: pick("import"),
    exported: pick("export"),
  };
}

function renderSizing(data) {
  const panel = $("sizing-output");
  if (!data.balance) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;

  renderSizingHeadline(data);
  renderMonthPicker(data);
  renderDay(data);
  renderCurve(data);
  renderSizingMonthly(data);
  renderConfidence(data);
}

function renderSizingHeadline(data) {
  const balance = data.balance;
  const battery = data.battery;

  $("sizing-hero").textContent = number(balance.avoided_import_kwh);

  const band = data.confidence.band;
  $("sizing-range").textContent = band
    ? `Between ${number(band.low_kwh)} and ${number(band.high_kwh)} kWh, depending on ` +
      `when the household actually uses electricity`
    : `Straight from your meter data — no assumption about when electricity is used`;

  const stats = [
    ["Self-sufficiency", `${balance.self_sufficiency_pct.toFixed(0)}%`,
      `of ${number(balance.load_kwh)} kWh used`],
    ["The battery's share", `${number(balance.battery_contribution_kwh)}`,
      `kWh beyond panels alone (${number(balance.pv_only_avoided_import_kwh)})`],
    ["Still bought", `${number(balance.import_kwh)}`, "kWh from the grid"],
    ["Exported", `${number(balance.export_kwh)}`, "kWh sent back"],
    ["Self-consumption", `${balance.self_consumption_pct.toFixed(0)}%`,
      `of ${number(balance.generation_kwh)} kWh generated`],
    ["Battery cycles", balance.equivalent_full_cycles.toFixed(0),
      `full cycles a year, ${battery.usable_capacity_kwh} kWh usable`],
  ];
  $("sizing-stats").replaceChildren(
    ...stats.map(([label, value, note]) => {
      const wrapper = document.createElement("div");
      wrapper.className = "stat";
      wrapper.innerHTML = `<dt>${label}</dt><dd>${value}<span class="stat-note">${note}</span></dd>`;
      return wrapper;
    })
  );

  $("sizing-provenance").textContent =
    `${number(balance.load_kwh)} kWh a year against a ${data.system.dc_capacity_kw} kWp array ` +
    `making ${number(balance.generation_kwh)} kWh, through a ${battery.usable_capacity_kwh} kWh ` +
    `battery charging and discharging at up to ${battery.max_charge_kw} kW with ` +
    `${(battery.round_trip_efficiency * 100).toFixed(0)}% round-trip efficiency. ` +
    `${data.consumption.source}.`;
}

function renderMonthPicker(data) {
  const host = $("day-months");
  const options = [["year", "Year"], ...MONTHS.map((m, i) => [i, m])];

  host.replaceChildren(
    ...options.map(([value, label]) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = label;
      button.setAttribute("aria-pressed", String(state.dayMonth === value));
      button.addEventListener("click", () => {
        state.dayMonth = value;
        renderMonthPicker(data);
        renderDay(data);
      });
      return button;
    })
  );
}

/**
 * One sentence on what this particular day does.
 *
 * The bands get thin in midsummer, when a household's demand is a fraction of
 * what the roof makes, so the reading is spelled out rather than left to the
 * pixels.
 */
function dayInsight(day) {
  const hh = (h) => `${String(h).padStart(2, "0")}:00`;
  const hours = (series) => series.filter((v) => v > 0.01).length;
  const peakHour = (series) => series.indexOf(Math.max(...series));
  const total = (series) => series.reduce((a, b) => a + b, 0);

  const boughtHours = hours(day.imported);
  const batteryHours = hours(day.fromBattery);
  const batteryNote = batteryHours
    ? ` The battery is supplying the house for ${batteryHours} of the 24, ` +
      `hardest around ${hh(peakHour(day.fromBattery))}.`
    : "";

  if (!boughtHours) {
    return `On this day the grid is never touched.${batteryNote}`;
  }
  const demand = total(day.direct) + total(day.fromBattery) + total(day.imported);
  const share = total(day.imported) / Math.max(1e-9, demand);

  // Grid hours usually wrap around midnight, so a first-to-last span would
  // read as the whole day. The count and the worst hour say more.
  return (
    `Grid electricity is needed for ${boughtHours} ${boughtHours === 1 ? "hour" : "hours"} ` +
    `of the 24, peaking around ${hh(peakHour(day.imported))}, and covers ` +
    `${(share * 100).toFixed(0)}% of the day's use.${batteryNote}`
  );
}

function renderDay(data) {
  const day = daySeries(data, state.dayMonth);
  const when = state.dayMonth === "year"
    ? "averaged over the whole year"
    : `an average ${MONTH_NAMES[state.dayMonth]} day`;

  const supply = [
    { label: "Sunlight, used as it arrives", color: flow("solar"), values: day.direct },
    { label: "From the battery", color: flow("battery"), values: day.fromBattery },
    { label: "Bought from the grid", color: flow("grid"), values: day.imported },
  ];
  const surplus = [
    { label: "Stored in the battery", color: flow("charge"), values: day.toBattery },
    { label: "Exported to the grid", color: flow("export"), values: day.exported },
  ];

  const demandAt = (h) => day.direct[h] + day.fromBattery[h] + day.imported[h];
  const total = (series) => series.reduce((a, b) => a + b, 0);

  $("day-sub").textContent =
    `Where each kilowatt came from and went, hour by hour, ${when}, in local ` +
    `standard time. Above the line is demand being met; below it is generation ` +
    `with nowhere else to go. The dashed line is total output. ` +
    dayInsight(day);

  dayProfile($("day-chart"), {
    supply,
    surplus,
    generation: day.generation,
    unit: "kW",
    describe: (h) => {
      const hh = (n) => String(n % 24).padStart(2, "0");
      const row = (label, value, colour) =>
        value > 0.005
          ? `<br><span class="tip-swatch" style="background:${colour}"></span>${label} ${value.toFixed(2)} kW`
          : "";
      return (
        `<strong>${hh(h)}:00–${hh(h + 1)}:00</strong>` +
        `<br><span class="tip-muted">using ${demandAt(h).toFixed(2)} kW, ` +
        `generating ${day.generation[h].toFixed(2)} kW</span>` +
        row("Sunlight direct", day.direct[h], flow("solar")) +
        row("From battery", day.fromBattery[h], flow("battery")) +
        row("From grid", day.imported[h], flow("grid")) +
        row("Into battery", day.toBattery[h], flow("charge")) +
        row("Exported", day.exported[h], flow("export"))
      );
    },
  });

  legend($("day-legend"), [
    ...supply.map((s) => ({ label: s.label, color: s.color, value: `${total(s.values).toFixed(1)} kWh` })),
    ...surplus.map((s) => ({ label: s.label, color: s.color, value: `${total(s.values).toFixed(1)} kWh` })),
  ]);

  table(
    $("day-table"),
    ["Hour", "Generated kW", "Direct kW", "From battery kW", "From grid kW", "Into battery kW", "Exported kW"],
    Array.from({ length: 24 }, (_, h) => [
      `${String(h).padStart(2, "0")}:00`,
      day.generation[h].toFixed(2), day.direct[h].toFixed(2), day.fromBattery[h].toFixed(2),
      day.imported[h].toFixed(2), day.toBattery[h].toFixed(2), day.exported[h].toFixed(2),
    ])
  );
}

function renderCurve(data) {
  const curve = data.sizing_curve;
  const suggested = data.suggested_capacity_kwh;
  const chosen = data.battery.usable_capacity_kwh;
  const atSuggested = curve.find((row) => row.capacity_kwh === suggested);

  $("curve-sub").textContent = suggested > 0
    ? `Each extra kilowatt-hour of storage buys less than the one before it. Past ` +
      `about ${suggested} kWh the curve flattens — that capacity already reaches ` +
      `morning with charge to spare, so a bigger one would mostly sit idle` +
      (atSuggested ? `, at ${atSuggested.self_sufficiency_pct.toFixed(0)}% self-sufficiency.` : ".")
    : `Generation already lands when this household needs it, so storage has ` +
      `little to do. A battery is hard to justify here on self-consumption alone.`;

  curveChart($("curve-chart"), {
    x: curve.map((row) => row.capacity_kwh),
    y: curve.map((row) => row.avoided_import_kwh),
    markX: suggested,
    xLabel: "Usable battery capacity (kWh)",
    unit: "kWh",
    format: (v) => number(v),
    describe: (i) => {
      const row = curve[i];
      return (
        `<strong>${row.capacity_kwh} kWh battery</strong>` +
        `<br>${number(row.avoided_import_kwh)} kWh not bought` +
        `<br><span class="tip-muted">${row.self_sufficiency_pct.toFixed(0)}% self-sufficient` +
        (i > 0 ? ` · this step buys ${number(row.marginal_kwh_per_kwh)} kWh per extra kWh` : "") +
        `</span>`
      );
    },
  });

  table(
    $("curve-table"),
    ["Battery kWh", "Not bought kWh", "Self-sufficiency", "Self-consumption", "Exported kWh", "Cycles/yr", "Per extra kWh"],
    curve.map((row) => [
      row.capacity_kwh === chosen ? `${row.capacity_kwh} (yours)` : row.capacity_kwh,
      number(row.avoided_import_kwh),
      `${row.self_sufficiency_pct.toFixed(0)}%`,
      `${row.self_consumption_pct.toFixed(0)}%`,
      number(row.export_kwh),
      row.equivalent_full_cycles.toFixed(0),
      number(row.marginal_kwh_per_kwh),
    ])
  );
}

function renderSizingMonthly(data) {
  const months = data.monthly_balance;
  const best = months.reduce((a, b) => (b.self_sufficiency_pct > a.self_sufficiency_pct ? b : a));
  const worst = months.reduce((a, b) => (b.self_sufficiency_pct < a.self_sufficiency_pct ? b : a));

  $("sizing-monthly-sub").textContent =
    `${MONTH_NAMES[best.month - 1]} runs at ${best.self_sufficiency_pct.toFixed(0)}% off ` +
    `your own roof; ${MONTH_NAMES[worst.month - 1]} manages ` +
    `${worst.self_sufficiency_pct.toFixed(0)}%. Winter is when the grid earns its keep, ` +
    `and no domestic battery bridges that gap — the shortfall is seasonal, not daily.`;

  barChart($("sizing-monthly-chart"), {
    labels: MONTHS,
    values: months.map((m) => m.self_sufficiency_pct),
    unit: "%",
    format: (v) => `${v.toFixed(0)}`,
  });

  table(
    $("sizing-monthly-table"),
    ["Month", "Used kWh", "Generated kWh", "Direct kWh", "From battery kWh", "Bought kWh", "Exported kWh", "Self-sufficiency"],
    months.map((m) => [
      MONTH_NAMES[m.month - 1],
      number(m.load_kwh), number(m.generation_kwh), number(m.direct_kwh),
      number(m.from_battery_kwh), number(m.import_kwh), number(m.export_kwh),
      `${m.self_sufficiency_pct.toFixed(0)}%`,
    ])
  );
}

function renderConfidence(data) {
  $("confidence-note").textContent = data.confidence.note;

  const rows = [
    ["With your battery", data.confidence.band],
    ["Panels alone, no battery", data.confidence.band_pv_only],
  ].filter(([, band]) => band);

  $("confidence-bands").replaceChildren(
    ...rows.map(([label, band]) => {
      const row = document.createElement("div");
      row.className = "band-row";
      row.innerHTML =
        `<span class="band-label">${label}</span>` +
        `<span><strong>${number(band.low_kwh)}–${number(band.high_kwh)}</strong> kWh ` +
        `<span class="band-spread">(±${(band.spread_pct / 2).toFixed(0)}%)</span></span>`;
      return row;
    })
  );
}

/* ------------------------------------------------------------ downloads */

async function downloadCsv() {
  if (!state.lastRequest) return;
  const button = $("download-csv");
  button.disabled = true;
  button.textContent = "Preparing…";
  try {
    const response = await fetch("/api/estimate.csv", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...state.lastRequest, include_hourly: true, optimise: false }),
    });
    if (!response.ok) throw new Error("download failed");
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "solar_typical_year.csv";
    link.click();
    URL.revokeObjectURL(url);
  } catch (error) {
    $("form-error").textContent = String(error.message || error);
    $("form-error").hidden = false;
  } finally {
    button.disabled = false;
    button.textContent = "Download hourly CSV";
  }
}

/* ----------------------------------------------------------------- wire */

function init() {
  loadOptions();

  $("controls").addEventListener("submit", runEstimate);

  $("place").addEventListener("input", (event) => {
    clearTimeout(state.searchTimer);
    const value = event.target.value;
    state.searchTimer = setTimeout(() => searchPlaces(value), 250);
  });

  $("locate").addEventListener("click", () => {
    if (!navigator.geolocation) {
      $("form-error").textContent = "This browser cannot report your location.";
      $("form-error").hidden = false;
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (position) => {
        $("latitude").value = position.coords.latitude.toFixed(4);
        $("longitude").value = position.coords.longitude.toFixed(4);
        $("site-label").textContent = "Your current location";
        $("place").value = "";
        $("place-results").hidden = true;
        applyLatitudeDefaults(position.coords.latitude);
      },
      () => {
        $("form-error").textContent = "Could not read your location — enter coordinates instead.";
        $("form-error").hidden = false;
      }
    );
  });

  $("tilt").addEventListener("input", (e) => { $("tilt-value").textContent = `${e.target.value}°`; });
  $("azimuth").addEventListener("input", (e) => {
    $("azimuth-value").textContent = compassName(Number(e.target.value));
  });
  $("years").addEventListener("input", (e) => { $("years-value").textContent = e.target.value; });
  $("shading").addEventListener("input", (e) => { $("shading-value").textContent = `${e.target.value}%`; });

  for (const button of document.querySelectorAll("[data-toggle]")) {
    button.addEventListener("click", () => {
      const target = $(button.dataset.toggle);
      target.hidden = !target.hidden;
      button.textContent = target.hidden ? "Show table" : "Hide table";
    });
  }

  // ---- battery sizing controls
  buildMonthInputs();

  $("enable-sizing").addEventListener("change", (event) => {
    $("sizing-fields").hidden = !event.target.checked;
    $("run").textContent = event.target.checked
      ? "Estimate generation & savings"
      : "Estimate a year";
  });

  for (const radio of document.querySelectorAll('input[name="demand-mode"]')) {
    radio.addEventListener("change", applyDemandMode);
  }
  applyDemandMode();

  $("archetype").addEventListener("change", applyArchetypeDefault);
  $("daytime").addEventListener("input", (e) => {
    $("daytime-value").textContent = `${e.target.value}%`;
  });
  $("battery-kwh").addEventListener("input", (e) => {
    const value = Number(e.target.value);
    $("battery-value").textContent = value > 0 ? `${value.toFixed(1)} kWh` : "No battery";
  });
  $("meter-file").addEventListener("change", (e) => loadMeterFile(e.target.files[0]));

  $("download-csv").addEventListener("click", downloadCsv);

  document.addEventListener("click", (event) => {
    if (!event.target.closest(".field-group")) $("place-results").hidden = true;
  });

  applyLatitudeDefaults(Number($("latitude").value));
}

init();
