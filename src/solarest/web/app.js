/**
 * Page controller: reads the form, calls the API, renders the results.
 */

import { barChart, heatmap, stackedBar, legend, table } from "./charts.js";

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
const state = { lastRequest: null, lastResult: null, searchTimer: null };

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

async function runEstimate(event) {
  event.preventDefault();
  const request = buildRequest(false);
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
    const response = await fetch("/api/estimate", {
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

  $("download-csv").addEventListener("click", downloadCsv);

  document.addEventListener("click", (event) => {
    if (!event.target.closest(".field-group")) $("place-results").hidden = true;
  });

  applyLatitudeDefaults(Number($("latitude").value));
}

init();
