/**
 * Where the numbers come from.
 *
 * The page runs in two places and must not care which:
 *
 * - **Served by `solarest serve`**, where a FastAPI process does the work and
 *   this is a handful of `fetch` calls.
 * - **Static hosting**, GitHub Pages included, where there is no server at
 *   all. The same Python package is then loaded into the tab under Pyodide
 *   and called directly, and Open-Meteo is fetched from the browser.
 *
 * Which one is live is decided once, at start-up, by asking for `/api/health`.
 * A real server answers; static hosting returns its 404 page, and that is the
 * cue to boot the WebAssembly runtime instead.
 *
 * The two implementations return identical payloads — a test asserts that key
 * for key — so everything downstream of this module is written once.
 */

const PYODIDE_VERSION = "0.26.4";
const PYODIDE_CDN = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

/**
 * Where to load the Python runtime from.
 *
 * Defaults to the public CDN. A `<meta name="pyodide-base">` in the page
 * overrides it, which is how a self-hosted or air-gapped deployment points at
 * its own copy — and how this is tested without reaching the internet. It is
 * deliberately not a URL parameter: that would let any link decide which
 * script the page executes.
 */
const PYODIDE_BASE =
  document.querySelector('meta[name="pyodide-base"]')?.content || PYODIDE_CDN;

/** The Python modules that make up the browser build, in dependency order. */
const PYTHON_MODULES = [
  "__init__.py",
  "solarpos.py",
  "irradiance.py",
  "pvmodel.py",
  "simulate.py",
  "results.py",
  "optimize.py",
  "load.py",
  "battery.py",
  "sizing.py",
  "geocode.py",
  "weather.py",
  "presenter.py",
  "browser.py",
  "webapp.py",
];

/** Progress messages are surfaced to the user; booting is not instant. */
let onProgress = () => {};

/** Set a callback invoked with a human-readable status string. */
export function reportProgress(callback) {
  onProgress = callback || (() => {});
}

/* ------------------------------------------------------------ served mode */

async function postJson(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || `request failed (${response.status})`);
  return payload;
}

const serverBackend = {
  kind: "server",
  async options() {
    const response = await fetch("/api/options");
    if (!response.ok) throw new Error("could not load options");
    return response.json();
  },
  async geocode(query) {
    const response = await fetch(`/api/geocode?q=${encodeURIComponent(query)}`);
    if (!response.ok) throw new Error("search failed");
    return response.json();
  },
  estimate: (request) => postJson("/api/estimate", request),
  sizing: (request) => postJson("/api/sizing", request),
  async csv(request) {
    const response = await fetch("/api/estimate.csv", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (!response.ok) throw new Error("download failed");
    return response.blob();
  },
};

/* ---------------------------------------------------------- browser mode */

/** Load a script tag and resolve when it has run. */
function loadScript(src) {
  return new Promise((resolve, reject) => {
    const tag = document.createElement("script");
    tag.src = src;
    tag.onload = resolve;
    tag.onerror = () => reject(new Error(`could not load ${src}`));
    document.head.appendChild(tag);
  });
}

/**
 * Boot Pyodide, install numpy, and write the solarest package into its
 * virtual filesystem.
 *
 * The Python sources are fetched from this site rather than bundled into a
 * wheel: it keeps the build a plain file copy, and it means what runs in the
 * browser is byte-for-byte what is in the repository.
 */
async function bootPyodide() {
  onProgress("Loading the Python runtime…");
  await loadScript(`${PYODIDE_BASE}pyodide.js`);
  const pyodide = await globalThis.loadPyodide({ indexURL: PYODIDE_BASE });

  onProgress("Loading numpy…");
  await pyodide.loadPackage("numpy");

  onProgress("Loading the model…");
  const base = new URL("./python/solarest/", import.meta.url);
  const sources = await Promise.all(
    PYTHON_MODULES.map(async (name) => {
      const response = await fetch(new URL(name, base));
      if (!response.ok) throw new Error(`missing Python module ${name}`);
      return [name, await response.text()];
    })
  );

  pyodide.FS.mkdirTree("/lib/solarest");
  for (const [name, source] of sources) {
    pyodide.FS.writeFile(`/lib/solarest/${name}`, source);
  }
  await pyodide.runPythonAsync(`
import sys
sys.path.insert(0, "/lib")
import pyodide_js
`);
  return pyodide;
}

/**
 * Call a coroutine in the bridge with a JS object, and convert the result
 * back to plain JS.
 *
 * Pyodide hands back proxies that must be destroyed explicitly or they leak,
 * so conversion and cleanup are done in one place.
 */
async function callBridge(pyodide, functionName, args) {
  const webapp = pyodide.pyimport("solarest.webapp");
  const target = webapp[functionName];
  let argument;
  let result;
  try {
    argument = pyodide.toPy(args);
    result = await target(argument);
    return result.toJs({ dict_converter: Object.fromEntries });
  } catch (error) {
    // Python tracebacks are long and end with the useful line.
    const message = String(error.message || error).trim().split("\n").pop();
    throw new Error(message.replace(/^\w*(Error|Exception):\s*/, ""));
  } finally {
    argument?.destroy?.();
    result?.destroy?.();
    target?.destroy?.();
    webapp?.destroy?.();
  }
}

function browserBackend(pyodide) {
  return {
    kind: "browser",
    async options() {
      const webapp = pyodide.pyimport("solarest.webapp");
      try {
        const result = webapp.options();
        try {
          return result.toJs({ dict_converter: Object.fromEntries });
        } finally {
          result.destroy();
        }
      } finally {
        webapp.destroy();
      }
    },
    geocode: (query) => callBridge(pyodide, "geocode", query),
    estimate: (request) => callBridge(pyodide, "estimate", request),
    sizing: (request) => callBridge(pyodide, "sizing", request),
    async csv(request) {
      // No server to build the CSV, so assemble it here from the same series
      // the download endpoint would have used.
      const payload = await callBridge(pyodide, "estimate", {
        ...request,
        include_hourly: true,
        optimise: false,
      });
      return new Blob([typicalYearCsv(payload)], { type: "text/csv" });
    },
  };
}

/** Build the hourly CSV, matching the server's columns and comment block. */
function typicalYearCsv(payload) {
  const typical = payload.typical_year;
  const site = payload.site;
  const lines = [
    "# Solar Power Estimation - typical year hourly AC output",
    `# site: ${site.latitude.toFixed(4)}, ${site.longitude.toFixed(4)}`,
    `# timezone: ${site.timezone} (local standard time, no DST)`,
    `# system: ${payload.system.dc_capacity_kw} kWp, tilt ${payload.system.tilt_deg.toFixed(0)} deg,` +
      ` azimuth ${payload.system.azimuth_deg.toFixed(0)} deg`,
    `# source: ${payload.weather.source}`,
    "timestamp,ac_kw,energy_kwh,poa_w_m2,air_temp_c",
  ];

  const start = Date.UTC(typical.reference_year, 0, 1);
  for (let hour = 0; hour < typical.ac_kw.length; hour += 1) {
    const stamp = new Date(start + hour * 3600_000).toISOString().slice(0, 19);
    lines.push(
      `${stamp},${typical.ac_kw[hour].toFixed(4)},${typical.ac_kw[hour].toFixed(4)},` +
        `${typical.poa_w_m2[hour].toFixed(1)},${typical.air_temp_c[hour].toFixed(2)}`
    );
  }
  return lines.join("\n");
}

/* ------------------------------------------------------------- selection */

let ready = null;

/** Is a FastAPI server answering on this origin? */
async function serverIsPresent() {
  try {
    const response = await fetch("/api/health", { headers: { Accept: "application/json" } });
    if (!response.ok) return false;
    const payload = await response.json();
    return payload?.status === "ok";
  } catch {
    // Static hosting: no such route, or the response is not JSON.
    return false;
  }
}

/**
 * The backend for this page, chosen once and reused.
 *
 * @returns {Promise<object>} An object with options, geocode, estimate,
 *   sizing and csv methods, plus a `kind` of "server" or "browser".
 */
export function backend() {
  if (ready) return ready;
  ready = (async () => {
    if (await serverIsPresent()) return serverBackend;
    const pyodide = await bootPyodide();
    onProgress("");
    return browserBackend(pyodide);
  })();
  return ready;
}
