/**
 * Hand-rolled SVG charts.
 *
 * No chart library: the four forms this page needs are small, and drawing
 * them directly keeps the page dependency-free and offline-capable.
 *
 * Every chart re-renders at its real pixel size on resize, so labels stay
 * legible instead of scaling with a viewBox, and every chart ships a hover
 * layer plus an accessible table alternative.
 */

const SVG_NS = "http://www.w3.org/2000/svg";

/** Create an SVG element with attributes applied. */
function el(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== null && value !== undefined) node.setAttribute(key, String(value));
  }
  return node;
}

/** Read a CSS custom property off the chart surface. */
function token(node, name, fallback) {
  const value = getComputedStyle(node).getPropertyValue(name).trim();
  return value || fallback;
}

/** A single shared tooltip element, positioned against the viewport. */
const tooltip = (() => {
  let node = null;
  return {
    show(html, x, y) {
      if (!node) {
        node = document.createElement("div");
        node.className = "chart-tooltip";
        node.setAttribute("role", "status");
        document.body.appendChild(node);
      }
      node.innerHTML = html;
      node.style.display = "block";
      const box = node.getBoundingClientRect();
      const left = Math.min(Math.max(8, x - box.width / 2), window.innerWidth - box.width - 8);
      const top = y - box.height - 12 < 8 ? y + 18 : y - box.height - 12;
      node.style.left = `${left + window.scrollX}px`;
      node.style.top = `${top + window.scrollY}px`;
    },
    hide() {
      if (node) node.style.display = "none";
    },
  };
})();

/** Attach hover, focus and touch handlers that drive the shared tooltip. */
function makeInteractive(node, describe) {
  node.addEventListener("mousemove", (event) => {
    tooltip.show(describe(), event.clientX, event.clientY);
  });
  node.addEventListener("mouseleave", () => tooltip.hide());
  node.addEventListener("focus", () => {
    const box = node.getBoundingClientRect();
    tooltip.show(describe(), box.left + box.width / 2, box.top);
  });
  node.addEventListener("blur", () => tooltip.hide());
  node.setAttribute("tabindex", "0");
  node.setAttribute("role", "img");
}

/**
 * Re-render `draw(width, height)` whenever the container is resized.
 * Charts are drawn at true pixel size so text never scales with the viewBox.
 */
function responsive(container, height, draw) {
  const render = () => {
    const width = container.clientWidth;
    if (width < 40) return;
    container.replaceChildren(draw(width, height));
  };
  render();
  if (container._resizeObserver) container._resizeObserver.disconnect();
  const observer = new ResizeObserver(() => render());
  observer.observe(container);
  container._resizeObserver = observer;
}

/** Choose a rounded axis maximum and a matching tick step. */
function niceScale(max, targetTicks = 4) {
  if (!(max > 0)) return { max: 1, step: 0.25 };
  const raw = max / targetTicks;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const normalised = raw / magnitude;
  const step = (normalised <= 1 ? 1 : normalised <= 2 ? 2 : normalised <= 5 ? 5 : 10) * magnitude;
  return { max: Math.ceil(max / step) * step, step };
}

/** Rounded-top bar path: square at the baseline, 4px radius at the data end. */
function barPath(x, y, width, height, radius = 4) {
  const r = Math.max(0, Math.min(radius, width / 2, height));
  if (height <= 0) return "";
  return [
    `M${x},${y + height}`,
    `L${x},${y + r}`,
    `Q${x},${y} ${x + r},${y}`,
    `L${x + width - r},${y}`,
    `Q${x + width},${y} ${x + width},${y + r}`,
    `L${x + width},${y + height}`,
    "Z",
  ].join(" ");
}

/**
 * Vertical bar chart for one series, with an optional min-max range whisker
 * and an optional mean reference line.
 *
 * @param {HTMLElement} container Element to render into.
 * @param {object} options
 * @param {string[]} options.labels Category labels along the x axis.
 * @param {number[]} options.values Bar heights.
 * @param {Array<[number, number]>} [options.ranges] Per-bar [min, max].
 * @param {number} [options.mean] Draws a dashed reference line.
 * @param {string} options.unit Unit shown in tooltips and the axis title.
 * @param {(v:number)=>string} [options.format] Value formatter.
 * @param {string} [options.rangeLabel] What the whisker represents.
 */
export function barChart(container, options) {
  const { labels, values, ranges, mean, unit, format = (v) => v.toFixed(0), rangeLabel } = options;

  responsive(container, 300, (width, height) => {
    const svg = el("svg", { width, height, class: "chart", "aria-hidden": "true" });
    const ink = token(container, "--text-primary", "#111");
    const muted = token(container, "--text-muted", "#777");
    const grid = token(container, "--grid", "#e5e5e5");
    const series = token(container, "--series-1", "#2a78d6");
    const surface = token(container, "--surface-1", "#fff");

    const pad = { top: 16, right: 12, bottom: 34, left: 52 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const peak = Math.max(...values, ...(ranges ? ranges.map((r) => r[1]) : []));
    const scale = niceScale(peak);
    const y = (value) => pad.top + plotH - (value / scale.max) * plotH;

    // Gridlines and y-axis labels, kept recessive.
    for (let tick = 0; tick <= scale.max + 1e-9; tick += scale.step) {
      svg.appendChild(
        el("line", {
          x1: pad.left, x2: pad.left + plotW, y1: y(tick), y2: y(tick),
          stroke: grid, "stroke-width": 1,
        })
      );
      const text = el("text", { x: pad.left - 8, y: y(tick) + 4, "text-anchor": "end", class: "axis-label", fill: muted });
      text.textContent = format(tick);
      svg.appendChild(text);
    }

    // 2px surface gap between adjacent bars.
    const slot = plotW / labels.length;
    const barW = Math.max(3, slot - 2 - Math.min(10, slot * 0.25));

    labels.forEach((label, index) => {
      const x = pad.left + slot * index + (slot - barW) / 2;
      const value = values[index];
      const top = y(value);

      const bar = el("path", { d: barPath(x, top, barW, pad.top + plotH - top), fill: series });
      const range = ranges ? ranges[index] : null;
      makeInteractive(bar, () =>
        `<strong>${label}</strong><br>${format(value)} ${unit}` +
        (range ? `<br><span class="tip-muted">${rangeLabel || "range"}: ${format(range[0])}–${format(range[1])} ${unit}</span>` : "")
      );
      bar.setAttribute("aria-label", `${label}: ${format(value)} ${unit}`);
      svg.appendChild(bar);

      if (range) {
        const cx = x + barW / 2;
        const cap = Math.min(5, barW / 3);
        const whisker = el("g", { class: "whisker", stroke: ink, "stroke-width": 1.5, "stroke-opacity": 0.55, fill: "none" });
        whisker.appendChild(el("line", { x1: cx, x2: cx, y1: y(range[0]), y2: y(range[1]) }));
        whisker.appendChild(el("line", { x1: cx - cap, x2: cx + cap, y1: y(range[1]), y2: y(range[1]) }));
        whisker.appendChild(el("line", { x1: cx - cap, x2: cx + cap, y1: y(range[0]), y2: y(range[0]) }));
        svg.appendChild(whisker);
      }

      const tick = el("text", { x: x + barW / 2, y: height - 12, "text-anchor": "middle", class: "axis-label", fill: muted });
      tick.textContent = label;
      svg.appendChild(tick);
    });

    if (mean !== undefined && mean > 0) {
      svg.appendChild(
        el("line", {
          x1: pad.left, x2: pad.left + plotW, y1: y(mean), y2: y(mean),
          stroke: ink, "stroke-width": 1.5, "stroke-dasharray": "5 4", "stroke-opacity": 0.6,
        })
      );
      const badge = el("text", { x: pad.left + plotW, y: y(mean) - 6, "text-anchor": "end", class: "axis-label", fill: ink, stroke: surface, "stroke-width": 3, "paint-order": "stroke" });
      badge.textContent = `mean ${format(mean)}`;
      svg.appendChild(badge);
    }

    return svg;
  });
}

/**
 * Sequential heatmap on a single-hue ramp.
 *
 * @param {HTMLElement} container Element to render into.
 * @param {object} options
 * @param {number[][]} options.grid Rows of values.
 * @param {string[]} options.rowLabels Label per row.
 * @param {string[]} options.colLabels Label per column.
 * @param {string} options.unit Unit shown in tooltips.
 * @param {(r:number,c:number)=>string} options.describe Tooltip text builder.
 * @param {(v:number)=>string} [options.format] Value formatter.
 * @param {"zero"|"range"} [options.baseline] Where the ramp starts. ``"zero"``
 *   when zero is meaningful (an hour with no sun); ``"range"`` when every cell
 *   is large and the interesting variation is between them.
 * @param {boolean} [options.sqrtScale] Compress the ramp so values near zero
 *   stay visible. Defaults to true for a zero baseline.
 */
export function heatmap(container, options) {
  const {
    grid, rowLabels, colLabels, unit, describe,
    format = (v) => v.toFixed(2),
    baseline = "zero",
    sqrtScale = baseline === "zero",
  } = options;

  responsive(container, 34 + grid.length * 22 + 26, (width, height) => {
    const svg = el("svg", { width, height, class: "chart", "aria-hidden": "true" });
    const muted = token(container, "--text-muted", "#777");
    const surface = token(container, "--surface-1", "#fff");
    // Sequential blue ramp, light -> dark. Index 0 means "near zero".
    const ramp = (token(container, "--ramp", "") || "#cde2fb,#9ec5f4,#6da7ec,#3987e5,#2a78d6,#256abf,#184f95,#0d366b").split(",");

    const pad = { top: 8, right: 8, bottom: 26, left: 44 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const cellW = plotW / colLabels.length;
    const cellH = plotH / rowLabels.length;

    const flat = grid.flat();
    const peak = Math.max(...flat);
    const floor = baseline === "range" ? Math.min(...flat) : 0;
    const span = peak - floor;
    const colour = (value) => {
      if (!(span > 0) || (baseline === "zero" && value <= 0)) return "transparent";
      // Square-root spacing where zero is meaningful: PV output spends many
      // hours near zero, and a linear ramp would wash out the shoulders of
      // the day entirely.
      const linear = (value - floor) / span;
      const fraction = sqrtScale ? Math.sqrt(linear) : linear;
      return ramp[Math.min(ramp.length - 1, Math.floor(fraction * ramp.length))];
    };

    grid.forEach((row, r) => {
      row.forEach((value, c) => {
        const cell = el("rect", {
          x: pad.left + c * cellW + 1, y: pad.top + r * cellH + 1,
          width: Math.max(1, cellW - 2), height: Math.max(1, cellH - 2),
          rx: 2, fill: colour(value), stroke: surface, "stroke-width": 0.5,
        });
        makeInteractive(cell, () => describe(r, c));
        cell.setAttribute("aria-label", `${rowLabels[r]} ${colLabels[c]}: ${format(value)} ${unit}`);
        svg.appendChild(cell);
      });

      const label = el("text", { x: pad.left - 8, y: pad.top + r * cellH + cellH / 2 + 4, "text-anchor": "end", class: "axis-label", fill: muted });
      label.textContent = rowLabels[r];
      svg.appendChild(label);
    });

    colLabels.forEach((label, c) => {
      if (label === "") return;
      const text = el("text", { x: pad.left + c * cellW + cellW / 2, y: height - 8, "text-anchor": "middle", class: "axis-label", fill: muted });
      text.textContent = label;
      svg.appendChild(text);
    });

    return svg;
  });
}

/**
 * Horizontal stacked bar showing how one total divides into parts.
 *
 * Segments carry direct labels wherever they fit, which is also the relief
 * required for the lighter categorical slots on a light surface.
 *
 * @param {HTMLElement} container Element to render into.
 * @param {Array<{label:string, value:number, color:string}>} segments Parts.
 * @param {string} unit Unit shown in tooltips.
 */
export function stackedBar(container, segments, unit) {
  responsive(container, 74, (width, height) => {
    const svg = el("svg", { width, height, class: "chart", "aria-hidden": "true" });
    const ink = token(container, "--text-primary", "#111");
    const surface = token(container, "--surface-1", "#fff");

    const total = segments.reduce((sum, s) => sum + Math.max(0, s.value), 0) || 1;
    const barTop = 8;
    const barH = 30;
    let x = 0;

    segments.forEach((segment) => {
      const raw = (Math.max(0, segment.value) / total) * width;
      // 2px surface gap between neighbouring fills.
      const w = Math.max(0, raw - 2);
      if (w <= 0) {
        x += raw;
        return;
      }
      const rect = el("rect", { x, y: barTop, width: w, height: barH, rx: 3, fill: segment.color });
      makeInteractive(rect, () => `<strong>${segment.label}</strong><br>${segment.value.toFixed(1)}${unit}`);
      rect.setAttribute("aria-label", `${segment.label}: ${segment.value.toFixed(1)}${unit}`);
      svg.appendChild(rect);

      const text = `${segment.value.toFixed(1)}%`;
      if (w > 30) {
        const label = el("text", {
          x: x + w / 2, y: barTop + barH / 2 + 4, "text-anchor": "middle",
          class: "segment-label", fill: surface,
        });
        label.textContent = text;
        svg.appendChild(label);
      } else if (w > 6) {
        const label = el("text", {
          x: x + w / 2, y: barTop + barH + 14, "text-anchor": "middle",
          class: "axis-label", fill: ink,
        });
        label.textContent = text;
        svg.appendChild(label);
      }
      x += raw;
    });

    return svg;
  });
}

/**
 * Build a legend, and a toggleable data table that carries the same numbers.
 *
 * @param {HTMLElement} container Element to render into.
 * @param {Array<{label:string, color:string, value:string}>} entries Legend rows.
 */
export function legend(container, entries) {
  container.replaceChildren(
    ...entries.map((entry) => {
      const item = document.createElement("span");
      item.className = "legend-item";
      const swatch = document.createElement("span");
      swatch.className = "legend-swatch";
      swatch.style.background = entry.color;
      const label = document.createElement("span");
      label.textContent = entry.label;
      const value = document.createElement("strong");
      value.textContent = entry.value;
      item.append(swatch, label, value);
      return item;
    })
  );
}

/**
 * Render a plain data table — the accessible equivalent of a chart.
 *
 * @param {HTMLElement} container Element to render into.
 * @param {string[]} columns Header cells.
 * @param {Array<Array<string|number>>} rows Body cells.
 */
export function table(container, columns, rows) {
  const el_ = (tag, text) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const tableEl = document.createElement("table");
  const head = el_("thead");
  const headRow = el_("tr");
  columns.forEach((c) => headRow.appendChild(el_("th", c)));
  head.appendChild(headRow);
  const body = el_("tbody");
  rows.forEach((row) => {
    const tr = el_("tr");
    row.forEach((cell) => tr.appendChild(el_("td", String(cell))));
    body.appendChild(tr);
  });
  tableEl.append(head, body);
  container.replaceChildren(tableEl);
}

/**
 * The energy balance across a typical day, as a diverging stacked area.
 *
 * Above the axis is where the household's demand came from — sunlight used as
 * it arrives, then the battery, then whatever had to be bought. Below the axis
 * is where generation went when it exceeded demand: into the battery, or out
 * to the grid. A dashed line traces total generation.
 *
 * Both directions share one kW-per-pixel scale, so a kilowatt bought looks
 * exactly as big as a kilowatt exported. Areas are stepped rather than
 * smoothed because the underlying data is an hourly mean — a curve through the
 * midpoints would imply detail the model does not have.
 *
 * @param {HTMLElement} container Element to render into.
 * @param {object} options
 * @param {Array<{label:string,color:string,values:number[]}>} options.supply
 *   Stacked upward, in order: what met demand.
 * @param {Array<{label:string,color:string,values:number[]}>} options.surplus
 *   Stacked downward: where the extra generation went.
 * @param {number[]} options.generation Total generation, drawn as a line.
 * @param {string} [options.unit] Unit for labels and tooltips.
 * @param {(hour:number)=>string} options.describe Tooltip text builder.
 */
export function dayProfile(container, options) {
  const {
    supply, surplus, generation, unit = "kW",
    format = (v) => v.toFixed(2), describe,
  } = options;

  responsive(container, 380, (width, height) => {
    const svg = el("svg", { width, height, class: "chart", "aria-hidden": "true" });
    const ink = token(container, "--text-primary", "#111");
    const muted = token(container, "--text-muted", "#777");
    const gridLine = token(container, "--grid", "#e5e5e5");
    const surface = token(container, "--surface-1", "#fff");

    const pad = { top: 14, right: 12, bottom: 30, left: 54 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const x = (hour) => pad.left + (hour / 24) * plotW;

    const totalAt = (stack, h) =>
      stack.reduce((sum, s) => sum + Math.max(0, s.values[h] || 0), 0);
    const hours = Array.from({ length: 24 }, (_, h) => h);
    const aboveMax = Math.max(...hours.map((h) => totalAt(supply, h)), ...generation, 0);
    const belowMax = Math.max(...hours.map((h) => totalAt(surplus, h)), 0);

    // One step for both halves, so the two directions stay comparable.
    const { step } = niceScale(Math.max(aboveMax, belowMax, 0.001), 3);
    const aboveTop = Math.max(step, Math.ceil(aboveMax / step) * step);
    const belowTop = Math.ceil(belowMax / step) * step;
    const span = aboveTop + belowTop;
    const y = (value) => pad.top + ((aboveTop - value) / span) * plotH;

    for (let tick = -belowTop; tick <= aboveTop + 1e-9; tick += step) {
      const zero = Math.abs(tick) < 1e-9;
      svg.appendChild(el("line", {
        x1: pad.left, x2: pad.left + plotW, y1: y(tick), y2: y(tick),
        stroke: zero ? muted : gridLine, "stroke-width": zero ? 1.25 : 1,
      }));
      const label = el("text", {
        x: pad.left - 8, y: y(tick) + 4, "text-anchor": "end",
        class: "axis-label", fill: muted,
      });
      label.textContent = format(Math.abs(tick));
      svg.appendChild(label);
    }

    /** Stepped band between two per-hour boundaries, as a closed path. */
    const band = (lo, hi) => {
      const parts = [`M${x(0)},${y(hi[0])}`];
      for (let h = 0; h < 24; h += 1) {
        parts.push(`L${x(h)},${y(hi[h])}`, `L${x(h + 1)},${y(hi[h])}`);
      }
      for (let h = 23; h >= 0; h -= 1) {
        parts.push(`L${x(h + 1)},${y(lo[h])}`, `L${x(h)},${y(lo[h])}`);
      }
      return `${parts.join(" ")} Z`;
    };

    const drawStack = (stack, sign, parent) => {
      const cursor = new Array(24).fill(0);
      for (const series of stack) {
        const lo = cursor.slice();
        const hi = cursor.map((base, h) => base + Math.max(0, series.values[h] || 0));
        parent.appendChild(el("path", {
          d: band(lo.map((v) => v * sign), hi.map((v) => v * sign)),
          fill: series.color, "fill-opacity": 0.92,
        }));
        hi.forEach((v, h) => { cursor[h] = v; });
      }
    };
    const fills = el("g");
    svg.appendChild(fills);
    drawStack(supply, 1, fills);
    drawStack(surplus, -1, fills);

    // Generation on top of the fills, so it reads as a reference not a layer.
    const line = [];
    for (let h = 0; h < 24; h += 1) {
      line.push(`${h === 0 ? "M" : "L"}${x(h)},${y(generation[h])}`, `L${x(h + 1)},${y(generation[h])}`);
    }
    svg.appendChild(el("path", {
      d: line.join(" "), fill: "none", stroke: ink, "stroke-width": 1.75,
      "stroke-dasharray": "5 3", "stroke-opacity": 0.75, "stroke-linejoin": "round",
    }));

    for (let h = 0; h < 24; h += 3) {
      const label = el("text", {
        x: x(h), y: height - 10, "text-anchor": "middle", class: "axis-label", fill: muted,
      });
      label.textContent = `${String(h).padStart(2, "0")}`;
      svg.appendChild(label);
    }

    // Full-height hover targets: one per hour, so the whole column responds.
    hours.forEach((h) => {
      const hit = el("rect", {
        x: x(h), y: pad.top, width: plotW / 24, height: plotH,
        fill: "transparent", class: "hour-hit",
      });
      makeInteractive(hit, () => describe(h));
      hit.setAttribute("aria-label", describe(h).replace(/<[^>]+>/g, " "));
      svg.appendChild(hit);
    });

    const caption = el("text", {
      x: pad.left, y: pad.top - 2, class: "axis-label", fill: muted,
      stroke: surface, "stroke-width": 3, "paint-order": "stroke",
    });
    caption.textContent = `${unit} — above: demand met · below: surplus`;
    svg.appendChild(caption);

    return svg;
  });
}

/**
 * A continuous relationship as a stepped-free line with points, plus an
 * optional marked value.
 *
 * Used for the battery sizing curve, where capacity is genuinely continuous —
 * bars would imply a set of discrete options rather than a curve with a knee.
 *
 * @param {HTMLElement} container Element to render into.
 * @param {object} options
 * @param {number[]} options.x Domain values.
 * @param {number[]} options.y Range values.
 * @param {number} [options.markX] Domain value to highlight.
 * @param {string} options.xLabel Axis caption.
 * @param {string} options.unit Unit shown in tooltips.
 * @param {(i:number)=>string} options.describe Tooltip text builder.
 */
export function curveChart(container, options) {
  const {
    x: xs, y: ys, markX, xLabel, unit,
    format = (v) => v.toFixed(0), describe,
  } = options;

  responsive(container, 260, (width, height) => {
    const svg = el("svg", { width, height, class: "chart", "aria-hidden": "true" });
    const ink = token(container, "--text-primary", "#111");
    const muted = token(container, "--text-muted", "#777");
    const gridLine = token(container, "--grid", "#e5e5e5");
    const accent = token(container, "--accent", "#2a78d6");
    const surface = token(container, "--surface-1", "#fff");

    const pad = { top: 16, right: 16, bottom: 38, left: 58 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const xMax = Math.max(...xs) || 1;
    const scale = niceScale(Math.max(...ys));
    const px = (v) => pad.left + (v / xMax) * plotW;
    const py = (v) => pad.top + plotH - (v / scale.max) * plotH;

    for (let tick = 0; tick <= scale.max + 1e-9; tick += scale.step) {
      svg.appendChild(el("line", {
        x1: pad.left, x2: pad.left + plotW, y1: py(tick), y2: py(tick),
        stroke: gridLine, "stroke-width": 1,
      }));
      const label = el("text", {
        x: pad.left - 8, y: py(tick) + 4, "text-anchor": "end",
        class: "axis-label", fill: muted,
      });
      label.textContent = format(tick);
      svg.appendChild(label);
    }

    svg.appendChild(el("path", {
      d: xs.map((v, i) => `${i === 0 ? "M" : "L"}${px(v)},${py(ys[i])}`).join(" "),
      fill: "none", stroke: accent, "stroke-width": 2.25,
      "stroke-linejoin": "round", "stroke-linecap": "round",
    }));

    if (markX !== undefined && markX > 0) {
      svg.appendChild(el("line", {
        x1: px(markX), x2: px(markX), y1: pad.top, y2: pad.top + plotH,
        stroke: ink, "stroke-width": 1.5, "stroke-dasharray": "4 3", "stroke-opacity": 0.55,
      }));
      const badge = el("text", {
        x: px(markX), y: pad.top + 10, "text-anchor": "middle", class: "axis-label",
        fill: ink, stroke: surface, "stroke-width": 3, "paint-order": "stroke",
      });
      badge.textContent = "suggested";
      svg.appendChild(badge);
    }

    xs.forEach((v, i) => {
      const dot = el("circle", {
        cx: px(v), cy: py(ys[i]), r: 4.5,
        fill: v === markX ? ink : accent, stroke: surface, "stroke-width": 1.5,
      });
      makeInteractive(dot, () => describe(i));
      dot.setAttribute("aria-label", describe(i).replace(/<[^>]+>/g, " "));
      svg.appendChild(dot);

      if (i === 0 || i === xs.length - 1 || xs.length <= 8 || i % 2 === 0) {
        const label = el("text", {
          x: px(v), y: height - 20, "text-anchor": "middle", class: "axis-label", fill: muted,
        });
        label.textContent = String(v);
        svg.appendChild(label);
      }
    });

    const caption = el("text", {
      x: pad.left + plotW / 2, y: height - 4, "text-anchor": "middle",
      class: "axis-label", fill: muted,
    });
    caption.textContent = xLabel;
    svg.appendChild(caption);

    return svg;
  });
}
