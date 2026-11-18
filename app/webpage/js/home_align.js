// Copyright (c) 2026 Yu Chen. All rights reserved.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

// SpatialArena home tab: the two alignment demos, each on a slide of its own. Both run the benchmark's own
// procedures, step by step: both on recorded runs of the
// evaluator's code.
//
// Depth: a real frame (iBims-1, laser-scanned GT; windows and glass have none) and MoGe's raw prediction, which is
// depth up to an unknown scale and shift. The steps are the benchmark's own full-resolution fit of that frame
// (assets/align, exported from the stored artifacts): least squares on every GT pixel, then Huber IRLS (delta 0.05 of
// the median GT depth) until the fit stops moving, as spatialarena/eval/huber_irls.py. Shown: the image, the GT,
// the raw prediction, the error of the current fit, the fit on a prediction-vs-GT scatter, and the Huber loss and
// AbsRel over the iterations.
//
// Trajectory: six scenarios of the alignment study (scripts/test/compare_trajectory_alignment.py): noise, outliers,
// a tracking jump, rotation noise, drift and a near-collinear path. Each case is one trial of the study on a real GT
// trajectory (TUM RGB-D, SpatialArena) or its synthetic straight path, with every step of the SLAM evaluator's
// estimator (pose_alignment.ransac_irls_align_sim3) recorded offline: the two-frame LMedS RANSAC samples, the inlier
// refit and each Cauchy IRLS iteration. Shown: the frustums in 3D, the ATE ratio over the fit, and this case against
// Umeyama on all centers (evo) and the previous default (R-DAGSfM), beside the study's medians.

(function () {
    "use strict";

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const state = { paused: false };

    // ---------------------------------------------------------------- shared helpers

    function mulberry32(seed) {
        let s = seed >>> 0;
        return () => {
            s = (s + 0x6d2b79f5) >>> 0;
            let t = s;
            t = Math.imul(t ^ (t >>> 15), t | 1);
            t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }
    const gaussian = rng => Math.sqrt(-2 * Math.log(Math.max(rng(), 1e-12))) * Math.cos(2 * Math.PI * rng());
    const ease = t => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
    const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
    const alpha = (hex, a) => {
        const h = hex.replace("#", "");
        if (h.length !== 6) return hex;
        return `rgba(${parseInt(h.slice(0, 2), 16)}, ${parseInt(h.slice(2, 4), 16)}, ${parseInt(h.slice(4, 6), 16)}, ${a})`;
    };
    const colors = () => ({ gt: css("--home-gt"), pred: css("--home-pred"), l2: css("--home-l2"), out: css("--home-out"),
        grid: css("--home-grid"), muted: css("--home-muted"), ink: css("--home-ink") });

    function prepare(canvas) {
        const ratio = window.devicePixelRatio || 1;
        const width = canvas.clientWidth, height = canvas.clientHeight;
        if (!width || !height) return null;
        if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
            canvas.width = Math.round(width * ratio);
            canvas.height = Math.round(height * ratio);
        }
        const ctx = canvas.getContext("2d");
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        ctx.clearRect(0, 0, width, height);
        return { ctx, width, height };
    }

    function line(ctx, x1, y1, x2, y2, color, width, dash) {
        ctx.save();
        ctx.beginPath();
        ctx.setLineDash(dash || []);
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.lineCap = "round";
        ctx.stroke();
        ctx.restore();
    }

    function polyline(ctx, points, color, width, dash) {
        if (points.length < 2) return;
        ctx.save();
        ctx.beginPath();
        ctx.setLineDash(dash || []);
        points.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.stroke();
        ctx.restore();
    }

    // Turbo (A. Mikhailov's polynomial fit), nearer warmer as the dataset heatmaps; and a dark-to-bright error ramp.
    function turbo(x) {
        x = Math.min(Math.max(x, 0), 1);
        const r = 0.13572138 + x * (4.6153926 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))));
        const g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))));
        const b = 0.1066733 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))));
        return [r, g, b].map(v => Math.round(255 * Math.min(Math.max(v, 0), 1)));
    }
    const ERROR_STOPS = [[0, [14, 16, 40]], [0.3, [84, 32, 128]], [0.6, [214, 64, 104]], [0.85, [250, 152, 56]], [1, [252, 240, 170]]];
    function errorRamp(x) {
        x = Math.min(Math.max(x, 0), 1);
        for (let i = 1; i < ERROR_STOPS.length; i += 1) {
            const [x1, c1] = ERROR_STOPS[i];
            const [x0, c0] = ERROR_STOPS[i - 1];
            if (x <= x1) {
                const k = (x - x0) / (x1 - x0);
                return c0.map((v, j) => Math.round(v + (c1[j] - v) * k));
            }
        }
        return ERROR_STOPS[ERROR_STOPS.length - 1][1];
    }

    function timeline(count, hold, tween, finalHold) {
        const segments = [];
        let start = 0;
        for (let i = 0; i < count - 1; i += 1) { segments.push({ i, start }); start += hold + tween; }
        return { segments, hold, tween, total: start + finalHold, last: count - 1 };
    }
    function position(tl, elapsed) {
        for (const segment of tl.segments) {
            if (elapsed < segment.start + tl.hold) return { index: segment.i, t: 0 };
            if (elapsed < segment.start + tl.hold + tl.tween) return { index: segment.i, t: (elapsed - segment.start - tl.hold) / tl.tween };
        }
        return { index: tl.last, t: 0 };
    }

    function slideVisible(element) {
        const panel = document.getElementById("benchmark-view-home");
        if (!panel || panel.hidden || document.hidden || !element) return false;
        const box = element.getBoundingClientRect();
        return box.bottom > 0 && box.top < window.innerHeight;
    }

    function setSteps(list, key, order) {
        if (!list) return;
        const active = order.indexOf(key);
        list.querySelectorAll("li").forEach(item => {
            const index = order.indexOf(item.dataset.step);
            item.classList.toggle("is-active", index === active);
            item.classList.toggle("is-done", index >= 0 && index < active);
        });
    }

    // ---------------------------------------------------------------- depth: real frames

    // assets/align/manifest.json lists the frames. Each <name>.bin holds the GT (uint16, 0 = no GT), then the raw
    // prediction (uint16 over [predMin, predMax]), both at half resolution. Each frame carries the IRLS trace of the
    // benchmark's full-resolution fit: scale, shift, Huber loss, AbsRel and delta < 1.25 per step.
    const ALIGN_ROOT = "assets/align/";
    const HUBER_DELTA = 0.05;
    let manifestPromise = null;
    const frameCache = new Map();

    function loadManifest() {
        if (!manifestPromise) {
            manifestPromise = fetch(`${ALIGN_ROOT}manifest.json`).then(r => {
                if (!r.ok) throw new Error(`manifest: HTTP ${r.status}`);
                return r.json();
            });
        }
        return manifestPromise;
    }

    function percentile(sorted, q) {
        return sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(sorted.length * q)))];
    }

    function loadFrame(meta) {
        if (frameCache.has(meta.name)) return frameCache.get(meta.name);
        const image = new Image();
        image.decoding = "async";
        const imageReady = new Promise((resolve, reject) => {
            image.onload = () => resolve(image);
            image.onerror = () => reject(new Error(`${meta.name}.jpg failed to load`));
        });
        image.src = `${ALIGN_ROOT}${meta.name}.jpg`;
        const promise = Promise.all([
            fetch(`${ALIGN_ROOT}${meta.name}.bin`).then(r => {
                if (!r.ok) throw new Error(`${meta.name}.bin: HTTP ${r.status}`);
                return r.arrayBuffer();
            }),
            imageReady,
        ]).then(([buffer]) => makeDepthScene(meta, buffer, image));
        frameCache.set(meta.name, promise);
        return promise;
    }

    function makeDepthScene(meta, buffer, image) {
        const n = meta.width * meta.height;
        const words = new DataView(buffer);
        const gt = new Float32Array(n), pred = new Float32Array(n), validMask = new Uint8Array(n);
        const valid = [];
        for (let i = 0; i < n; i += 1) {
            const g = words.getUint16(2 * i, true), p = words.getUint16(2 * (n + i), true);
            pred[i] = meta.predMin + p / 65535 * (meta.predMax - meta.predMin);
            if (g > 0) { gt[i] = (g - 1) / 65534 * meta.gtMax; validMask[i] = 1; valid.push(i); }
        }
        const gtSorted = valid.map(i => gt[i]).sort((a, b) => a - b);
        const predSorted = Array.from(pred).sort((a, b) => a - b);
        const scene = {
            meta, image, gt, pred, validMask, valid, width: meta.width, height: meta.height, unit: meta.unit,
            range: [percentile(gtSorted, 0.02), percentile(gtSorted, 0.98)],
            predRange: [percentile(predSorted, 0.02), percentile(predSorted, 0.98)],
            iterations: meta.steps.map(s => ({ ...s, iteration: s.step, a: s.scale, b: s.shift })),
        };
        // the maps step through every IRLS iteration when there are few, a spread of them otherwise
        const last = scene.iterations.length - 1;
        const shown = new Set([0, 1, 2, 3, 4, 5, 6, 8, 11, 15, 20, 30, 45, last].filter(i => i <= last));
        scene.states = [{ key: "raw", a: 1, b: 0, iteration: -1 }]
            .concat(scene.iterations.filter(s => shown.has(s.iteration))
                .map(s => ({ key: s.iteration === 0 ? "l2" : "irls", ...s })));
        const sampleRng = mulberry32(17);
        scene.sample = Array.from({ length: 2400 }, () => valid[Math.floor(sampleRng() * valid.length)]);
        scene.gtImage = depthImage(scene, gt, validMask, scene.range);
        scene.predImage = depthImage(scene, pred, null, scene.predRange);
        return scene;
    }

    function hatch(data, o, x, y) {
        const v = ((x + y) % 6) < 3 ? 120 : 92;
        data[o] = v; data[o + 1] = v; data[o + 2] = v + 6; data[o + 3] = 255;
    }

    function depthImage(scene, values, mask, range) {
        const { width } = scene;
        const image = new ImageData(width, scene.height);
        const [lo, hi] = range;
        for (let i = 0; i < values.length; i += 1) {
            const o = 4 * i;
            if (mask && !mask[i]) { hatch(image.data, o, i % width, (i / width) | 0); continue; }
            const [r, g, b] = turbo(Math.min(1, Math.max(0, 1 - (values[i] - lo) / (hi - lo))));
            image.data[o] = r; image.data[o + 1] = g; image.data[o + 2] = b; image.data[o + 3] = 255;
        }
        return image;
    }

    function errorImage(scene, a, b) {
        const { width } = scene;
        const error = new ImageData(width, scene.height);
        for (let i = 0; i < scene.gt.length; i += 1) {
            const o = 4 * i;
            if (!scene.validMask[i]) { hatch(error.data, o, i % width, (i / width) | 0); continue; }
            const value = a * scene.pred[i] + b;
            const [r, g, bb] = errorRamp(Math.abs(value - scene.gt[i]) / scene.gt[i] / 0.25);
            error.data[o] = r; error.data[o + 1] = g; error.data[o + 2] = bb; error.data[o + 3] = 255;
        }
        return error;
    }

    const scratch = document.createElement("canvas");
    function drawImage(canvas, source) {
        const target = prepare(canvas);
        if (!target) return;
        let drawable = source;
        if (source instanceof ImageData) {
            if (scratch.width !== source.width || scratch.height !== source.height) {
                scratch.width = source.width; scratch.height = source.height;
            }
            scratch.getContext("2d").putImageData(source, 0, 0);
            drawable = scratch;
        }
        const { ctx, width, height } = target;
        const sw = drawable.naturalWidth || drawable.width, sh = drawable.naturalHeight || drawable.height;
        const scale = Math.min(width / sw, height / sh);
        const w = sw * scale, h = sh * scale;
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(drawable, (width - w) / 2, (height - h) / 2, w, h);
    }

    // ---------------------------------------------------------------- plots
    //
    // One small chart frame for every curve on the two slides: nice ticks on both axes, a recessive grid, axis titles
    // outside the data, 2 px lines, and a hover read-out (crosshair and tooltip) on the series plots. Text is drawn in
    // the ink colors; the series color only marks the data.

    const FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    const surface = () => css("--home-surface");

    function niceStep(span, count) {
        const raw = span / Math.max(count, 1);
        const magnitude = 10 ** Math.floor(Math.log10(raw || 1));
        return [1, 2, 2.5, 5, 10].map(m => m * magnitude).find(step => step >= raw * 0.999) || 10 * magnitude;
    }

    // Ticks covering [lo, hi], widened to whole steps.
    function niceScale(lo, hi, count = 4) {
        if (!(hi > lo)) { const pad = Math.abs(hi) * 0.1 || 1; lo -= pad; hi += pad; }
        const step = niceStep(hi - lo, count);
        const first = Math.floor(lo / step + 1e-9) * step, last = Math.ceil(hi / step - 1e-9) * step;
        const ticks = [];
        for (let v = first; v <= last + step * 1e-6; v += step) ticks.push(Number(v.toPrecision(12)));
        return { lo: first, hi: last, ticks, step };
    }

    function trackHover(canvas) {
        if (!canvas.alignHover) {
            const hover = { x: null, y: null };
            canvas.addEventListener("pointermove", event => {
                const box = canvas.getBoundingClientRect();
                hover.x = event.clientX - box.left;
                hover.y = event.clientY - box.top;
            });
            canvas.addEventListener("pointerleave", () => { hover.x = null; });
            canvas.alignHover = hover;
        }
        return canvas.alignHover;
    }

    function roundRect(ctx, x, y, w, h, r) {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.arcTo(x + w, y, x + w, y + h, r);
        ctx.arcTo(x + w, y + h, x, y + h, r);
        ctx.arcTo(x, y + h, x, y, r);
        ctx.arcTo(x, y, x + w, y, r);
        ctx.closePath();
    }

    // A tooltip beside (x, y): the first line bold, the rest muted; flipped to stay inside the canvas.
    function tooltip(ctx, x, y, lines, width, height, c) {
        ctx.save();
        ctx.font = `600 11.5px ${FONT}`;
        const w = Math.max(...lines.map((text, i) => {
            ctx.font = `${i ? 400 : 600} 11.5px ${FONT}`;
            return ctx.measureText(text).width;
        })) + 18;
        const h = lines.length * 16 + 10;
        let left = x + 12, top = y - h - 8;
        if (left + w > width - 4) left = x - w - 12;
        if (top < 4) top = Math.min(y + 12, height - h - 4);
        ctx.shadowColor = "rgba(15, 20, 30, 0.18)";
        ctx.shadowBlur = 10;
        ctx.shadowOffsetY = 2;
        roundRect(ctx, left, top, w, h, 7);
        ctx.fillStyle = surface();
        ctx.fill();
        ctx.shadowColor = "transparent";
        ctx.strokeStyle = c.grid;
        ctx.lineWidth = 1;
        ctx.stroke();
        lines.forEach((text, i) => {
            ctx.font = `${i ? 400 : 600} 11.5px ${FONT}`;
            ctx.fillStyle = i ? c.muted : c.ink;
            ctx.fillText(text, left + 9, top + 18 + i * 16);
        });
        ctx.restore();
    }

    // A marker with a surface-colored ring, so it stays readable over lines and points.
    function marker(ctx, x, y, r, color) {
        ctx.beginPath();
        ctx.arc(x, y, r + 2, 0, 2 * Math.PI);
        ctx.fillStyle = surface();
        ctx.fill();
        ctx.beginPath();
        ctx.arc(x, y, r, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
    }

    // Grid, baseline, tick labels and axis titles. x / y: { at: value -> px, ticks, format, title }.
    function frame(ctx, box, x, y, c) {
        ctx.save();
        ctx.font = `11px ${FONT}`;
        ctx.fillStyle = c.muted;
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        y.ticks.forEach(value => {
            const py = Math.round(y.at(value)) + 0.5;
            line(ctx, box.left, py, box.right, py, c.grid, 1);
            ctx.fillText(y.format(value), box.left - 7, py);
        });
        line(ctx, box.left, box.bottom + 0.5, box.right, box.bottom + 0.5, alpha(c.l2, 0.55), 1);
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        x.ticks.forEach(value => {
            const px = x.at(value);
            line(ctx, px, box.bottom, px, box.bottom + 4, alpha(c.l2, 0.55), 1);
            ctx.fillText(x.format(value), px, box.bottom + 7);
        });
        if (x.title) {
            ctx.textAlign = "right";
            ctx.fillText(x.title, box.right, box.bottom + 21);
        }
        if (y.title) {
            ctx.textAlign = "left";
            ctx.textBaseline = "alphabetic";
            ctx.fillText(y.title, box.left - 7 - (y.titleIndent || 0), box.top - 8);
        }
        ctx.restore();
    }

    function drawScatter(canvas, scene, view) {
        const target = prepare(canvas);
        if (!target) return;
        const { ctx, width, height } = target, c = colors();
        const box = { left: 46, right: width - 16, top: 24, bottom: height - 40 };
        const [p0, p1] = scene.predRange, [d0, d1] = scene.range;
        const xs = niceScale(p0 - 0.05 * (p1 - p0), p1 + 0.08 * (p1 - p0), Math.max(3, Math.round((box.right - box.left) / 90)));
        const ys = niceScale(Math.max(0, d0 - 0.1 * (d1 - d0)), d1 + 0.15 * (d1 - d0), 4);
        const px = p => box.left + (p - xs.lo) / (xs.hi - xs.lo) * (box.right - box.left);
        const py = d => box.bottom - (d - ys.lo) / (ys.hi - ys.lo) * (box.bottom - box.top);
        const digits = step => Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
        frame(ctx, box, { at: px, ticks: xs.ticks, format: v => v.toFixed(digits(xs.step)), title: "prediction (model units)" },
            { at: py, ticks: ys.ticks, format: v => v.toFixed(digits(ys.step)), title: "GT depth (m)", titleIndent: 32 }, c);
        ctx.save();
        ctx.beginPath(); ctx.rect(box.left, box.top - 4, box.right - box.left, box.bottom - box.top + 4); ctx.clip();
        const limit = HUBER_DELTA * scene.unit / 0.5;  // Huber weight below 0.5
        const down = [];
        ctx.fillStyle = alpha(c.gt, 0.22);
        for (const i of scene.sample) {
            const x = px(scene.pred[i]), y = py(scene.gt[i]);
            if (view.weighted && Math.abs(view.a * scene.pred[i] + view.b - scene.gt[i]) > limit) { down.push([x, y]); continue; }
            ctx.beginPath(); ctx.arc(x, y, 1.6, 0, 2 * Math.PI); ctx.fill();
        }
        ctx.fillStyle = alpha(c.out, 0.85);
        down.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(x, y, 2.6, 0, 2 * Math.PI); ctx.fill(); });
        const fitLine = (a, b, color, width, dash) => line(ctx, px(xs.lo), py(a * xs.lo + b), px(xs.hi), py(a * xs.hi + b), color, width, dash);
        const l2 = scene.iterations[0];
        if (view.showL2) fitLine(l2.a, l2.b, c.l2, 1.5, [5, 4]);
        fitLine(view.a, view.b, c.pred, 2.2);
        ctx.restore();
        // direct labels at the lines' right ends, kept apart
        ctx.save();
        ctx.font = `600 11px ${FONT}`;
        ctx.textAlign = "right";
        const labels = [{ y: py(view.a * xs.hi + view.b), text: `fit  d = ${view.a.toFixed(2)}·p ${view.b < 0 ? "−" : "+"} ${Math.abs(view.b).toFixed(2)}`, color: c.ink }];
        if (view.showL2) labels.push({ y: py(l2.a * xs.hi + l2.b), text: "least squares", color: c.muted });
        labels.forEach(label => { label.y = Math.min(Math.max(label.y, box.top + 4), box.bottom - 6); });
        if (labels.length === 2 && Math.abs(labels[0].y - labels[1].y) < 14) {
            const [hi, lo] = labels[0].y < labels[1].y ? labels : [labels[1], labels[0]];
            hi.y -= (14 - (lo.y - hi.y)) / 2; lo.y = hi.y + 14;
        }
        labels.forEach(label => {
            ctx.fillStyle = surface();
            const w = ctx.measureText(label.text).width;
            ctx.globalAlpha = 0.8; ctx.fillRect(box.right - w - 6, label.y - 18, w + 6, 15); ctx.globalAlpha = 1;
            ctx.fillStyle = label.color;
            ctx.fillText(label.text, box.right - 3, label.y - 6);
        });
        ctx.restore();
    }

    // A series over steps 0 .. n-1, drawn up to `upto` (the rest as a faint preview of where it goes).
    //   cfg: values, upto, log, domain, yTicks, format, xTitle, yTitle, xTick(i) -> label | null, refs [{ value, label,
    //   dash, strong }], phases [{ from, to, label }], hover(i) -> [lines]
    function drawSeries(canvas, cfg) {
        const target = prepare(canvas);
        if (!target) return;
        const { ctx, width, height } = target, c = colors();
        const refs = cfg.refs || [];
        const box = { left: 46, right: width - (refs.length ? 104 : 16), top: cfg.phases ? 34 : 24, bottom: height - 40 };
        const n = cfg.values.length;
        const tf = cfg.log ? Math.log10 : v => v;
        let ys;
        if (cfg.domain) ys = { lo: cfg.domain[0], hi: cfg.domain[1], ticks: cfg.yTicks };
        else ys = niceScale(Math.min(...cfg.values), Math.max(...cfg.values), 4);
        const clamp = v => Math.min(ys.hi, Math.max(ys.lo, v));
        const vx = i => box.left + i / Math.max(n - 1, 1) * (box.right - box.left);
        const vy = v => box.bottom - (tf(clamp(v)) - tf(ys.lo)) / (tf(ys.hi) - tf(ys.lo)) * (box.bottom - box.top);
        // phase bands behind everything
        (cfg.phases || []).forEach((phase, k) => {
            const x0 = phase.from === 0 ? box.left : (vx(phase.from - 1) + vx(phase.from)) / 2;
            const x1 = phase.to === n - 1 ? box.right : (vx(phase.to) + vx(phase.to + 1)) / 2;
            if (k % 2 === 0) { ctx.fillStyle = alpha(c.l2, 0.08); ctx.fillRect(x0, box.top - 12, x1 - x0, box.bottom - box.top + 12); }
            ctx.save();
            ctx.font = `600 10.5px ${FONT}`;
            ctx.fillStyle = c.muted;
            ctx.textAlign = "center";
            if (ctx.measureText(phase.label).width < x1 - x0 - 4) ctx.fillText(phase.label, (x0 + x1) / 2, box.top - 2);
            ctx.restore();
        });
        const xTicks = [];
        for (let i = 0; i < n; i += 1) if (cfg.xTick(i) !== null) xTicks.push(i);
        frame(ctx, box, { at: vx, ticks: xTicks, format: cfg.xTick, title: cfg.xTitle },
            { at: vy, ticks: ys.ticks, format: cfg.format, title: cfg.yTitle, titleIndent: 32 }, c);
        // reference lines, labelled in the right margin (spread so they never overlap)
        const placed = refs.map(ref => ({ ...ref, y: vy(ref.value) })).sort((a, b) => a.y - b.y);
        placed.forEach((ref, k) => {
            line(ctx, box.left, ref.y, box.right, ref.y, ref.strong ? alpha(c.gt, 0.5) : c.l2, ref.strong ? 1.2 : 1.3, ref.dash);
            ref.labelY = k ? Math.max(ref.y, placed[k - 1].labelY + 13) : ref.y;
        });
        ctx.save();
        ctx.font = `11px ${FONT}`;
        ctx.textBaseline = "middle";
        placed.forEach(ref => {
            ctx.fillStyle = ref.strong ? c.ink : c.muted;
            ctx.fillText(ref.label, box.right + 8, Math.min(ref.labelY, height - 8));
        });
        ctx.restore();
        const points = cfg.values.map((v, i) => [vx(i), vy(v)]);
        polyline(ctx, points, alpha(c.pred, 0.18), 1.5, [3, 4]);
        if (cfg.upto >= 0) {
            const shown = points.slice(0, cfg.upto + 1);
            if (shown.length > 1) {
                const gradient = ctx.createLinearGradient(0, box.top, 0, box.bottom);
                gradient.addColorStop(0, alpha(c.pred, 0.2));
                gradient.addColorStop(1, alpha(c.pred, 0));
                ctx.beginPath();
                shown.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
                ctx.lineTo(shown[shown.length - 1][0], box.bottom);
                ctx.lineTo(shown[0][0], box.bottom);
                ctx.closePath();
                ctx.fillStyle = gradient;
                ctx.fill();
            }
            polyline(ctx, shown, c.pred, 2);
            if (n <= 40) shown.slice(0, -1).forEach(([x, y]) => marker(ctx, x, y, 2.2, c.pred));
            const [lx, ly] = shown[shown.length - 1];
            marker(ctx, lx, ly, 4.5, c.pred);
        }
        const hover = trackHover(canvas);
        if (hover.x !== null && hover.x >= box.left - 6 && hover.x <= box.right + 6) {
            const i = Math.round((hover.x - box.left) / (box.right - box.left) * (n - 1));
            const k = Math.min(n - 1, Math.max(0, i));
            line(ctx, vx(k), box.top - 4, vx(k), box.bottom, alpha(c.ink, 0.35), 1, [2, 3]);
            marker(ctx, vx(k), vy(cfg.values[k]), 4, k <= cfg.upto ? c.pred : alpha(c.pred, 0.45));
            tooltip(ctx, vx(k), vy(cfg.values[k]), cfg.hover(k), width, height, c);
        }
    }

    function iterationTicks(n, pixels) {
        const step = Math.max(1, Math.round(niceStep(n - 1, Math.max(2, Math.floor(pixels / 48)))));
        return i => (i % step === 0 || i === n - 1) && !(i !== n - 1 && n - 1 - i < step * 0.5) ? String(i) : null;
    }

    function drawCurve(canvas, scene, upto, key, format) {
        const values = scene.iterations.map(s => s[key]);
        drawSeries(canvas, {
            values, upto, format, xTitle: "IRLS iteration",
            xTick: iterationTicks(values.length, canvas.clientWidth - 70),
            hover: i => [i === 0 ? "least-squares start" : `IRLS iteration ${i}`, format(values[i], true)],
        });
    }

    function createDepthDemo() {
        const root = document.getElementById("home-depth-demo");
        if (!root) return null;
        const q = id => document.getElementById(id);
        const demo = { root, seed: 0, elapsed: 0, scene: null };
        const order = ["raw", "l2", "irls"];
        let request = 0;
        demo.reset = () => {
            const ticket = (request += 1);
            demo.elapsed = 0;
            loadManifest().then(manifest => {
                const meta = manifest.frames[demo.seed % manifest.frames.length];
                return loadFrame(meta).then(scene => ({ manifest, scene }));
            }).then(({ manifest, scene }) => {
                if (ticket !== request) return;
                demo.scene = scene;
                demo.timeline = timeline(scene.states.length, 1300, 650, 3600);
                demo.elapsed = 0;
                drawImage(q("home-depth-rgb"), scene.image);
                drawImage(q("home-depth-gt"), scene.gtImage);
                drawImage(q("home-depth-raw"), scene.predImage);
                const source = q("home-depth-source");
                if (source) {
                    const { fitPixels, holes, name } = scene.meta;
                    source.textContent = `${manifest.dataset} · ${name} · ${manifest.model} · `
                        + `${fitPixels.toLocaleString("en")} GT pixels, ${(100 * holes).toFixed(1)} % without GT · `
                        + `${manifest.credit}`;
                }
                if (demo.onReady) demo.onReady();
            }).catch(error => {
                const source = q("home-depth-source");
                if (source) source.textContent = `Could not load the example frame (${error.message}).`;
            });
        };
        demo.render = (index, t) => {
            if (!demo.scene) return;
            const states = demo.scene.states;
            const from = states[index], to = states[Math.min(index + 1, states.length - 1)];
            const k = ease(t);
            const a = from.a + (to.a - from.a) * k, b = from.b + (to.b - from.b) * k;
            const shown = k > 0.5 ? to : from;
            drawImage(q("home-depth-error"), errorImage(demo.scene, a, b));
            const view = { a, b, weighted: shown.key === "irls", showL2: index >= 1 };
            drawScatter(q("home-depth-scatter"), demo.scene, view);
            const upto = shown.key === "raw" ? -1 : shown.iteration;
            drawCurve(q("home-depth-loss"), demo.scene, upto, "loss", (v, full) => v.toFixed(full ? 5 : 4));
            const percent = (v, full) => `${(100 * v).toFixed(full ? 2 : 1)} %`;
            drawCurve(q("home-depth-absrel"), demo.scene, upto, "absRel", percent);
            const current = upto >= 0 ? demo.scene.iterations[upto] : null;
            const setValue = (id, text) => { const el = q(id); if (el && el.textContent !== text) el.textContent = text; };
            setValue("home-depth-loss-value", current ? current.loss.toFixed(4) : "–");
            setValue("home-depth-absrel-value", current ? percent(current.absRel, true) : "–");
            setValue("home-depth-scatter-value", shown.key === "raw" ? "before alignment"
                : shown.key === "l2" ? "least-squares start" : `IRLS ${shown.iteration}`);
            setSteps(q("home-depth-steps"), shown.key, order);
            const its = demo.scene.iterations.length - 1;
            const stats = q("home-depth-stats");
            if (stats) {
                const raw = shown.key === "raw";
                const label = raw ? "before alignment" : shown.key === "l2" ? "least-squares start"
                    : `IRLS ${shown.iteration} / ${its}${shown.iteration === its ? " · converged" : ""}`;
                stats.innerHTML = [["step", label], ["scale", raw ? "–" : shown.a.toFixed(3)],
                    ["shift", raw ? "–" : `${shown.b.toFixed(3)} m`],
                    ["AbsRel", raw ? "–" : `${(100 * shown.absRel).toFixed(2)} %`],
                    ["δ < 1.25", raw ? "–" : `${(100 * shown.delta1).toFixed(1)} %`]]
                    .map(([k2, v]) => `<div><span>${k2}</span><b>${v}</b></div>`).join("");
            }
        };
        return demo;
    }

    // ---------------------------------------------------------------- trajectory: 3D maths

    const v3 = {
        add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
        sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
        mul: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
        dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
        cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
        norm: a => Math.hypot(a[0], a[1], a[2]),
        unit: a => { const n = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / n, a[1] / n, a[2] / n]; },
    };
    // 3x3 matrices, row-major arrays of 9
    const m3 = {
        mul: (A, B) => { const C = new Array(9).fill(0); for (let i = 0; i < 3; i += 1) for (let j = 0; j < 3; j += 1) for (let k = 0; k < 3; k += 1) C[3 * i + j] += A[3 * i + k] * B[3 * k + j]; return C; },
        vec: (A, v) => [A[0] * v[0] + A[1] * v[1] + A[2] * v[2], A[3] * v[0] + A[4] * v[1] + A[5] * v[2], A[6] * v[0] + A[7] * v[1] + A[8] * v[2]],
        t: A => [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]],
        axisAngle: (axis, angle) => {
            const [x, y, z] = v3.unit(axis), c = Math.cos(angle), s = Math.sin(angle), C = 1 - c;
            return [c + x * x * C, x * y * C - z * s, x * z * C + y * s, y * x * C + z * s, c + y * y * C, y * z * C - x * s, z * x * C - y * s, z * y * C + x * s, c + z * z * C];
        },
        angle: (A, B) => { const R = m3.mul(A, m3.t(B)); return Math.acos(Math.min(1, Math.max(-1, (R[0] + R[4] + R[8] - 1) / 2))); },
    };

    const applySim = (sim, p) => v3.add(v3.mul(m3.vec(sim.R, p), sim.s), sim.t);

    function quatToMatrix([x, y, z, w]) {  // scipy order (x, y, z, w) -> row-major 3x3
        return [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)];
    }
    const simOf = sim => ({ R: quatToMatrix(sim.q), s: sim.s, t: sim.t });

    // ---------------------------------------------------------------- trajectory: cases

    // assets/align/trajectory/manifest.json lists six scenarios of the alignment study
    // (scripts/test/compare_trajectory_alignment.py), three cases each. A case is one trial of the study on a real GT
    // trajectory (or its synthetic near-collinear path), with every step of the SLAM evaluator's estimator recorded:
    // the 25 two-frame RANSAC samples, the best one and its inliers, the inlier refit, each Cauchy IRLS iteration and
    // the final estimate, plus Umeyama on all centers (evo) and the previous default (R-DAGSfM) for comparison.
    const TRAJ_ROOT = "assets/align/trajectory/";
    const SHOWN_SAMPLES = 6;
    let trajManifest = null;
    const caseCache = new Map();

    function loadTrajManifest() {
        if (!trajManifest) {
            trajManifest = fetch(`${TRAJ_ROOT}manifest.json`).then(r => {
                if (!r.ok) throw new Error(`manifest: HTTP ${r.status}`);
                return r.json();
            });
        }
        return trajManifest;
    }

    function loadCase(name) {
        if (!caseCache.has(name)) {
            caseCache.set(name, fetch(`${TRAJ_ROOT}${name}`).then(r => {
                if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
                return r.json();
            }).then(makeTrajectoryScene));
        }
        return caseCache.get(name);
    }

    function bounds(points) {
        const center = v3.mul(points.reduce((s, p) => v3.add(s, p), [0, 0, 0]), 1 / points.length);
        return { center, radius: Math.max(...points.map(p => v3.norm(v3.sub(p, center))), 1e-9) * 1.08 };
    }

    function makeTrajectoryScene(data) {
        const gt = data.gt.c, gtR = data.gt.q.map(quatToMatrix);
        const pred = data.pred.c, predR = data.pred.q.map(quatToMatrix);
        const n = gt.length;
        // display basis: the GT cameras' mean "up" (OpenCV cameras: up = -y column) becomes +y on screen
        let up = [0, 0, 0];
        gtR.forEach(R => { up = v3.add(up, [-R[1], -R[4], -R[7]]); });
        up = v3.unit(up);
        const helper = Math.abs(up[0]) < 0.9 ? [1, 0, 0] : [0, 0, 1];
        const ex = v3.unit(v3.cross(helper, up)), ez = v3.cross(ex, up);
        const display = [ex[0], ex[1], ex[2], up[0], up[1], up[2], ez[0], ez[1], ez[2]];
        const view = p => m3.vec(display, p);
        const viewGt = bounds(gt.map(view));
        // the raw prediction is drawn beside the GT at GT size (its own frame has an arbitrary scale)
        const predBounds = bounds(pred);
        const side = v3.add(viewGt.center, [2.4 * viewGt.radius, 0, 0]);
        const s0 = viewGt.radius / predBounds.radius;
        const back = m3.t(display);  // display -> data axes
        const rawT = v3.sub(m3.vec(back, side), v3.mul(m3.vec(back, predBounds.center), s0));
        const rawSim = { R: back, s: s0, t: rawT };
        const viewAll = bounds(gt.map(view).concat(pred.map(p => view(applySim(rawSim, p)))));
        // the study's LMedS inlier threshold; floored for display, since noise-free cases put it at ~1e-9, below the
        // rounding of the exported poses
        const threshold = Math.max(data.best.threshold, 1e-3 * data.extent);
        const centerErrors = sim => pred.map((p, i) => v3.norm(v3.sub(applySim(sim, p), gt[i])));
        const state = (key, sim, stats, extra = {}) => {
            const S = simOf(sim);
            return { key, ...S, stats, errors: centerErrors(S), ...extra };
        };
        const states = [{ key: "raw", ...rawSim, stats: null, errors: null }];
        data.samples.slice(0, SHOWN_SAMPLES).forEach((sample, k) => {
            states.push(state("ransac", sample.sim, sample.stats, { pair: sample.pair, sample: k + 1,
                lmeds: sample.lmeds }));
        });
        states.push(state("ransac", data.best.sim, data.best.stats, { chosen: true,
            inliers: data.best.inliers.map(Boolean) }));
        states.push(state("refit", data.refit.sim, data.refit.stats, { inliers: data.best.inliers.map(Boolean) }));
        data.irls.forEach((step, k) => {
            states.push(state("irls", step.sim, step.stats, { iteration: k + 1, weights: step.weights }));
        });
        const last = states[states.length - 1];
        last.final = true;
        if (data.final.scaleCollapsed) states.push(state("irls", data.final.sim, data.final.stats, { final: true }));
        const umeyama = simOf(data.umeyama.sim);
        const outlierFrames = data.scenario === "outliers" || data.scenario === "collinear"
            ? data.clean.map(v => !v) : null;
        return { data, gt, gtR, pred, predR, n, display, view, viewGt, viewAll, threshold, states, umeyama, outlierFrames,
            clean: data.clean.map(Boolean), extent: data.extent };
    }

    function interpolateSim(from, to, k) {
        // rotation: slerp through the relative axis-angle; scale: geometric; translation: linear
        const rel = m3.mul(to.R, m3.t(from.R));
        const angle = Math.acos(Math.min(1, Math.max(-1, (rel[0] + rel[4] + rel[8] - 1) / 2)));
        let R = to.R;
        if (angle > 1e-6 && angle < Math.PI - 1e-3) {
            const axis = [rel[7] - rel[5], rel[2] - rel[6], rel[3] - rel[1]];
            R = m3.mul(m3.axisAngle(axis, angle * k), from.R);
        } else if (angle <= 1e-6) {
            R = from.R;
        }
        return { R, s: Math.exp(Math.log(from.s) + (Math.log(to.s) - Math.log(from.s)) * k),
            t: v3.add(from.t, v3.mul(v3.sub(to.t, from.t), k)) };
    }

    // ---------------------------------------------------------------- trajectory: drawing

    function projector(width, height, frame, azimuth) {
        const elevation = 0.5, ca = Math.cos(azimuth), sa = Math.sin(azimuth);
        const ce = Math.cos(elevation), se = Math.sin(elevation);
        const scale = Math.min(width, height) * 0.46 / frame.radius;
        const distance = frame.radius * 3.2;
        return p => {
            const q = v3.sub(p, frame.center);
            const x = ca * q[0] + sa * q[2], z0 = -sa * q[0] + ca * q[2];
            const y = ce * q[1] - se * z0, z = se * q[1] + ce * z0;
            const f = distance / Math.max(distance + z, distance * 0.2);
            return [width / 2 + x * scale * f, height / 2 - y * scale * f, z];
        };
    }

    function frustum(ctx, project, center, R, size, color, width) {
        // OpenCV camera-to-world in display axes: right = x column, up = -y column, forward = z column
        const right = [R[0], R[3], R[6]], up = [-R[1], -R[4], -R[7]], forward = [R[2], R[5], R[8]];
        const base = v3.add(center, v3.mul(forward, size));
        const corners = [[1, 1], [1, -1], [-1, -1], [-1, 1]].map(([a, b]) =>
            project(v3.add(base, v3.add(v3.mul(right, a * size * 0.62), v3.mul(up, b * size * 0.45)))));
        const apex = project(center);
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.lineJoin = "round";
        ctx.beginPath();
        corners.forEach(([x, y]) => { ctx.moveTo(apex[0], apex[1]); ctx.lineTo(x, y); });
        corners.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        ctx.closePath();
        ctx.stroke();
        ctx.restore();
    }

    // frames the current step treats as outliers: beyond the RANSAC threshold, or Cauchy weight below 1/2
    function outlierMask(scene, view) {
        if (view.key === "raw") return null;
        if (view.weights) return view.weights.map(w => w < 0.5);
        if (view.inliers) return view.inliers.map(v => !v);
        return view.errors.map(e => e > scene.threshold);
    }

    function drawTrajectory(canvas, scene, view, azimuth) {
        const target = prepare(canvas);
        if (!target) return;
        const { ctx, width, height } = target, c = colors();
        const project = projector(width, height, view.frame, azimuth);
        const D = scene.display, toView = p => m3.vec(D, p);
        // ground grid under the GT path
        const { center, radius } = scene.viewGt;
        const floor = Math.min(...scene.gt.map(p => toView(p)[1])) - 0.25 * radius;
        const step = radius / 3;
        for (let k = -6; k <= 6; k += 1) {
            const a = project([center[0] + k * step, floor, center[2] - 6 * step]);
            const b = project([center[0] + k * step, floor, center[2] + 6 * step]);
            line(ctx, a[0], a[1], b[0], b[1], c.grid, 1);
            const d = project([center[0] - 6 * step, floor, center[2] + k * step]);
            const e = project([center[0] + 6 * step, floor, center[2] + k * step]);
            line(ctx, d[0], d[1], e[0], e[1], c.grid, 1);
        }
        const P = p => project(toView(p));
        // paths break at frames far from both neighbours' track (outlier poses), so they draw as frustums only
        const path = (points, color, width, dash, skip) => {
            const limit = 0.25 * scene.extent;  // every drawn point is at GT scale
            let run = [], previous = null;
            const flush = () => { if (run.length > 1) polyline(ctx, run, color, width, dash); run = []; };
            points.forEach((p, i) => {
                if (skip && skip[i]) { flush(); previous = null; return; }
                if (previous && v3.norm(v3.sub(p, previous)) > limit) flush();
                run.push(P(p));
                previous = p;
            });
            flush();
        };
        const outliers = outlierMask(scene, view);
        if (view.showUmeyama) {
            path(scene.pred.map(p => applySim(scene.umeyama, p)), c.l2, 1.4, [5, 4], scene.outlierFrames);
        }
        const aligned = scene.pred.map(p => applySim(view, p));
        path(scene.gt, alpha(c.gt, 0.6), 1.6, null, null);
        path(aligned, alpha(c.pred, 0.75), 1.4, null, view.key === "raw" ? scene.outlierFrames : outliers);
        if (view.key !== "raw") {  // per-frame error lines
            const every = Math.max(1, Math.round(scene.n / 70));
            aligned.forEach((p, i) => {
                if (i % every) return;
                const a = P(p), b = P(scene.gt[i]);
                if (outliers && outliers[i]) return;  // outlier frames: their frustum alone
                line(ctx, a[0], a[1], b[0], b[1], alpha(c.l2, 0.45), 1);
            });
        }
        const size = radius * 0.07;
        const every = Math.max(1, Math.round(scene.n / 28));
        for (let i = 0; i < scene.n; i += every) {
            frustum(ctx, P, scene.gt[i], m3.mul(D, scene.gtR[i]), size, alpha(c.gt, 0.75), 1.1);
        }
        for (let i = 0; i < scene.n; i += every) {
            const R = m3.mul(D, m3.mul(view.R, scene.predR[i]));
            const bad = outliers && outliers[i];
            frustum(ctx, project, toView(aligned[i]), R, size, bad ? c.out : c.pred, bad ? 1.2 : 1.4);
        }
        if (view.pair) {
            view.pair.forEach(i => {
                [[P(scene.gt[i]), c.gt], [P(aligned[i]), c.pred]].forEach(([[x, y], color]) => {
                    ctx.beginPath(); ctx.arc(x, y, 7, 0, 2 * Math.PI);
                    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
                });
            });
        }
    }

    const RATIO_LO = 0.1, RATIO_HI = 10;
    const stepLabel = s => (s.key === "ransac" ? (s.chosen ? "best" : `s${s.sample}`) : s.key === "refit" ? "refit"
        : `C${s.iteration || ""}`);
    const stepName = (s, samples) => (s.key === "ransac" ? (s.chosen ? `best of ${samples} samples (LMedS)`
        : `two-frame sample ${s.sample}`) : s.key === "refit" ? "inlier refit"
        : s.iteration ? `Cauchy IRLS ${s.iteration}` : "final estimate");

    function drawRatioCurve(canvas, scene, upto) {
        const states = scene.states.slice(1);
        const values = states.map(s => s.stats.ateRatio);
        const phases = [];
        states.forEach((s, i) => {
            const label = s.key === "ransac" ? "RANSAC" : s.key === "refit" ? "refit" : "Cauchy IRLS";
            if (phases.length && phases[phases.length - 1].label === label) phases[phases.length - 1].to = i;
            else phases.push({ from: i, to: i, label });
        });
        const every = Math.max(1, Math.ceil(states.length / Math.max(4, Math.floor((canvas.clientWidth - 150) / 34))));
        const d = scene.data;
        drawSeries(canvas, {
            values, upto, log: true, domain: [RATIO_LO, RATIO_HI], yTicks: [0.1, 0.3, 1, 3, 10],
            format: v => `${v}`, yTitle: "ATE ratio (log)",
            xTick: i => (i % every === 0 || i === states.length - 1 ? stepLabel(states[i]) : null),
            refs: [{ value: 1, label: "1 = exact", strong: true },
                { value: d.umeyama.stats.ateRatio, label: `Umeyama ${fmtRatio(d.umeyama.stats.ateRatio)}`, dash: [5, 4] },
                { value: d.dagsfm.stats.ateRatio, label: `R-DAGSfM ${fmtRatio(d.dagsfm.stats.ateRatio)}`, dash: [2, 3] }],
            phases,
            hover: i => [stepName(states[i], d.samples.length), `ATE ratio ${fmtRatio(values[i])}`,
                `rotation error ${fmtDeg(states[i].stats.rotErr)}`],
        });
    }

    // ---------------------------------------------------------------- trajectory: three.js view
    //
    // The frustums in a WebGL scene, as on the FreeScale project page: a floor grid, the GT path and cameras, the
    // predicted cameras moving with each step of the fit (outliers in their own color), the error of every frame, and
    // Umeyama's result once the fit has started. Drag orbits, Ctrl/Cmd + wheel zooms. Falls back to the 2D canvas
    // drawing if three.js cannot load.

    let threePromise = null;
    function loadThree() {
        threePromise ||= Promise.all([import("three"), import("three/addons/controls/OrbitControls.js")])
            .then(([THREE, addons]) => ({ THREE, OrbitControls: addons.OrbitControls }));
        return threePromise;
    }

    function createTraj3D(canvas, { THREE, OrbitControls }) {
        const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        renderer.setClearColor(0x000000, 0);
        const scene3 = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 200);
        const HOME = new THREE.Vector3(1.45, 1.0, 1.9);
        camera.position.copy(HOME);
        const controls = new OrbitControls(camera, canvas);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.enablePan = false;
        controls.enableZoom = false;
        controls.autoRotate = !reducedMotion;
        controls.autoRotateSpeed = 0.55;
        controls.target.set(0, 0, 0);
        controls.update();
        canvas.addEventListener("wheel", event => {
            if (!(event.ctrlKey || event.metaKey)) return;  // plain scrolling keeps moving the page
            event.preventDefault();
            const offset = camera.position.clone().sub(controls.target);
            offset.multiplyScalar(event.deltaY > 0 ? 1.1 : 0.9);
            offset.setLength(Math.min(9, Math.max(0.6, offset.length())));
            camera.position.copy(controls.target).add(offset);
        }, { passive: false });
        canvas.addEventListener("dblclick", () => { camera.position.copy(HOME); controls.target.set(0, 0, 0); });

        const world = new THREE.Group();
        scene3.add(world);
        const mats = {};
        const material = (key, make) => (mats[key] ||= make());
        const lineMat = (key, opacity) => material(key, () => new THREE.LineBasicMaterial({ transparent: true, opacity }));
        const fillMat = (key, opacity) => material(key, () => new THREE.MeshBasicMaterial({ transparent: true, opacity,
            side: THREE.DoubleSide, depthWrite: false }));
        const dashMat = () => material("umeyama", () => new THREE.LineDashedMaterial({ transparent: true, opacity: 0.85,
            dashSize: 1, gapSize: 1 }));
        const dotMat = key => material(`dot-${key}`, () => new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.9 }));
        let palette = "";
        function applyColors() {
            const c = colors();
            const key = [c.gt, c.pred, c.out, c.l2, c.grid].join();
            if (key === palette) return;
            palette = key;
            const set = (name, color) => mats[name]?.color.set(new THREE.Color(color));
            const ring = css("--home-ring");
            set("grid", c.l2); set("gtPath", c.gt); set("gtLine", c.gt); set("gtFill", c.gt);
            set("predPath", c.pred); set("predLine", c.pred); set("predFill", c.pred);
            set("outLine", c.out); set("outFill", c.out); set("err", c.l2); set("umeyama", c.l2);
            set("dot-gt", c.gt); set("dot-pred", c.pred); set("dot-start", ring);
        }

        // one frustum in OpenCV camera axes (x right, y down, z forward), unit size
        const frustumLines = new THREE.BufferGeometry();
        {
            const corners = [[0.62, 0.45], [-0.62, 0.45], [-0.62, -0.45], [0.62, -0.45]].map(([x, y]) => [x, y, 1]);
            const pts = [];
            corners.forEach((corner, i) => { pts.push(0, 0, 0, ...corner, ...corner, ...corners[(i + 1) % 4]); });
            pts.push(0, -0.45, 1, 0, -0.75, 1);  // "up" tick (OpenCV up is -y), so the camera's roll is visible
            frustumLines.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
        }
        const frustumFill = new THREE.PlaneGeometry(1.24, 0.9);
        frustumFill.translate(0, 0, 1);
        const sphere = new THREE.SphereGeometry(1, 16, 12);

        function makeFrustum(line, fill) {
            const group = new THREE.Group();
            group.matrixAutoUpdate = false;
            group.add(new THREE.LineSegments(frustumLines, line));
            group.add(new THREE.Mesh(frustumFill, fill));
            return group;
        }
        const matrix = new THREE.Matrix4();
        function place(object, R, p, size) {  // R: row-major camera-to-world rotation in display axes
            matrix.set(R[0] * size, R[1] * size, R[2] * size, p[0], R[3] * size, R[4] * size, R[5] * size, p[1],
                R[6] * size, R[7] * size, R[8] * size, p[2], 0, 0, 0, 1);
            object.matrix.copy(matrix);
            object.matrixWorldNeedsUpdate = true;
        }
        function segments(count, mat) {
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(count * 6), 3));
            const object = new THREE.LineSegments(geometry, mat);
            object.frustumCulled = false;
            return object;
        }
        function setSegments(object, pairs) {  // pairs: [[a, b] | null], a / b display points; null hides a slot
            const array = object.geometry.attributes.position.array;
            pairs.forEach((pair, k) => {
                const [a, b] = pair || [[0, 0, 0], [0, 0, 0]];
                array.set(a, 6 * k); array.set(pair ? b : a, 6 * k + 3);
            });
            object.geometry.attributes.position.needsUpdate = true;
        }

        let current = null;
        function clear() {
            world.children.slice().forEach(child => {
                world.remove(child);
                child.traverse(node => { if (node.geometry && ![frustumLines, frustumFill, sphere].includes(node.geometry)) node.geometry.dispose(); });
            });
        }

        function setCase(sc) {
            clear();
            const D = sc.display, toView = p => m3.vec(D, p);
            const size = sc.viewGt.radius * 0.085;
            const every = Math.max(1, Math.round(sc.n / 18));
            const shown = [];
            for (let i = 0; i < sc.n; i += every) shown.push(i);
            const gtView = sc.gt.map(toView);
            // floor grid just under the GT path
            const { center, radius } = sc.viewGt;
            const floor = Math.min(...gtView.map(p => p[1])) - 0.3 * radius;
            const half = 1.6 * radius, cells = 12, grid = [];
            for (let k = 0; k <= cells; k += 1) {
                const t = -half + 2 * half * k / cells;
                grid.push(center[0] + t, floor, center[2] - half, center[0] + t, floor, center[2] + half);
                grid.push(center[0] - half, floor, center[2] + t, center[0] + half, floor, center[2] + t);
            }
            const gridGeometry = new THREE.BufferGeometry();
            gridGeometry.setAttribute("position", new THREE.Float32BufferAttribute(grid, 3));
            world.add(new THREE.LineSegments(gridGeometry, lineMat("grid", 0.22)));
            // GT: path, cameras, start
            const gtPath = new THREE.Line(new THREE.BufferGeometry().setFromPoints(gtView.map(p => new THREE.Vector3(...p))),
                lineMat("gtPath", 0.55));
            world.add(gtPath);
            shown.forEach(i => {
                const f = makeFrustum(lineMat("gtLine", 0.6), fillMat("gtFill", 0.07));
                place(f, m3.mul(D, sc.gtR[i]), gtView[i], size);
                world.add(f);
            });
            const start = new THREE.Mesh(sphere, dotMat("start"));
            start.scale.setScalar(size * 0.22);
            start.position.set(...gtView[0]);
            world.add(start);
            // prediction: path (breaks at outliers), cameras, error lines, Umeyama; updated every frame
            const predPath = segments(sc.n - 1, lineMat("predPath", 0.8));
            const errors = segments(sc.n, lineMat("err", 0.5));
            const umeyama = segments(sc.n - 1, dashMat());
            const predFrustums = shown.map(() => makeFrustum(lineMat("predLine", 0.95), fillMat("predFill", 0.16)));
            const pair = [dotMat("gt"), dotMat("pred"), dotMat("gt"), dotMat("pred")].map(mat => {
                const dot = new THREE.Mesh(sphere, mat);
                dot.scale.setScalar(size * 0.3);
                return dot;
            });
            world.add(predPath, errors, umeyama, ...predFrustums, ...pair);
            // Umeyama is fixed per case
            const limit = 0.25 * sc.extent;
            const umeyamaView = sc.pred.map(p => toView(applySim(sc.umeyama, p)));
            setSegments(umeyama, umeyamaView.slice(1).map((p, k) => {
                const skip = sc.outlierFrames && (sc.outlierFrames[k] || sc.outlierFrames[k + 1]);
                return skip || v3.norm(v3.sub(p, umeyamaView[k])) > limit ? null : [umeyamaView[k], p];
            }));
            umeyama.computeLineDistances();
            dashMat().dashSize = dashMat().gapSize = sc.viewGt.radius * 0.035;
            current = { sc, toView, size, shown, predPath, errors, umeyama, predFrustums, pair, limit };
            palette = "";
            applyColors();
        }

        function update(view) {
            if (!current) return;
            const { sc, toView, size, shown, predPath, errors, umeyama, predFrustums, pair, limit } = current;
            applyColors();
            const D = sc.display;
            const aligned = sc.pred.map(p => toView(applySim(view, p)));
            const gtView = sc.gt.map(toView);
            const outliers = outlierMask(sc, view);
            const skip = view.key === "raw" ? sc.outlierFrames : outliers;
            setSegments(predPath, aligned.slice(1).map((p, k) => {
                if (skip && (skip[k] || skip[k + 1])) return null;
                return v3.norm(v3.sub(p, aligned[k])) > limit ? null : [aligned[k], p];
            }));
            const errorEvery = Math.max(1, Math.round(sc.n / 70));
            setSegments(errors, aligned.map((p, i) => (view.key === "raw" || i % errorEvery || (outliers && outliers[i])
                ? null : [p, gtView[i]])));
            umeyama.visible = Boolean(view.showUmeyama);
            shown.forEach((i, k) => {
                const f = predFrustums[k];
                const bad = Boolean(outliers && outliers[i]);
                f.children[0].material = bad ? lineMat("outLine", 0.95) : lineMat("predLine", 0.95);
                f.children[1].material = bad ? fillMat("outFill", 0.24) : fillMat("predFill", 0.16);
                place(f, m3.mul(D, m3.mul(view.R, sc.predR[i])), aligned[i], size);
            });
            pair.forEach((dot, k) => {
                const i = view.pair?.[k >> 1];
                dot.visible = i !== undefined;
                if (dot.visible) dot.position.set(...(k % 2 ? aligned[i] : gtView[i]));
            });
            // frame: world units are the shown region's radius, centered
            const f = view.frame;
            world.scale.setScalar(1 / f.radius);
            world.position.set(-f.center[0] / f.radius, -f.center[1] / f.radius, -f.center[2] / f.radius);
            const w = canvas.clientWidth, h = canvas.clientHeight;
            if (w && h && (canvas.width !== Math.round(w * renderer.getPixelRatio()) || canvas.height !== Math.round(h * renderer.getPixelRatio()))) {
                renderer.setSize(w, h, false);
                camera.aspect = w / h;
                camera.updateProjectionMatrix();
            }
            controls.autoRotate = !reducedMotion && !state.paused;
            controls.update();
            renderer.render(scene3, camera);
        }

        return { setCase, update, resetView: () => { camera.position.copy(HOME); controls.target.set(0, 0, 0); } };
    }

    const fmtRatio = v => (v === null || v === undefined ? "–" : v.toFixed(2));
    const fmtDeg = v => {
        if (v === null || v === undefined) return "–";
        return v < 0.005 ? "0°" : `${v < 10 ? v.toFixed(2) : v.toFixed(1)}°`;
    };
    const fmtCv = v => (v === undefined ? "–" : v < 1e-4 ? "< 0.01 %" : `${(100 * v).toFixed(1)} %`);

    function comparisonTable(scene, scenario, done) {
        const d = scene.data, paper = scenario.paper;
        const paperCell = key => {
            const [ratio, rot] = paper[key];
            return ratio !== null ? fmtRatio(ratio) : fmtDeg(rot);
        };
        const paperHead = paper.ours[0] !== null ? "study median ATE ratio" : "study median rot. error";
        const rows = [["Umeyama (evo)", d.umeyama.stats, undefined, "umeyama"],
            ["R-DAGSfM (previous)", d.dagsfm.stats, d.seedCV.dagsfm, "dagsfm"],
            ["Ours", done ? d.final.stats : null, done ? d.seedCV.ours : undefined, "ours"]];
        return `<table class="align-compare"><thead><tr><th>this case</th><th>ATE ratio</th><th>rot. error</th>`
            + `<th>seed spread</th><th title="${paper.source}">${paperHead}</th></tr></thead><tbody>`
            + rows.map(([label, st, cv, key]) => `<tr class="${key === "ours" ? "is-ours" : ""}"><th>${label}</th>`
                + `<td>${st ? fmtRatio(st.ateRatio) : "…"}</td><td>${st ? fmtDeg(st.rotErr) : "…"}</td>`
                + `<td>${key === "umeyama" ? "n/a" : st ? fmtCv(cv) : "…"}</td><td>${paperCell(key)}</td></tr>`).join("")
            + "</tbody></table>";
    }

    function createTrajectoryDemo() {
        const root = document.getElementById("home-traj-demo");
        if (!root) return null;
        const q = id => document.getElementById(id);
        const demo = { root, seed: 0, elapsed: 0, azimuth: 0.7, scene: null, scenario: 0, view3d: null };
        const order = ["raw", "ransac", "refit", "irls"];
        let request = 0;
        const tabs = q("home-traj-scenarios");
        const canvas3d = q("home-traj-3d");
        if (canvas3d) {
            loadThree().then(lib => {
                demo.view3d = createTraj3D(canvas3d, lib);
                if (demo.scene) demo.view3d.setCase(demo.scene);
                root.classList.add("has-3d");
            }).catch(error => console.warn("home trajectory: three.js unavailable, drawing in 2D", error));
        }
        q("home-traj-reset")?.addEventListener("click", () => demo.view3d?.resetView());
        demo.reset = () => {
            const ticket = (request += 1);
            demo.elapsed = 0;
            loadTrajManifest().then(manifest => {
                if (tabs && !tabs.childElementCount) {
                    tabs.innerHTML = manifest.scenarios.map((s, i) => `<button type="button" role="tab" `
                        + `data-scenario="${i}" aria-selected="false">${s.label}</button>`).join("");
                    tabs.querySelectorAll("button").forEach(button => button.addEventListener("click", () => {
                        demo.scenario = Number(button.dataset.scenario);
                        demo.seed = 0;
                        demo.reset();
                    }));
                }
                const scenario = manifest.scenarios[demo.scenario % manifest.scenarios.length];
                const name = scenario.cases[demo.seed % scenario.cases.length];
                return loadCase(name).then(scene => ({ manifest, scenario, scene }));
            }).then(({ manifest, scenario, scene }) => {
                if (ticket !== request) return;
                demo.scene = scene;
                demo.scenarioInfo = scenario;
                if (demo.view3d) demo.view3d.setCase(scene);
                const caseLabel = q("home-traj-case");
                if (caseLabel) caseLabel.textContent = `${scenario.label} · ${scene.data.source.split(" · ")[0]}`;
                demo.timeline = timeline(scene.states.length, 1000, 700, 5200);
                demo.elapsed = 0;
                if (tabs) {
                    tabs.querySelectorAll("button").forEach(b => {
                        b.setAttribute("aria-selected", String(Number(b.dataset.scenario) === demo.scenario));
                    });
                }
                const note = q("home-traj-note");
                if (note) note.textContent = scenario.note;
                const source = q("home-traj-source");
                if (source) {
                    const dataset = scene.data.source.startsWith("TUM") ? "TUM RGB-D"
                        : scene.data.source.startsWith("SpatialArena") ? "SpatialArena" : null;
                    const every = dataset ? ` (every ${manifest.stride}th GT pose`
                        + `${dataset === "TUM RGB-D" ? ", as the short range track" : ""})` : "";
                    const credit = dataset ? ` · ${manifest.credits[dataset]}` : "";
                    source.textContent = `${scene.data.source} · ${scene.n} frames${every}${credit}`
                        + ` · study: ${scenario.paper.source}`;
                }
                if (demo.onReady) demo.onReady();
            }).catch(error => {
                const note = q("home-traj-note");
                if (note) note.textContent = `Could not load the trajectory cases (${error.message}).`;
            });
        };
        demo.render = (index, t, delta) => {
            if (!demo.scene) return;
            const scene = demo.scene, states = scene.states;
            const from = states[index], to = states[Math.min(index + 1, states.length - 1)];
            const k = ease(t);
            const shown = k > 0.5 ? to : from;
            if (!reducedMotion && !state.paused) demo.azimuth += (delta || 0) * 0.00012;
            const sim = interpolateSim(from, to, k);
            // zoom from both frames onto the GT path as the first fit lands
            const { viewAll, viewGt } = scene;
            const z = from.key === "raw" ? (to.key === "raw" ? 0 : k) : 1;
            const frame = { center: v3.add(viewAll.center, v3.mul(v3.sub(viewGt.center, viewAll.center), z)),
                radius: viewAll.radius + (viewGt.radius - viewAll.radius) * z };
            const done = shown.final === true;
            const view = { ...sim, frame, key: shown.key, errors: shown.errors, inliers: shown.inliers,
                weights: shown.weights, pair: shown.pair || null, showUmeyama: order.indexOf(shown.key) >= 2 };
            if (demo.view3d) demo.view3d.update(view);
            else drawTrajectory(q("home-traj-canvas"), scene, view, demo.azimuth);
            drawRatioCurve(q("home-traj-curve"), scene, states.indexOf(shown) - 1);
            const name = stepName(shown, scene.data.samples.length);
            const stepText = shown.key === "raw" ? "Own frame (before alignment)"
                : `${name[0].toUpperCase()}${name.slice(1)}${done ? " · converged" : ""}`;
            const stepPill = q("home-traj-step");
            if (stepPill && stepPill.textContent !== stepText) stepPill.textContent = stepText;
            const ratio = q("home-traj-ratio-value");
            const ratioText = shown.stats ? fmtRatio(shown.stats.ateRatio) : "–";
            if (ratio && ratio.textContent !== ratioText) ratio.textContent = ratioText;
            setSteps(q("home-traj-steps"), shown.key, order);
            const compare = q("home-traj-compare");
            if (compare) compare.innerHTML = comparisonTable(scene, demo.scenarioInfo, done);
            const stats = q("home-traj-stats");
            if (stats) {
                const samples = scene.data.samples.length;
                const its = scene.data.irls.length;
                const label = shown.key === "raw" ? "own frame"
                    : shown.key === "ransac" ? (shown.chosen ? `best of ${samples} samples (LMedS)`
                        : `sample ${shown.sample} / ${samples}`)
                        : shown.key === "refit" ? "inlier refit"
                            : `Cauchy IRLS ${Math.min(shown.iteration || its, its)} / ${its}`
                                + `${done ? " · converged" : ""}`;
                const mask = outlierMask(scene, { ...shown });
                const kept = mask ? mask.filter(v => !v).length : scene.n;
                const st = shown.stats;
                stats.innerHTML = [["step", label], ["ATE ratio", st ? fmtRatio(st.ateRatio) : "–"],
                    ["rotation error", st ? fmtDeg(st.rotErr) : "–"],
                    ["scale error", st ? `${(100 * (Math.exp(st.scaleErr) - 1)).toFixed(1)} %` : "–"],
                    [shown.weights ? "weight ≥ ½" : "inliers", shown.key === "raw" ? "–" : `${kept} / ${scene.n}`]]
                    .map(([k2, v]) => `<div><span>${k2}</span><b>${v}</b></div>`).join("");
            }
        };
        return demo;
    }

    // ---------------------------------------------------------------- loop and controls

    const demos = [];
    let last = null;
    function loop(now) {
        const delta = last === null ? 0 : Math.min(now - last, 100);
        last = now;
        demos.forEach(demo => {
            if (!demo.scene || !slideVisible(demo.root)) return;
            if (!state.paused) demo.elapsed += delta;
            if (demo.elapsed > demo.timeline.total) { demo.seed += 1; demo.reset(); }
            const { index, t } = position(demo.timeline, demo.elapsed);
            demo.render(index, t, delta);
        });
        window.requestAnimationFrame(loop);
    }

    function finalFrame(demo) {
        if (!demo.scene) return;
        demo.render(demo.scene.states.length - 1, 0, 0);
    }

    document.addEventListener("DOMContentLoaded", () => {
        [createDepthDemo(), createTrajectoryDemo()].forEach(demo => {
            if (!demo) return;
            if (reducedMotion) demo.onReady = () => finalFrame(demo);
            demo.reset();
            demos.push(demo);
        });
        if (reducedMotion) {
            const show = () => demos.forEach(finalFrame);
            window.setTimeout(show, 50);
            window.addEventListener("resize", show);
            new MutationObserver(show).observe(document.getElementById("benchmark-view-home"), { attributes: true, attributeFilter: ["hidden"] });
        } else {
            window.requestAnimationFrame(loop);
        }
        document.querySelectorAll("[data-align-replay]").forEach(button => button.addEventListener("click", () => {
            const slide = button.closest(".home-slide");
            demos.filter(demo => !slide || slide.contains(demo.root)).forEach(demo => {
                demo.seed += 1;
                demo.reset();
                if (reducedMotion) finalFrame(demo);
            });
        }));
        document.querySelectorAll("[data-align-pause]").forEach(button => button.addEventListener("click", () => {
            state.paused = !state.paused;
            document.querySelectorAll("[data-align-pause]").forEach(b => {
                b.setAttribute("aria-pressed", String(state.paused));
                b.innerHTML = state.paused ? '<i class="fas fa-play" aria-hidden="true"></i> Play'
                    : '<i class="fas fa-pause" aria-hidden="true"></i> Pause';
            });
        }));
        if (reducedMotion) document.querySelectorAll("[data-align-pause]").forEach(b => { b.hidden = true; });
    });
})();
