// Copyright (c) 2026 Yu Chen. All rights reserved.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

// Figure export, as on docs/webpage: every Chart.js chart and every artifact-viewer panel (depth tiles, 3D views,
// pose curves) gets an Export bar (PNG / SVG / PDF), always drawn in the light theme.
//   charts: redrawn as vector SVG/PDF (canvas2svg, svg2pdf); PNGs at 4x.
//   panels: laid-out HTML redrawn as SVG shapes and text, with images and canvas snapshots (WebGL views, 2D canvases)
//           embedded as PNG; PDF via svg2pdf, PNG by rasterizing that SVG at 2x.
// A page calls initChartExport({ redraw, prefix }) once: `redraw` re-renders its charts after a theme change.

(function () {
    const EXPORT_LIBS = {
        c2s: "https://cdn.jsdelivr.net/npm/canvas2svg@1.0.16/canvas2svg.js",
        jspdf: "https://cdn.jsdelivr.net/npm/jspdf@2.5.1/dist/jspdf.umd.min.js",
        svg2pdf: "https://cdn.jsdelivr.net/npm/svg2pdf.js@2.2.3/dist/svg2pdf.umd.min.js",
    };
    const EXPORT_SCALE = 4;
    const HOST_SELECTOR = ".benchmark-radar-panel, .benchmark-bar-panel";
    // Artifact viewer panels: the single-case panel, the depth comparison's tiles and legend, the 3D views of the
    // trajectory / reconstruction / demo modes (with their insets), and the trajectory mode's pose-accuracy chart.
    const PANEL_SELECTOR = ".depth-compare-main, .compare-canvas-wrap, .trajectory-auc";
    const PANEL_SCALE = 2;
    const EXPORT_SKIP_SELECTOR = ".compare-add-button, .compare-layer .compare-icon-button";
    let panelCounter = 0;
    let pageRedraw = () => {};
    let filePrefix = "ffb";

    const libCache = {};
    const loadLib = key => libCache[key] ||= new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = EXPORT_LIBS[key];
        script.onload = resolve;
        script.onerror = () => {
            delete libCache[key];
            reject(new Error(`Could not load ${EXPORT_LIBS[key]}`));
        };
        document.head.appendChild(script);
    });
    const needLibs = async (...keys) => { for (const key of keys) await loadLib(key); };  // svg2pdf needs jspdf first

    // Redraw the page in the light theme for the duration of an export (the dashboard bakes colors in at render).
    async function inLightTheme(fn) {
        const root = document.documentElement;
        const previous = root.getAttribute("data-theme");
        const dark = previous === "dark";
        const settle = () => new Promise(resolve => setTimeout(resolve, 30));
        if (dark) {
            root.setAttribute("data-theme", "light");
            pageRedraw();
            await settle();
        }
        try {
            return await fn();
        } finally {
            if (dark) {
                root.setAttribute("data-theme", previous);
                pageRedraw();
                await settle();
            }
        }
    }

    // Draw a copy of a live chart onto any canvas-like target at its on-screen size.
    function redrawChart(source, canvas, dpr, extraPlugins = []) {
        const config = source.config;
        return new Chart(canvas, {
            type: config.type,
            data: config.data,
            plugins: [...(config.plugins || []), ...extraPlugins],
            platform: Chart.BasicPlatform,
            options: { ...config.options, responsive: false, maintainAspectRatio: false, animation: false, devicePixelRatio: dpr },
        });
    }
    const whiteBackground = {
        id: "export-bg",
        beforeDraw(chart) {
            const context = chart.ctx;
            context.save();
            context.fillStyle = "#ffffff";
            context.fillRect(0, 0, chart.width, chart.height);
            context.restore();
        },
    };

    function chartPng(source) {
        const canvas = document.createElement("canvas");
        canvas.width = Math.round(source.width);
        canvas.height = Math.round(source.height);
        const chart = redrawChart(source, canvas, EXPORT_SCALE, [whiteBackground]);
        return new Promise(resolve => canvas.toBlob(blob => { chart.destroy(); resolve(blob); }, "image/png"));
    }

    class PathRecorder { constructor() { this.ops = []; } }
    for (const method of ["moveTo", "lineTo", "bezierCurveTo", "quadraticCurveTo", "arc", "arcTo", "rect", "ellipse", "closePath"]) {
        PathRecorder.prototype[method] = function (...args) { this.ops.push([method, args]); };
    }

    // A loaded <img> (a model logo on a chart) rasterized at EXPORT_SCALE times its drawn size, so the embedded PNG
    // stays sharp when the figure is enlarged; null while it has not loaded.
    const embeddedImages = new Map();
    function embeddedImage(image) {
        if (!image.complete || !image.naturalWidth) return null;
        const key = `${image.currentSrc || image.src}@${image.width}x${image.height}`;
        if (!embeddedImages.has(key)) {
            const canvas = document.createElement("canvas");
            canvas.width = Math.round(image.width * EXPORT_SCALE);
            canvas.height = Math.round(image.height * EXPORT_SCALE);
            canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
            embeddedImages.set(key, canvas);
        }
        return embeddedImages.get(key);
    }

    function chartSvg(source) {
        const width = Math.round(source.width);
        const height = Math.round(source.height);
        const context = new C2S(width, height);
        // canvas2svg lacks these; dashes are copied onto each stroked element.
        let dash = [];
        context.setLineDash = value => { dash = value || []; };
        context.getLineDash = () => dash;
        context.resetTransform = () => {};
        // Its own font parser rejects quoted family names such as 'Inter'.
        context.__parseFont = function () {
            const match = /^\s*((?:(?:italic|oblique|normal|small-caps|bold(?:er)?|lighter|[1-9]00)\s+)*)([.\d]+(?:px|pt|em|rem|%))(?:\s*\/\s*\S+)?\s+(.+)$/i
                .exec(this.font) || [];
            const prefix = (match[1] || "").toLowerCase();
            return {
                style: (/italic|oblique/.exec(prefix) || ["normal"])[0], size: match[2] || "10px", family: match[3] || "sans-serif",
                weight: (/bold(?:er)?|lighter|[1-9]00/.exec(prefix) || ["normal"])[0], decoration: "normal", href: null,
            };
        };
        const replay = (target, path) => {
            if (path instanceof PathRecorder) {
                target.beginPath();
                for (const [method, args] of path.ops) target[method](...args);
            }
        };
        const { stroke, fill } = context;
        context.stroke = function (path) {
            replay(this, path);
            stroke.call(this);
            if (dash.length && this.__currentElement) this.__currentElement.setAttribute("stroke-dasharray", dash.join(" "));
        };
        context.fill = function (path) { replay(this, path); fill.call(this); };
        // canvas2svg links an <img> by its src ("assets/logos/models/qwen.svg"), which a saved file cannot resolve:
        // draw it from a PNG copy instead, which canvas2svg embeds as a data URL.
        const { drawImage } = context;
        context.drawImage = function (image, ...args) {
            if (image?.nodeName !== "IMG" || !image.width || !image.height) return drawImage.call(this, image, ...args);
            const png = embeddedImage(image);
            if (!png) return undefined;
            const k = png.width / image.width;
            if (args.length === 2) args = [...args, image.width, image.height];
            else if (args.length === 8) args = [...args.slice(0, 4).map(value => value * k), ...args.slice(4)];
            return drawImage.call(this, png, ...args);
        };
        // canvas2svg's clip paths leak into later drawing; charts stay inside their axes without them.
        context.clip = function () { this.beginPath(); };
        const canvas = { width, height, style: {}, getContext: () => context };
        // Chart.js strokes lines through Path2D, which canvas2svg cannot read: record those paths and replay them.
        const NativePath2D = window.Path2D;
        window.Path2D = PathRecorder;
        let chart;
        try {
            chart = redrawChart(source, canvas, 1, [whiteBackground]);
        } finally {
            window.Path2D = NativePath2D;
        }
        const svg = context.getSvg().cloneNode(true);
        svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
        svg.setAttribute("xmlns", "http://www.w3.org/2000/svg");
        convertHslColors(svg);
        chart.destroy();  // clears the recorded drawing, hence the clone above
        return { svg, width, height };
    }

    // hsl()/hsla() colors (the per-model colors) as rgb plus an opacity attribute: svg2pdf and older vector editors
    // do not read CSS hsl.
    function hslToRgb(hue, saturation, lightness) {
        const s = saturation / 100;
        const l = lightness / 100;
        const k = n => (n + hue / 30) % 12;
        const a = s * Math.min(l, 1 - l);
        const f = n => Math.round(255 * (l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)))));
        return `rgb(${f(0)}, ${f(8)}, ${f(4)})`;
    }

    function convertHslColors(svg) {
        // "transparent" (and fully transparent rgba) is not an SVG 1.1 paint: several viewers draw it black.
        const clear = /^\s*(transparent|rgba\(\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*,\s*0(\.0*)?\s*\))\s*$/i;
        for (const element of svg.querySelectorAll("[fill], [stroke]")) {
            for (const attribute of ["fill", "stroke"]) {
                if (clear.test(element.getAttribute(attribute) || "")) {
                    element.setAttribute(attribute, "none");
                    element.removeAttribute(`${attribute}-opacity`);
                }
            }
        }
        const pattern = /^\s*hsla?\(\s*([\d.]+)(?:deg)?[\s,]+([\d.]+)%[\s,]+([\d.]+)%(?:[\s,/]+([\d.]+%?))?\s*\)\s*$/i;
        for (const element of svg.querySelectorAll("[fill], [stroke]")) {
            for (const attribute of ["fill", "stroke"]) {
                const match = pattern.exec(element.getAttribute(attribute) || "");
                if (!match) continue;
                element.setAttribute(attribute, hslToRgb(Number(match[1]), Number(match[2]), Number(match[3])));
                if (match[4] !== undefined) {
                    const alpha = match[4].endsWith("%") ? Number(match[4].slice(0, -1)) / 100 : Number(match[4]);
                    const previous = Number(element.getAttribute(`${attribute}-opacity`) || 1);
                    element.setAttribute(`${attribute}-opacity`, String(+(alpha * previous).toFixed(3)));
                }
            }
        }
    }

    async function svgToPdf(svg, width, height) {
        const doc = new window.jspdf.jsPDF({ unit: "pt", format: [width, height], orientation: width > height ? "landscape" : "portrait" });
        const host = document.createElement("div");
        host.style.cssText = "position:fixed;left:-99999px;top:0";
        // jsPDF's standard fonts know only normal and bold, and not the page's web fonts.
        // Their WinAnsi encoding covers Latin-1 plus a few typographic marks: Greek letters (VGGT-\u03a9 would print as
        // VGGT-\u00a9) and math symbols are spelled out, anything else outside it becomes "?".
        const spelled = {
            "\u03a9": "Omega", "\u03c0": "pi", "\u03bc": "mu", "\u03b1": "alpha", "\u03b2": "beta", "\u03b4": "delta",
            "\u03b8": "theta", "\u03c3": "sigma", "\u2265": ">=", "\u2264": "<=", "\u2248": "~", "\u2260": "!=",
            "\u2212": "-", "\u2192": "->", "\u2190": "<-", "\u221e": "inf", "\u2026": "...",
        };
        const winAnsiExtras = "\u20ac\u201a\u0192\u201e\u2026\u2020\u2021\u02c6\u2030\u0160\u2039\u0152\u017d\u2018\u2019"
            + "\u201c\u201d\u2022\u2013\u2014\u02dc\u2122\u0161\u203a\u0153\u017e\u0178";
        for (const text of svg.querySelectorAll("text")) {
            text.textContent = text.textContent.replace(/[^\u0000-\u00ff]/g,
                char => spelled[char] ?? (winAnsiExtras.includes(char) ? char : "?"));
            text.setAttribute("font-family", "helvetica");
            text.setAttribute("font-weight", /bold|[6-9]00/.test(text.getAttribute("font-weight") || "") ? "bold" : "normal");
        }
        host.appendChild(svg);
        document.body.appendChild(host);
        try {
            await window.svg2pdf.svg2pdf(svg, doc, { x: 0, y: 0, width, height });
        } finally {
            host.remove();
        }
        return doc.output("blob");
    }

    // Export the chart drawn on the canvas with this id (looked up again after the light-theme redraw, which may
    // replace the canvas).
    async function exportChart(canvasId, name, format) {
        if (format !== "png") await needLibs("c2s", ...(format === "pdf" ? ["jspdf", "svg2pdf"] : []));
        return inLightTheme(async () => {
            const canvas = document.getElementById(canvasId);
            const source = canvas && Chart.getChart(canvas);
            if (!source || !source.width || !source.height) throw new Error("This chart is not drawn right now");
            if (format === "png") return [`${name}.png`, await chartPng(source)];
            const { svg, width, height } = chartSvg(source);
            if (format === "svg") return [`${name}.svg`, new Blob([new XMLSerializer().serializeToString(svg)], { type: "image/svg+xml" })];
            return [`${name}.pdf`, await svgToPdf(svg, width, height)];
        });
    }

    // ---- artifact panels ----------------------------------------------------------------------------------------

    // Light-theme CSS for the duration of a panel export, without re-rendering (3D views keep their camera).
    async function inLightCss(fn) {
        const root = document.documentElement;
        const previous = root.getAttribute("data-theme");
        const frame = () => new Promise(resolve => requestAnimationFrame(() => resolve()));
        if (previous === "dark") {
            root.setAttribute("data-theme", "light");
            await frame();
            await frame();
        }
        try {
            return await fn();
        } finally {
            if (previous === "dark") root.setAttribute("data-theme", previous);
        }
    }

    const imageCache = new Map();
    // An <img> as a PNG data URL at its natural size (WebP tiles included: jsPDF reads only PNG/JPEG).
    function imagePng(img) {
        const key = img.currentSrc || img.src;
        if (imageCache.has(key)) return imageCache.get(key);
        const canvas = document.createElement("canvas");
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        canvas.getContext("2d").drawImage(img, 0, 0);
        const url = canvas.toDataURL("image/png");
        if (imageCache.size > 200) imageCache.clear();
        imageCache.set(key, url);
        return url;
    }

    // The visible part of an image drawn at ``box`` (page px) under the clip rect ``clip``, as a plain image:
    // {left, top, width, height (page px), href}. Cropped into the PNG itself rather than an SVG clip-path, which
    // PowerPoint's (and other editors') SVG import ignores or drops. ``fullHref`` gives the uncropped PNG.
    function croppedImage(source, sourceWidth, sourceHeight, box, clip, fullHref) {
        const left = Math.max(box.left, clip.left), top = Math.max(box.top, clip.top);
        const right = Math.min(box.left + box.width, clip.right), bottom = Math.min(box.top + box.height, clip.bottom);
        if (right - left < 0.5 || bottom - top < 0.5) return null;
        try {
            if (left <= box.left + 0.5 && top <= box.top + 0.5 && right >= box.left + box.width - 0.5
                && bottom >= box.top + box.height - 0.5) {
                return { left: box.left, top: box.top, width: box.width, height: box.height, href: fullHref() };
            }
            const sx = sourceWidth / box.width, sy = sourceHeight / box.height;
            const crop = { x: (left - box.left) * sx, y: (top - box.top) * sy, w: (right - left) * sx, h: (bottom - top) * sy };
            // At least the panel export's resolution, at most the source's own pixels.
            const scale = Math.max(1, PANEL_SCALE / Math.max(sx, sy));
            const canvas = document.createElement("canvas");
            canvas.width = Math.max(1, Math.round(crop.w * scale));
            canvas.height = Math.max(1, Math.round(crop.h * scale));
            canvas.getContext("2d").drawImage(source, crop.x, crop.y, crop.w, crop.h, 0, 0, canvas.width, canvas.height);
            return { left, top, width: right - left, height: bottom - top, href: canvas.toDataURL("image/png") };
        } catch (_error) {
            return null;
        }
    }

    // The url() of a computed background-image, resolved against the page (null when it is not a single url()).
    function backgroundUrl(style) {
        const match = /^url\((['"]?)(.*)\1\)$/.exec((style.backgroundImage || "").trim());
        if (!match) return null;
        try { return new URL(match[2].replace(/\\(.)/g, "$1"), document.baseURI).href; } catch (_error) { return null; }
    }

    // Load every background-image url() in the panel (the depth magnifier's patches) before it is drawn, so
    // backgroundImagePatch can rasterize them synchronously: url -> loaded <img> (failures are left out).
    async function loadBackgroundImages(node) {
        const urls = new Set();
        for (const element of [node, ...node.querySelectorAll("*")]) {
            if (element.closest("[hidden]")) continue;
            const url = backgroundUrl(getComputedStyle(element));
            if (url) urls.add(url);
        }
        const loaded = new Map();
        await Promise.all([...urls].map(async url => {
            const image = new Image();
            image.crossOrigin = "anonymous";
            image.src = url;
            try { await image.decode(); loaded.set(url, image); } catch (_error) { /* drawn as its background color */ }
        }));
        return loaded;
    }

    // A background-image url() drawn at a px background-size/-position (the depth magnifier's zoomed patch), cropped
    // and scaled here into a PNG of the element's padding box: {href, width, height}. A plain <image> of the visible
    // patch, not the whole image offset and clipped -- many SVG viewers and editors drop or misplace that.
    function backgroundImagePatch(element, style, images, rect) {
        const url = backgroundUrl(style);
        const img = url && images?.get(url);
        if (!img || !img.naturalWidth || !style.backgroundSize.includes("px")) return null;
        const [width, height] = style.backgroundSize.split(/\s+/).map(parseFloat);
        const [left, top] = style.backgroundPosition.split(/\s+/).map(parseFloat);
        if (![width, height, left, top].every(Number.isFinite) || width <= 0 || height <= 0) return null;
        const border = side => parseFloat(style[`border${side}Width`]) || 0;
        const boxWidth = rect.width - border("Left") - border("Right");
        const boxHeight = rect.height - border("Top") - border("Bottom");
        if (boxWidth <= 0 || boxHeight <= 0) return null;
        // The part of the drawn image inside the box, in box pixels and in image pixels.
        const x0 = Math.max(0, left), y0 = Math.max(0, top);
        const x1 = Math.min(boxWidth, left + width), y1 = Math.min(boxHeight, top + height);
        const scale = PANEL_SCALE;
        const canvas = document.createElement("canvas");
        canvas.width = Math.round(boxWidth * scale);
        canvas.height = Math.round(boxHeight * scale);
        const context = canvas.getContext("2d");
        context.imageSmoothingEnabled = style.imageRendering !== "pixelated";
        const radius = Math.max(0, (parseFloat(style.borderTopLeftRadius) || 0) - border("Left")) * scale;
        if (radius && context.roundRect) {
            context.beginPath();
            context.roundRect(0, 0, canvas.width, canvas.height, radius);
            context.clip();
        }
        if (x1 > x0 && y1 > y0) {
            const sx = img.naturalWidth / width, sy = img.naturalHeight / height;
            context.drawImage(img, (x0 - left) * sx, (y0 - top) * sy, (x1 - x0) * sx, (y1 - y0) * sy,
                x0 * scale, y0 * scale, (x1 - x0) * scale, (y1 - y0) * scale);
        }
        let href = "";
        try { href = canvas.toDataURL("image/png"); } catch (_error) { return null; }
        return { href, width: boxWidth, height: boxHeight };
    }

    // The CSS clip-path inset() of an element, as a rect in page pixels (null when none).
    function insetClip(element, rect) {
        const match = /inset\(([^)]+)\)/.exec(getComputedStyle(element).clipPath || "");
        if (!match) return null;
        const values = match[1].trim().split(/\s+/).slice(0, 4);
        while (values.length < 4) values.push(values[values.length === 3 ? 1 : values.length === 2 ? 0 : 0]);
        const length = (value, size) => (value.endsWith("%") ? parseFloat(value) / 100 * size : parseFloat(value) || 0);
        const [top, right, bottom, left] = values;
        return {
            left: rect.left + length(left, rect.width), right: rect.right - length(right, rect.width),
            top: rect.top + length(top, rect.height), bottom: rect.bottom - length(bottom, rect.height),
        };
    }

    // Where an element is visible: its own rect cut by its clip-path and by every clipping ancestor inside the panel.
    function visibleRect(element, node) {
        const rect = element.getBoundingClientRect();
        let clip = { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom };
        const cut = other => {
            if (!other) return;
            clip = {
                left: Math.max(clip.left, other.left), right: Math.min(clip.right, other.right),
                top: Math.max(clip.top, other.top), bottom: Math.min(clip.bottom, other.bottom),
            };
        };
        cut(insetClip(element, rect));
        for (let ancestor = element.parentElement; ancestor && node.contains(ancestor); ancestor = ancestor.parentElement) {
            const style = getComputedStyle(ancestor);
            if (style.overflowX !== "visible" || style.overflowY !== "visible") cut(ancestor.getBoundingClientRect());
            cut(insetClip(ancestor, ancestor.getBoundingClientRect()));
            if (ancestor === node) break;
        }
        return { rect, clip };
    }

    // The drawn area of a replaced element (img, canvas) under object-fit.
    function contentBox(element, rect, naturalWidth, naturalHeight) {
        const fit = getComputedStyle(element).objectFit;
        if (!naturalWidth || !naturalHeight || (fit !== "contain" && fit !== "cover" && fit !== "scale-down")) return rect;
        const scaleX = rect.width / naturalWidth;
        const scaleY = rect.height / naturalHeight;
        let scale = fit === "cover" ? Math.max(scaleX, scaleY) : Math.min(scaleX, scaleY);
        if (fit === "scale-down") scale = Math.min(1, scale);
        const width = naturalWidth * scale;
        const height = naturalHeight * scale;
        return { left: rect.left + (rect.width - width) / 2, top: rect.top + (rect.height - height) / 2, width, height };
    }

    const SVG_STYLE_PROPERTIES = ["fill", "fill-opacity", "stroke", "stroke-width", "stroke-opacity", "stroke-dasharray",
        "stroke-linecap", "stroke-linejoin", "opacity", "font-size", "font-family", "font-weight", "text-anchor",
        "dominant-baseline", "visibility", "display"];

    // An inline <svg> styled by the page's CSS, with its computed styles written onto the clone.
    function inlineSvgCopy(source) {
        const copy = source.cloneNode(true);
        const sources = [source, ...source.querySelectorAll("*")];
        const copies = [copy, ...copy.querySelectorAll("*")];
        sources.forEach((element, index) => {
            const style = getComputedStyle(element);
            for (const property of SVG_STYLE_PROPERTIES) {
                const value = style.getPropertyValue(property);
                if (value && value !== "normal" && value !== "auto") copies[index].setAttribute(property, value);
            }
        });
        return copy;
    }

    // A CSS linear-gradient() (color bars) as an SVG gradient id, horizontal or vertical; null if not parseable.
    function gradientFill(svg, add, defs, backgroundImage) {
        const match = /linear-gradient\((.*)\)\s*$/.exec(backgroundImage || "");
        if (!match) return null;
        const parts = match[1].split(/,(?![^(]*\))/).map(part => part.trim());
        let vertical = false;
        if (/^(to |[-\d.]+deg)/.test(parts[0])) {
            const direction = parts.shift();
            vertical = /to (top|bottom)$/.test(direction) || /^(0|180|360)deg$/.test(direction);
        }
        const stops = parts.map(part => /^(rgba?\([^)]+\)|#[0-9a-f]{3,8}|[a-z]+)(?:\s+([\d.]+)%)?/i.exec(part)).filter(Boolean);
        if (stops.length < 2) return null;
        const id = `export-gradient-${svg.childElementCount}`;
        const gradient = add("linearGradient", { id, x1: 0, y1: vertical ? 1 : 0, x2: vertical ? 0 : 1, y2: 0 }, null, defs);
        stops.forEach((stop, index) => {
            const offset = stop[2] !== undefined ? Number(stop[2]) / 100 : index / (stops.length - 1);
            const element = document.createElementNS("http://www.w3.org/2000/svg", "stop");
            element.setAttribute("offset", offset);
            element.setAttribute("stop-color", stop[1]);
            gradient.appendChild(element);
        });
        return id;
    }

    // Redraw a laid-out panel as plain SVG: boxes, borders, text, color bars, and images / canvases as embedded PNGs.
    function panelSvg(node, backgrounds = null) {
        const NS = "http://www.w3.org/2000/svg";
        const origin = node.getBoundingClientRect();
        const W = Math.ceil(origin.width);
        const H = Math.ceil(origin.height);
        const svg = document.createElementNS(NS, "svg");
        svg.setAttribute("xmlns", NS);
        svg.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
        svg.setAttribute("width", W);
        svg.setAttribute("height", H);
        svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
        const add = (tag, attrs, text, parent = svg) => {
            const element = document.createElementNS(NS, tag);
            for (const key in attrs) if (attrs[key] != null) element.setAttribute(key, attrs[key]);
            if (text != null) element.textContent = text;
            parent.appendChild(element);
            return element;
        };
        const defs = add("defs", {});
        add("rect", { x: 0, y: 0, width: W, height: H, fill: "#ffffff" });
        const rgba = color => {
            const match = /rgba?\(([^)]+)\)/.exec(color || "");
            if (!match) return null;
            const p = match[1].split(/[ ,/]+/).filter(Boolean).map(Number);
            const alpha = p.length > 3 ? p[3] : 1;
            return alpha > 0.01 ? { rgb: `rgb(${p[0]},${p[1]},${p[2]})`, a: alpha } : null;
        };
        // Controls stay out of the figure: the export bar, and a layer list's add / remove buttons.
        const skip = element => element.closest(`.chart-export-bar, ${EXPORT_SKIP_SELECTOR}`)
            || (element.closest("svg") && element.tagName.toLowerCase() !== "svg");
        const opacityOf = element => {
            let alpha = 1;
            for (let e = element; e && e !== node.parentElement; e = e.parentElement) alpha *= +getComputedStyle(e).opacity;
            return alpha;
        };
        const fontStack = family => `${family.split(",")[0].replace(/['"]/g, "").trim()}, Arial, Helvetica, sans-serif`;

        for (const element of [node, ...node.querySelectorAll("*")]) {
            if (skip(element)) continue;
            const style = getComputedStyle(element);
            if (style.display === "none" || style.visibility === "hidden" || element.closest("[hidden]")) continue;
            const rect = element.getBoundingClientRect();
            if (!rect.width || !rect.height) continue;
            const x = rect.left - origin.left;
            const y = rect.top - origin.top;
            const opacity = opacityOf(element);
            const radius = Math.min(parseFloat(style.borderTopLeftRadius) || 0, rect.width / 2, rect.height / 2);
            const tag = element.tagName.toLowerCase();
            if (element !== node) {
                const background = rgba(style.backgroundColor);
                if (background) {
                    add("rect", { x, y, width: rect.width, height: rect.height, rx: radius || null, fill: background.rgb,
                        "fill-opacity": +(background.a * opacity).toFixed(3) });
                }
                const gradient = gradientFill(svg, add, defs, style.backgroundImage);
                if (gradient) add("rect", { x, y, width: rect.width, height: rect.height, rx: radius || null, fill: `url(#${gradient})` });
                const patch = backgroundImagePatch(element, style, backgrounds, rect);
                if (patch) {
                    add("image", { x: x + (parseFloat(style.borderLeftWidth) || 0), y: y + (parseFloat(style.borderTopWidth) || 0),
                        width: patch.width, height: patch.height, preserveAspectRatio: "none",
                        href: patch.href, "xlink:href": patch.href, opacity: opacity < 1 ? opacity : null });
                }
            }
            if (tag === "img" && element.complete && element.naturalWidth) {
                const box = contentBox(element, rect, element.naturalWidth, element.naturalHeight);
                const part = croppedImage(element, element.naturalWidth, element.naturalHeight, box,
                    visibleRect(element, node).clip, () => imagePng(element));
                if (part) {
                    add("image", { x: part.left - origin.left, y: part.top - origin.top, width: part.width, height: part.height,
                        preserveAspectRatio: "none", opacity: opacity < 1 ? opacity : null, href: part.href, "xlink:href": part.href });
                }
            } else if (tag === "canvas" && element.width && element.height) {
                const box = { left: rect.left, top: rect.top, width: rect.width, height: rect.height };
                const part = croppedImage(element, element.width, element.height, box, visibleRect(element, node).clip,
                    () => element.toDataURL("image/png"));
                if (part) {
                    add("image", { x: part.left - origin.left, y: part.top - origin.top, width: part.width, height: part.height,
                        preserveAspectRatio: "none", href: part.href, "xlink:href": part.href });
                }
            } else if (tag === "svg") {
                const copy = inlineSvgCopy(element);
                copy.setAttribute("x", x);
                copy.setAttribute("y", y);
                copy.setAttribute("width", rect.width);
                copy.setAttribute("height", rect.height);
                svg.appendChild(copy);
                continue;
            }
            const sides = [["Top", x, y, x + rect.width, y], ["Right", x + rect.width, y, x + rect.width, y + rect.height],
                ["Bottom", x, y + rect.height, x + rect.width, y + rect.height], ["Left", x, y, x, y + rect.height]]
                .map(([key, ...points]) => ({ width: parseFloat(style[`border${key}Width`]) || 0,
                    color: style[`border${key}Style`] !== "none" ? rgba(style[`border${key}Color`]) : null, points }));
            if (element !== node && sides.every(b => b.width && b.color && b.width === sides[0].width && b.color.rgb === sides[0].color.rgb)) {
                const w = sides[0].width;
                add("rect", { x: x + w / 2, y: y + w / 2, width: rect.width - w, height: rect.height - w,
                    rx: radius ? Math.max(0, radius - w / 2) : null, fill: "none", stroke: sides[0].color.rgb,
                    "stroke-opacity": +(sides[0].color.a * opacity).toFixed(3), "stroke-width": w });
            } else if (element !== node) {
                for (const side of sides) {
                    if (!side.width || !side.color) continue;
                    const [x1, y1, x2, y2] = side.points;
                    add("line", { x1, y1, x2, y2, stroke: side.color.rgb, "stroke-opacity": +(side.color.a * opacity).toFixed(3),
                        "stroke-width": side.width });
                }
            }
        }

        // Text, regrouped by rendered line (form controls show their value instead).
        const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
        for (let textNode = walker.nextNode(); textNode; textNode = walker.nextNode()) {
            const element = textNode.parentElement;
            if (!element || skip(element) || element.closest("select, option, script, style") || !textNode.data.trim()) continue;
            const style = getComputedStyle(element);
            if (style.display === "none" || style.visibility === "hidden" || element.closest("[hidden]")) continue;
            const color = rgba(style.color);
            if (!color) continue;
            const fontSize = parseFloat(style.fontSize);
            const textStyle = { "font-family": fontStack(style.fontFamily), "font-size": fontSize, "font-weight": style.fontWeight,
                fill: color.rgb, "fill-opacity": +(color.a * opacityOf(element)).toFixed(3),
                "font-style": style.fontStyle !== "normal" ? style.fontStyle : null };
            const transform = text => (style.textTransform === "uppercase" ? text.toUpperCase() : text);
            const lines = [];
            const words = /\S+/g;
            let match;
            while ((match = words.exec(textNode.data))) {
                const range = document.createRange();
                range.setStart(textNode, match.index);
                range.setEnd(textNode, match.index + match[0].length);
                const box = range.getClientRects()[0];
                if (!box) continue;
                const last = lines[lines.length - 1];
                if (last && Math.abs(last.top - box.top) < fontSize * 0.5) {
                    last.words.push(match[0]);
                } else {
                    lines.push({ top: box.top, bottom: box.bottom, left: box.left, words: [match[0]] });
                }
            }
            for (const line of lines) {
                add("text", { ...textStyle, x: line.left - origin.left, y: (line.top + line.bottom) / 2 - origin.top + 0.35 * fontSize },
                    transform(line.words.join(" ")));
            }
        }
        return { svg, width: W, height: H };
    }

    async function svgPng(svg, width, height) {
        const xml = new XMLSerializer().serializeToString(svg);
        const url = URL.createObjectURL(new Blob([xml], { type: "image/svg+xml" }));
        try {
            const image = new Image();
            await new Promise((resolve, reject) => {
                image.onload = resolve;
                image.onerror = () => reject(new Error("Could not rasterize the panel"));
                image.src = url;
            });
            const canvas = document.createElement("canvas");
            canvas.width = Math.round(width * PANEL_SCALE);
            canvas.height = Math.round(height * PANEL_SCALE);
            const context = canvas.getContext("2d");
            context.fillStyle = "#ffffff";
            context.fillRect(0, 0, canvas.width, canvas.height);
            context.drawImage(image, 0, 0, canvas.width, canvas.height);
            return await new Promise(resolve => canvas.toBlob(resolve, "image/png"));
        } finally {
            URL.revokeObjectURL(url);
        }
    }

    async function exportPanel(node, name, format) {
        if (format === "pdf") await needLibs("jspdf", "svg2pdf");
        return inLightCss(async () => {
            const { svg, width, height } = panelSvg(node, await loadBackgroundImages(node));
            if (format === "svg") return [`${name}.svg`, new Blob([new XMLSerializer().serializeToString(svg)], { type: "image/svg+xml" })];
            if (format === "png") return [`${name}.png`, await svgPng(svg, width, height)];
            convertHslColors(svg);
            return [`${name}.pdf`, await svgToPdf(svg, width, height)];
        });
    }

    // "<prefix>-<mode>-<dataset>-<scene>" from the comparison state in the URL.
    function panelName(node) {
        const [, query = ""] = window.location.hash.split("?");
        const state = new URLSearchParams(query);
        const parts = [state.get("compare") || "artifact", node.classList.contains("trajectory-auc") ? "pose-accuracy" : "",
            state.get("dataset") || state.get("dataset_key"), state.get("scene")];
        return [filePrefix, ...parts.filter(Boolean).map(slug).filter(Boolean)].join("-");
    }

    function attachPanelBars() {
        for (const node of document.querySelectorAll(PANEL_SELECTOR)) {
            const existing = node.querySelector(":scope > .chart-export-bar");
            const empty = !node.querySelector("img, canvas, svg");
            if (existing) {
                existing.hidden = empty;
                continue;
            }
            if (empty) continue;
            node.dataset.exportPanel ||= String(++panelCounter);
            const bar = document.createElement("div");
            bar.className = "chart-export-bar chart-export-bar-panel";
            bar.dataset.panel = node.dataset.exportPanel;
            bar.innerHTML = "<span>Export</span>"
                + ["png", "svg", "pdf"].map(format => `<button type="button" data-format="${format}">${format.toUpperCase()}</button>`).join("");
            node.classList.add("chart-export-host");
            node.prepend(bar);
        }
    }

    function saveFile(filename, blob) {
        const link = document.createElement("a");
        link.download = filename;
        link.href = URL.createObjectURL(blob);
        document.body.appendChild(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(link.href), 4000);
    }

    const slug = text => String(text || "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
    // "<prefix>-<card heading>-<panel title>", e.g. ffb-top-10-scoreboards-monodepth.
    function exportName(canvas) {
        const card = canvas.closest(".card")?.querySelector(".card-header h2")?.textContent;
        // A page without the dashboard's panel titles names its charts by canvas id instead.
        const panel = canvas.closest(".benchmark-radar-panel, .benchmark-bar-panel")
            ?.querySelector(".benchmark-radar-title")?.textContent || canvas.id;
        const parts = [card, panel].filter(Boolean).map(slug).filter(Boolean);
        return [filePrefix, ...(parts.length ? [...new Set(parts)] : [slug(canvas.id)])].join("-");
    }

    // Give every Chart.js chart's panel an Export bar (panels re-rendered by the page get theirs again); the bar
    // hides while its chart is not drawn (e.g. a radar with too few axes).
    function attachExportBars() {
        for (const canvas of document.querySelectorAll("canvas[id]")) {
            const host = canvas.closest(HOST_SELECTOR) || canvas.parentElement;
            if (!host) continue;
            const existing = host.querySelector(":scope > .chart-export-bar");
            const drawn = Boolean(Chart.getChart(canvas)) && !canvas.hidden;
            if (existing) {
                existing.hidden = !drawn;
                continue;
            }
            if (!drawn) continue;
            const bar = document.createElement("div");
            bar.className = "chart-export-bar";
            bar.dataset.canvas = canvas.id;
            bar.innerHTML = "<span>Export</span>"
                + ["png", "svg", "pdf"].map(format => `<button type="button" data-format="${format}">${format.toUpperCase()}</button>`).join("");
            host.classList.add("chart-export-host");
            host.prepend(bar);
        }
    }

    async function handleExportClick(event) {
        const button = event.target.closest(".chart-export-bar button");
        if (!button) return;
        const bar = button.closest(".chart-export-bar");
        const panel = bar.dataset.panel ? document.querySelector(`[data-export-panel="${bar.dataset.panel}"]`) : null;
        // A comparison view exports with its layer list (the legend of methods and colors) beside it.
        const stage = panel?.matches(".compare-canvas-wrap") ? panel.closest(".compare-stage") : null;
        const exportNode = stage?.querySelector(":scope > .compare-layers:not([hidden])") ? stage : panel;
        const canvas = panel ? null : document.getElementById(bar.dataset.canvas);
        if (!panel && !canvas) return;
        const label = button.textContent;
        button.disabled = true;
        button.textContent = "…";
        try {
            const [file, blob] = panel
                ? await exportPanel(exportNode, panelName(panel), button.dataset.format)
                : await exportChart(canvas.id, exportName(canvas), button.dataset.format);
            saveFile(file, blob);
            button.textContent = "✓";
        } catch (error) {
            console.error(error);
            button.textContent = "!";
            button.title = String(error.message || error);
        }
        // The light-theme redraw may have rebuilt this panel; restore whichever button is current.
        setTimeout(() => {
            const selector = panel ? `[data-panel="${bar.dataset.panel}"]` : `[data-canvas="${canvas.id}"]`;
            const current = document.querySelector(`.chart-export-bar${selector} button[data-format="${button.dataset.format}"]`) || button;
            current.textContent = label;
            current.disabled = false;
        }, 1400);
    }

    window.initChartExport = function ({ redraw, prefix } = {}) {
        if (typeof redraw === "function") pageRedraw = redraw;
        if (prefix) filePrefix = prefix;
        document.addEventListener("click", handleExportClick);
        attachExportBars();
        attachPanelBars();
        let pending = false;
        new MutationObserver(() => {
            if (pending) return;
            pending = true;
            requestAnimationFrame(() => { pending = false; attachExportBars(); attachPanelBars(); });
        }).observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["hidden"] });
    };
}());
