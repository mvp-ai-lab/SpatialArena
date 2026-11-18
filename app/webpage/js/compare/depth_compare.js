// Copyright (c) 2026 Yu Chen. All rights reserved.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

// Depth comparison (MonoDepth / VideoDepth). Frames come from the released depth assets (artifacts/): every
// method's aligned depth on the scene's GT color scale, plus its error map. Two views:
// - Swipe: two layers stacked in one block; the cursor moves the divider between them.
// - Grid (the default): GT and every loaded method side by side (three per row), with zoom, pan and a crosshair
//   shared by all tiles. A magnifier (on by default) lays a zoomed-in patch over every tile, at the cursor.
// The RGB input is a reference thumbnail next to the viewer layers (it marks where the cursor is), not a tile.
// Frames play back like a video (VideoDepth) or step as a slideshow (MonoDepth).

import {
    ASSET_ROOT, INDEX_ROOT, backgroundControlHtml, bindBackgroundControl, datasetLabel, escapeHtml, fetchAssetJson,
    formatNumber, loadBackground, methodLabel, moveItem, bindCtrlDragReorder, openAddDialog,
    renderLayerList, taskLabel, writeCompareState,
} from './compare_common.js';

const FPS_OPTIONS = [2, 5, 10, 15, 24, 30];
const PRELOAD_AHEAD = 12;
const MAX_ZOOM = 16;
const GRID_COLUMNS = 3;
// Magnifier patch side (px) and magnification relative to the tile as displayed.
const MAGNIFIER_SIZES = [100, 140, 180, 240, 320];
const MAGNIFIER_ZOOMS = [2, 3, 4, 6, 8];

function colormapGradient(name) {
    // CSS approximations of the matplotlib colormaps the builder uses (Spectral: near red -> far blue; turbo).
    if (name === 'turbo') {
        return 'linear-gradient(90deg,#30123b,#4662d7,#36aaf9,#1ae4b6,#72fe5e,#c8ef34,#faba39,#f66b19,#ca2a04,#7a0403)';
    }
    return 'linear-gradient(90deg,#9e0142,#d53e4f,#f46d43,#fdae61,#fee08b,#ffffbf,#e6f598,#abdda4,#66c2a5,#3288bd,#5e4fa2)';
}

function optionsHtml(values, selected, label) {
    return values.map(value => `<option value="${value}"${value === selected ? ' selected' : ''}>${label(value)}</option>`).join('');
}

export class DepthCompare {
    constructor(root) {
        this.root = root;
        this.scene = null;  // index entry
        this.manifest = null;  // scene.json
        this.methods = [];  // loaded method names, in display order (the GT tile always comes first)
        this.colors = new Map();  // method -> its swatch color, kept when the method is moved
        this.hidden = new Set();
        this.frame = 0;
        this.view = 'grid';
        this.mapKind = 'depth';  // what method layers show: depth or error
        this.left = 'gt';
        this.right = null;
        this.fps = 10;
        this.zoom = 1;
        this.pan = { x: 0, y: 0 };
        this.swipe = 0.5;
        this.magnifier = { enabled: true, size: 180, zoom: 3 };
        this.pointer = null;  // last cursor position over a tile, as fractions of the tile
        this.pinned = false;  // right-click pins the magnifier/crosshair: it stays put when the cursor moves away
        this.imageCache = new Map();
        this.background = loadBackground('depth');
        this.buildDom();
        this.root.style.setProperty('--depth-compare-bg', this.background);
        this.render();
    }

    buildDom() {
        this.root.innerHTML = `
            <div class="compare-toolbar">
                <div class="compare-segmented" data-role="view">
                    <button type="button" data-view="swipe">Swipe</button>
                    <button type="button" data-view="grid" class="active">Grid</button>
                </div>
                <div class="compare-segmented" data-role="map">
                    <button type="button" data-map="depth" class="active">Depth</button>
                    <button type="button" data-map="error">Error</button>
                </div>
                <label class="compare-inline" data-role="left-label">Left <select data-role="left"></select></label>
                <label class="compare-inline" data-role="right-label">Right <select data-role="right"></select></label>
                <span class="compare-inline" data-role="magnifier-tools">
                    <label class="compare-inline" title="Show a zoomed-in patch at the cursor on every tile; right-click a tile to pin it there (right-click again to release)">
                        <input type="checkbox" data-role="magnifier"> Magnifier
                    </label>
                    <label class="compare-inline">Patch
                        <select data-role="magnifier-size">${optionsHtml(MAGNIFIER_SIZES, this.magnifier.size, size => `${size}px`)}</select>
                    </label>
                    <label class="compare-inline">Zoom
                        <select data-role="magnifier-zoom">${optionsHtml(MAGNIFIER_ZOOMS, this.magnifier.zoom, zoom => `×${zoom}`)}</select>
                    </label>
                </span>
                <button type="button" class="compare-button" data-action="reset-zoom" title="Reset zoom (or double-click)">
                    <i class="fas fa-compress"></i> 1:1
                </button>
                ${backgroundControlHtml(this.background)}
            </div>
            <div class="compare-toolbar">
                <div class="compare-frame-control">
                    <button type="button" class="compare-icon-button" data-action="prev" title="Previous frame (←)"><i class="fas fa-step-backward"></i></button>
                    <button type="button" class="compare-icon-button" data-action="play" title="Play (space)"><i class="fas fa-play"></i></button>
                    <button type="button" class="compare-icon-button" data-action="next" title="Next frame (→)"><i class="fas fa-step-forward"></i></button>
                    <input type="range" data-role="frame" min="0" max="0" value="0">
                    <span data-role="frame-label" class="compare-muted">-</span>
                </div>
                <label class="compare-inline">FPS
                    <select data-role="fps">${optionsHtml(FPS_OPTIONS, this.fps, fps => fps)}</select>
                </label>
            </div>
            <div class="depth-compare-stage">
                <div class="compare-layers depth-compare-panel">
                    <div class="depth-compare-layer-list" data-role="layers"></div>
                    <figure class="depth-rgb-thumb" data-role="rgb-thumb" hidden>
                        <div class="depth-rgb-frame" data-role="rgb-frame">
                            <img data-role="rgb-img" alt="RGB input" draggable="false">
                            <div class="depth-rgb-marker" data-role="rgb-marker" hidden></div>
                        </div>
                        <figcaption class="compare-muted">RGB input</figcaption>
                    </figure>
                </div>
                <div class="depth-compare-main">
                    <div class="depth-compare-area" data-role="area" tabindex="0">
                        <div class="compare-canvas-hint depth-compare-hint" data-role="hint">Add methods to compare their depth.</div>
                    </div>
                    <div class="depth-compare-legend" data-role="legend"></div>
                </div>
            </div>
            <div class="compare-readout" data-role="readout"></div>
        `;
        this.area = this.root.querySelector('[data-role="area"]');
        this.frameInput = this.root.querySelector('[data-role="frame"]');
        this.root.querySelector('[data-role="view"]').addEventListener('click', event => {
            const view = event.target.closest('[data-view]')?.dataset.view;
            if (view) { this.view = view; this.render(); }
        });
        this.root.querySelector('[data-role="map"]').addEventListener('click', event => {
            const kind = event.target.closest('[data-map]')?.dataset.map;
            if (kind) { this.mapKind = kind; this.render(); }
        });
        this.root.querySelector('[data-role="left"]').addEventListener('change', event => { this.left = event.target.value; this.render(); });
        this.root.querySelector('[data-role="right"]').addEventListener('change', event => { this.right = event.target.value; this.render(); });
        this.root.querySelector('[data-role="magnifier"]').addEventListener('change', event => {
            this.magnifier.enabled = event.target.checked;
            this.updatePointerOverlays();
            this.writeState();
        });
        this.root.querySelector('[data-role="magnifier-size"]').addEventListener('change', event => {
            this.magnifier.size = Number(event.target.value);
            this.updatePointerOverlays();
            this.writeState();
        });
        this.root.querySelector('[data-role="magnifier-zoom"]').addEventListener('change', event => {
            this.magnifier.zoom = Number(event.target.value);
            this.updatePointerOverlays();
            this.writeState();
        });
        bindBackgroundControl(this.root, 'depth', color => {
            this.background = color;
            this.root.style.setProperty('--depth-compare-bg', color);
        });
        this.root.querySelector('[data-role="fps"]').addEventListener('change', event => {
            this.fps = Number(event.target.value);
            if (this.playing) { this.stop(); this.play(); }
        });
        this.root.querySelector('[data-action="reset-zoom"]').addEventListener('click', () => this.resetZoom());
        this.root.querySelector('[data-action="play"]').addEventListener('click', () => (this.playing ? this.stop() : this.play()));
        this.root.querySelector('[data-action="prev"]').addEventListener('click', () => this.step(-1));
        this.root.querySelector('[data-action="next"]').addEventListener('click', () => this.step(1));
        this.frameInput.addEventListener('input', () => this.setFrame(Number(this.frameInput.value)));
        this.installPointer();
        bindCtrlDragReorder(this.area, {
            // The pointer is captured while dragging, so the tile is found by position, not by the event target.
            pick: event => {
                const layer = document.elementFromPoint(event.clientX, event.clientY)
                    ?.closest('.depth-compare-tile')?.dataset.layer;
                return layer ? { id: layer, movable: layer !== 'gt' } : null;
            },
            onMove: (fromId, toId) => this.moveMethodTo(fromId, toId),
            showDrop: id => this.area.querySelectorAll('.depth-compare-tile').forEach(tile => (
                tile.classList.toggle('is-drop-target', tile.dataset.layer === id))),
        });
        this.area.addEventListener('keydown', event => {
            if (event.key === 'ArrowRight') { this.step(1); event.preventDefault(); }
            if (event.key === 'ArrowLeft') { this.step(-1); event.preventDefault(); }
            if (event.key === ' ') { (this.playing ? this.stop() : this.play()); event.preventDefault(); }
        });
        this.renderLayers();
    }

    // ---- layers ---------------------------------------------------------------------------------------------------

    async loadIndex() {
        this.index = this.index || await fetchAssetJson(`${INDEX_ROOT}/depth.json`);
        return this.index;
    }

    async openAdd() {
        const index = await this.loadIndex();
        const result = await openAddDialog({
            title: 'Add depth maps',
            scenes: index.scenes,
            current: this.scene,
            loadedMethods: this.methods,
            describeMethod: entry => `AbsRel ${formatNumber(entry.absrel)}`,
        });
        if (result) await this.addMethods(result.scene, result.methods);
    }

    async addMethods(scene, methods) {
        const sameScene = this.scene && this.scene.task === scene.task && this.scene.dataset_key === scene.dataset_key
            && this.scene.scene === scene.scene;
        if (!sameScene) {
            this.stop();
            this.methods = [];
            this.colors.clear();
            this.hidden.clear();
            this.imageCache.clear();
            this.frame = 0;
            this.manifest = { ...scene, frames: [], methods: {} };
            this.scene = scene;
            this.left = 'gt';
            this.right = null;
        }
        for (const method of methods) {
            const entry = scene.methods.find(item => item.method === method);
            if (!entry || this.methods.includes(method)) continue;
            try {
                const data = await fetchAssetJson(`${entry.path}/frames.json`);
                this.manifest.methods[method] = {
                    path: entry.path,
                    frames: data.frames.map(frame => frame.frame),
                    frame_absrel: data.frames.map(frame => frame.absrel),
                    metrics: data.metrics || {},
                };
                this.manifest.size = this.manifest.size || data.size;
                // Frames in evaluation order; a method's extra frames (if any) are appended.
                const known = new Set(this.manifest.frames);
                this.manifest.frames.push(...this.manifest.methods[method].frames.filter(frame => !known.has(frame)));
                this.methods.push(method);
                if (!this.colors.has(method)) this.colors.set(method, `hsl(${(this.colors.size * 67) % 360} 70% 55%)`);
            } catch (error) {
                console.error(error);
            }
        }
        if (!this.right || !this.methods.includes(this.right)) this.right = this.methods[0] || null;
        this.afterLayersChanged();
    }

    // Put method ``from`` where method ``to`` is (the others shift over).
    moveMethodTo(from, to) {
        if (!moveItem(this.methods, this.methods.indexOf(from), this.methods.indexOf(to))) return;
        this.renderLayers();
        this.render();
        this.writeState();
    }

    removeMethod(method) {
        this.methods = this.methods.filter(item => item !== method);
        this.hidden.delete(method);
        if (this.left === method) this.left = 'gt';
        if (this.right === method) this.right = this.methods[0] || null;
        if (!this.methods.length) { this.stop(); this.scene = null; this.manifest = null; }
        this.afterLayersChanged();
    }

    afterLayersChanged() {
        const frames = this.manifest ? this.manifest.frames.length : 0;
        this.frameInput.max = String(Math.max(0, frames - 1));
        this.frame = Math.min(this.frame, Math.max(0, frames - 1));
        this.frameInput.value = String(this.frame);
        this.renderLayers();
        this.render();
    }

    layerOptions() {
        // Depth layers only: the RGB input is the reference thumbnail beside the viewer layers.
        return [{ value: 'gt', label: 'Ground truth' }, ...this.methods.map(method => ({ value: method, label: methodLabel(method) }))];
    }

    layerUrl(layer, frameName) {
        if (!this.scene || !frameName) return null;
        const shared = `${ASSET_ROOT}/${this.scene.scene_path}`;
        if (layer === 'gt') return `${shared}/gt/${encodeURIComponent(frameName)}.webp`;
        if (layer === 'rgb') return this.manifest.has_rgb ? `${shared}/rgb/${encodeURIComponent(frameName)}.webp` : null;
        const info = this.manifest.methods[layer];
        if (!info || !info.frames.includes(frameName)) return null;
        return `${ASSET_ROOT}/${info.path}/${this.mapKind}/${encodeURIComponent(frameName)}.webp`;
    }

    layerLabel(layer) {
        if (layer === 'gt') return 'Ground truth';
        return `${methodLabel(layer)}${this.mapKind === 'error' ? ' · error' : ''}`;
    }

    tileLayers() {
        return ['gt', ...this.methods.filter(method => !this.hidden.has(method))];
    }

    renderLayers() {
        const container = this.root.querySelector('[data-role="layers"]');
        const layers = this.methods.map(method => {
            const info = this.manifest.methods[method];
            return {
                id: method, label: methodLabel(method), color: this.colors.get(method),
                visible: !this.hidden.has(method),
                detail: `AbsRel ${formatNumber(info.metrics.abs_relative_difference)} · δ<1.25 ${formatNumber(info.metrics.delta125_acc)}`,
            };
        });
        const sceneLabel = this.scene
            ? `${taskLabel(this.scene.task)} · ${datasetLabel(this.scene.dataset_key)} · ${this.scene.scene}` : '';
        renderLayerList(container, {
            title: 'Viewer layers', sceneLabel, layers, addLabel: 'Add methods',
            onToggle: id => { this.hidden.has(id) ? this.hidden.delete(id) : this.hidden.add(id); this.renderLayers(); this.render(); },
            onRemove: id => this.removeMethod(id),
            onAdd: () => this.openAdd(),
            onMove: (id, delta) => {
                const neighbour = this.methods[this.methods.indexOf(id) + delta];
                if (neighbour) this.moveMethodTo(id, neighbour);
            },
        });
        const fill = (select, value) => {
            select.innerHTML = this.layerOptions().map(option => (
                `<option value="${escapeHtml(option.value)}"${option.value === value ? ' selected' : ''}>${escapeHtml(option.label)}</option>`
            )).join('');
        };
        fill(this.root.querySelector('[data-role="left"]'), this.left);
        fill(this.root.querySelector('[data-role="right"]'), this.right);
    }

    // ---- rendering ----------------------------------------------------------------------------------------------

    frameName() {
        return this.manifest?.frames[this.frame] || null;
    }

    render() {
        this.root.querySelectorAll('[data-view]').forEach(button => button.classList.toggle('active', button.dataset.view === this.view));
        this.root.querySelectorAll('[data-map]').forEach(button => button.classList.toggle('active', button.dataset.map === this.mapKind));
        this.root.querySelector('[data-role="left-label"]').hidden = this.view !== 'swipe';
        this.root.querySelector('[data-role="right-label"]').hidden = this.view !== 'swipe';
        this.root.querySelector('[data-role="magnifier-tools"]').hidden = this.view !== 'grid';
        this.root.querySelector('[data-role="magnifier"]').checked = this.magnifier.enabled;
        this.root.querySelector('[data-role="magnifier-size"]').value = String(this.magnifier.size);
        this.root.querySelector('[data-role="magnifier-zoom"]').value = String(this.magnifier.zoom);
        const hint = this.root.querySelector('[data-role="hint"]');
        const thumb = this.root.querySelector('[data-role="rgb-thumb"]');
        if (!this.manifest || !this.methods.length) {
            this.area.querySelectorAll('.depth-compare-block, .depth-compare-grid').forEach(node => node.remove());
            hint.hidden = false;
            thumb.hidden = true;
            this.root.querySelector('[data-role="legend"]').innerHTML = '';
            this.root.querySelector('[data-role="readout"]').innerHTML = '';
            this.root.querySelector('[data-role="frame-label"]').textContent = '-';
            this.writeState();
            return;
        }
        hint.hidden = true;
        thumb.hidden = !this.manifest.has_rgb;
        this.root.querySelector('[data-role="rgb-frame"]').style.aspectRatio = this.blockAspect();
        if (this.view === 'swipe') this.renderSwipe(); else this.renderGrid();
        this.renderLegend();
        this.updateFrame();
        this.writeState();
    }

    imageSize() {
        return this.manifest?.size || { width: 4, height: 3 };
    }

    blockAspect() {
        const size = this.imageSize();
        return `${size.width} / ${size.height}`;
    }

    renderSwipe() {
        this.area.querySelectorAll('.depth-compare-grid').forEach(node => node.remove());
        let block = this.area.querySelector('.depth-compare-block');
        if (!block) {
            block = document.createElement('div');
            block.className = 'depth-compare-block';
            block.innerHTML = `
                <div class="depth-compare-zoom">
                    <img data-side="left" alt="" draggable="false">
                    <img data-side="right" alt="" draggable="false">
                </div>
                <div class="depth-compare-divider"></div>
                <span class="depth-compare-tag depth-compare-tag-left" data-tag="left"></span>
                <span class="depth-compare-tag depth-compare-tag-right" data-tag="right"></span>
            `;
            this.area.appendChild(block);
        }
        block.style.aspectRatio = this.blockAspect();
        block.querySelector('[data-tag="left"]').textContent = this.layerLabel(this.left);
        block.querySelector('[data-tag="right"]').textContent = this.right ? this.layerLabel(this.right) : '';
        this.applySwipe();
        this.applyZoom();
    }

    renderGrid() {
        this.area.querySelectorAll('.depth-compare-block').forEach(node => node.remove());
        let grid = this.area.querySelector('.depth-compare-grid');
        if (!grid) {
            grid = document.createElement('div');
            grid.className = 'depth-compare-grid';
            this.area.appendChild(grid);
        }
        const layers = this.tileLayers();
        grid.style.setProperty('--depth-grid-columns', String(Math.min(layers.length, GRID_COLUMNS)));
        grid.innerHTML = layers.map(layer => `
            <div class="depth-compare-tile" style="aspect-ratio:${this.blockAspect()}" data-layer="${escapeHtml(layer)}">
                <div class="depth-compare-zoom"><img data-layer-img="${escapeHtml(layer)}" alt="" draggable="false"></div>
                <div class="depth-compare-crosshair" hidden></div>
                <div class="depth-compare-lens" hidden></div>
                <span class="depth-compare-tag depth-compare-tag-left">${escapeHtml(this.layerLabel(layer))}</span>
                <span class="depth-compare-tag depth-compare-tag-right" data-metric="${escapeHtml(layer)}"></span>
            </div>
        `).join('');
        this.applyZoom();
    }

    renderLegend() {
        const legend = this.root.querySelector('[data-role="legend"]');
        const [low, high] = this.manifest.depth_range;
        const error = this.manifest.error;
        legend.innerHTML = `
            <span class="depth-legend-item">
                <span class="compare-muted">Depth (m)</span> ${formatNumber(low, 2)}
                <span class="depth-legend-bar" style="background:${colormapGradient(this.manifest.depth_colormap)}"></span>
                ${formatNumber(high, 2)}
            </span>
            <span class="depth-legend-item">
                <span class="compare-muted">AbsRel error</span> ${error.vmin}
                <span class="depth-legend-bar" style="background:${colormapGradient(error.colormap)}"></span>
                ≥${error.vmax}
            </span>
            <span class="compare-muted">All methods share the GT's color scale; black or grey = no valid depth.</span>
        `;
    }

    updateFrame() {
        const name = this.frameName();
        const total = this.manifest.frames.length;
        this.root.querySelector('[data-role="frame-label"]').textContent = `${this.frame + 1} / ${total} · ${name}`;
        this.setImage(this.root.querySelector('[data-role="rgb-img"]'), this.layerUrl('rgb', name));
        if (this.view === 'swipe') {
            this.setImage(this.area.querySelector('img[data-side="left"]'), this.layerUrl(this.left, name));
            this.setImage(this.area.querySelector('img[data-side="right"]'), this.right ? this.layerUrl(this.right, name) : null);
        } else {
            this.area.querySelectorAll('img[data-layer-img]').forEach(img => this.setImage(img, this.layerUrl(img.dataset.layerImg, name)));
            this.area.querySelectorAll('[data-metric]').forEach(tag => {
                const info = this.manifest.methods[tag.dataset.metric];
                const index = info ? info.frames.indexOf(name) : -1;
                tag.textContent = index >= 0 ? `AbsRel ${formatNumber(info.frame_absrel[index])}` : '';
            });
        }
        const readout = this.methods.map(method => {
            const info = this.manifest.methods[method];
            const index = info.frames.indexOf(name);
            return `<span class="compare-readout-item">${escapeHtml(methodLabel(method))}
                <strong>${index >= 0 ? formatNumber(info.frame_absrel[index]) : 'no frame'}</strong></span>`;
        });
        this.root.querySelector('[data-role="readout"]').innerHTML = readout.length
            ? `<span class="compare-muted">AbsRel at this frame:</span> ${readout.join('')}` : '';
        this.updatePointerOverlays();  // the magnifier patches show the new frame
        this.preload();
    }

    setImage(img, url) {
        if (!img) return;
        if (!url) {
            img.removeAttribute('src');
            img.classList.add('is-missing');
            return;
        }
        img.classList.remove('is-missing');
        if (img.getAttribute('src') !== url) img.src = this.cachedImage(url).src;
    }

    cachedImage(url) {
        if (!this.imageCache.has(url)) {
            const image = new Image();
            image.decoding = 'async';
            image.src = url;
            this.imageCache.set(url, image);
        }
        return this.imageCache.get(url);
    }

    visibleLayers() {
        const rgb = this.manifest?.has_rgb ? ['rgb'] : [];
        if (this.view === 'swipe') return [this.left, this.right, ...rgb].filter(Boolean);
        return [...this.tileLayers(), ...rgb];
    }

    preload() {
        const frames = this.manifest.frames;
        for (let offset = 1; offset <= PRELOAD_AHEAD; offset += 1) {
            const name = frames[(this.frame + offset) % frames.length];
            for (const layer of this.visibleLayers()) {
                const url = this.layerUrl(layer, name);
                if (url) this.cachedImage(url);
            }
        }
    }

    // ---- frames and playback ----------------------------------------------------------------------------------

    setFrame(frame) {
        if (!this.manifest) return;
        const total = this.manifest.frames.length;
        this.frame = ((frame % total) + total) % total;
        this.frameInput.value = String(this.frame);
        this.updateFrame();
        this.writeState();
    }

    step(delta) {
        this.stop();
        this.setFrame(this.frame + delta);
    }

    frameReady(frame) {
        const name = this.manifest.frames[frame];
        return this.visibleLayers().every(layer => {
            const url = this.layerUrl(layer, name);
            return !url || this.cachedImage(url).complete;
        });
    }

    play() {
        if (!this.manifest) return;
        this.playing = true;
        this.root.querySelector('[data-action="play"] i').className = 'fas fa-pause';
        this.playTimer = setInterval(() => {
            const next = (this.frame + 1) % this.manifest.frames.length;
            // Wait for the next frame's images rather than showing a half-loaded frame.
            if (this.frameReady(next)) this.setFrame(next); else this.preload();
        }, 1000 / this.fps);
    }

    stop() {
        this.playing = false;
        clearInterval(this.playTimer);
        const icon = this.root.querySelector('[data-action="play"] i');
        if (icon) icon.className = 'fas fa-play';
    }

    // ---- swipe, zoom, pan, crosshair, magnifier ---------------------------------------------------------------

    applySwipe() {
        const block = this.area.querySelector('.depth-compare-block');
        if (!block) return;
        const percent = `${(this.swipe * 100).toFixed(2)}%`;
        block.querySelector('img[data-side="right"]').style.clipPath = `inset(0 0 0 ${percent})`;
        block.querySelector('.depth-compare-divider').style.left = percent;
    }

    applyZoom() {
        const transform = `translate(${this.pan.x}px, ${this.pan.y}px) scale(${this.zoom})`;
        this.area.querySelectorAll('.depth-compare-zoom').forEach(node => { node.style.transform = transform; });
        this.updatePointerOverlays();
    }

    resetZoom() {
        this.zoom = 1;
        this.pan = { x: 0, y: 0 };
        this.applyZoom();
    }

    imageRect(tile) {
        // Where the image is drawn in an unzoomed tile (object-fit: contain), in tile pixels.
        const width = tile.clientWidth;
        const height = tile.clientHeight;
        const size = this.imageSize();
        const drawnWidth = Math.min(width, (height * size.width) / size.height);
        const drawnHeight = (drawnWidth * size.height) / size.width;
        return { x: (width - drawnWidth) / 2, y: (height - drawnHeight) / 2, width: drawnWidth, height: drawnHeight };
    }

    imageFraction(tile, fx, fy) {
        // The image point under (fx, fy) of the tile, undoing the shared zoom and pan; as fractions of the image.
        const rect = this.imageRect(tile);
        const x = (fx * tile.clientWidth - this.pan.x) / this.zoom;
        const y = (fy * tile.clientHeight - this.pan.y) / this.zoom;
        return { u: (x - rect.x) / rect.width, v: (y - rect.y) / rect.height, rect };
    }

    updatePointerOverlays() {
        // The same relative position in every tile, so details line up across methods: a crosshair, or with the
        // magnifier on, a zoomed-in patch of each tile's own image centered at the cursor.
        const pointer = this.pointer;
        const magnify = Boolean(pointer) && this.magnifier.enabled && this.view === 'grid';
        this.area.querySelectorAll('.depth-compare-crosshair').forEach(cross => {
            cross.hidden = !pointer || magnify;
            if (pointer) { cross.style.left = `${pointer.fx * 100}%`; cross.style.top = `${pointer.fy * 100}%`; }
        });
        const { size, zoom } = this.magnifier;
        this.area.querySelectorAll('.depth-compare-tile').forEach(tile => {
            const lens = tile.querySelector('.depth-compare-lens');
            const img = tile.querySelector('img[data-layer-img]');
            const src = img?.getAttribute('src');
            lens.hidden = !magnify;
            if (!magnify) return;
            const { u, v, rect } = this.imageFraction(tile, pointer.fx, pointer.fy);
            const width = rect.width * this.zoom * zoom;
            const height = rect.height * this.zoom * zoom;
            Object.assign(lens.style, {
                width: `${size}px`,
                height: `${size}px`,
                left: `${pointer.fx * tile.clientWidth - size / 2}px`,
                top: `${pointer.fy * tile.clientHeight - size / 2}px`,
                backgroundImage: src ? `url("${src}")` : 'none',
                backgroundSize: `${width}px ${height}px`,
                backgroundPosition: `${size / 2 - u * width}px ${size / 2 - v * height}px`,
            });
        });
        this.updateThumbMarker();
    }

    updateThumbMarker() {
        // The RGB thumbnail marks the image point under the cursor (with the magnifier: the patch's footprint).
        const marker = this.root.querySelector('[data-role="rgb-marker"]');
        const tile = this.pointer && this.area.querySelector('.depth-compare-tile, .depth-compare-block');
        if (!marker || !tile) { if (marker) marker.hidden = true; return; }
        const { u, v, rect } = this.imageFraction(tile, this.pointer.fx, this.pointer.fy);
        let width = 0;
        let height = 0;
        if (this.magnifier.enabled && this.view === 'grid') {
            width = this.magnifier.size / (rect.width * this.zoom * this.magnifier.zoom);
            height = this.magnifier.size / (rect.height * this.zoom * this.magnifier.zoom);
        }
        marker.classList.toggle('is-point', !width);
        Object.assign(marker.style, {
            left: `${(u - width / 2) * 100}%`, top: `${(v - height / 2) * 100}%`,
            width: `${width * 100}%`, height: `${height * 100}%`,
        });
        marker.hidden = u < 0 || u > 1 || v < 0 || v > 1;
    }

    installPointer() {
        let drag = null;
        const tileOf = event => event.target.closest('.depth-compare-block, .depth-compare-tile');
        const fractions = (event, tile) => {
            const rect = tile.getBoundingClientRect();
            return { fx: (event.clientX - rect.left) / rect.width, fy: (event.clientY - rect.top) / rect.height };
        };
        this.area.addEventListener('pointermove', event => {
            const tile = tileOf(event);
            if (!tile) return;
            const { fx, fy } = fractions(event, tile);
            if (this.pinned) {  // a pinned patch stays where it is; dragging still pans the tiles under it
                if (drag) {
                    this.pan.x = drag.panX + event.clientX - drag.x;
                    this.pan.y = drag.panY + event.clientY - drag.y;
                    this.applyZoom();
                }
                return;
            }
            if (drag) {
                this.pan.x = drag.panX + event.clientX - drag.x;
                this.pan.y = drag.panY + event.clientY - drag.y;
                this.applyZoom();
            } else if (tile.classList.contains('depth-compare-block')) {
                this.swipe = Math.min(1, Math.max(0, fx));
                this.applySwipe();
            }
            this.pointer = { fx, fy };
            this.updatePointerOverlays();
        });
        this.area.addEventListener('pointerleave', () => {
            if (this.pinned) return;
            this.pointer = null;
            this.updatePointerOverlays();
        });
        this.area.addEventListener('contextmenu', event => {
            const tile = tileOf(event);
            if (!tile) return;
            event.preventDefault();
            this.pinned = !this.pinned;
            this.pointer = fractions(event, tile);
            this.area.classList.toggle('is-pinned', this.pinned);
            this.updatePointerOverlays();
        });
        this.area.addEventListener('pointerdown', event => {
            if (event.button !== 0) return;  // right-click pins (contextmenu), it does not pan
            if (!tileOf(event) || (this.zoom === 1 && this.view === 'swipe' && !event.shiftKey)) return;
            drag = { x: event.clientX, y: event.clientY, panX: this.pan.x, panY: this.pan.y };
            this.area.setPointerCapture(event.pointerId);
        });
        this.area.addEventListener('pointerup', () => { drag = null; });
        this.area.addEventListener('dblclick', () => this.resetZoom());
        this.area.addEventListener('wheel', event => {
            const tile = tileOf(event);
            if (!tile) return;
            event.preventDefault();
            const rect = tile.getBoundingClientRect();
            const cx = event.clientX - rect.left;
            const cy = event.clientY - rect.top;
            const next = Math.min(MAX_ZOOM, Math.max(1, this.zoom * (event.deltaY < 0 ? 1.15 : 1 / 1.15)));
            // Keep the point under the cursor fixed while zooming.
            this.pan.x = cx - ((cx - this.pan.x) * next) / this.zoom;
            this.pan.y = cy - ((cy - this.pan.y) * next) / this.zoom;
            this.zoom = next;
            if (this.zoom === 1) this.pan = { x: 0, y: 0 };
            this.applyZoom();
        }, { passive: false });
    }

    // ---- state --------------------------------------------------------------------------------------------------

    writeState() {
        const magnifier = this.view === 'grid' && this.magnifier.enabled;
        writeCompareState(this.scene ? {
            compare: 'depth', task: this.scene.task, dataset: this.scene.dataset_key, scene: this.scene.scene,
            methods: this.methods, view: this.view, map: this.mapKind, left: this.left, right: this.right,
            mag: this.magnifier.enabled ? 1 : 0,
            mag_size: magnifier ? this.magnifier.size : undefined,
            mag_zoom: magnifier ? this.magnifier.zoom : undefined,
            frame: this.frame || undefined,
        } : { compare: 'depth' });
    }

    resize() {}

    async restore(state) {
        if (!state.task || !state.dataset || !state.scene) return;
        const index = await this.loadIndex();
        const scene = index.scenes.find(item => item.task === state.task && item.dataset_key === state.dataset
            && item.scene === state.scene);
        if (!scene) return;
        await this.addMethods(scene, state.methods || []);
        // Grid with the magnifier on by default; view=magnifier: links from before the magnifier moved into the grid.
        this.view = state.view === 'swipe' ? 'swipe' : 'grid';
        this.magnifier.enabled = state.mag !== '0';
        if (MAGNIFIER_SIZES.includes(Number(state.mag_size))) this.magnifier.size = Number(state.mag_size);
        if (MAGNIFIER_ZOOMS.includes(Number(state.mag_zoom))) this.magnifier.zoom = Number(state.mag_zoom);
        this.mapKind = state.map === 'error' ? 'error' : 'depth';
        const known = new Set(this.layerOptions().map(option => option.value));
        if (known.has(state.left)) this.left = state.left;
        if (known.has(state.right)) this.right = state.right;
        this.renderLayers();
        this.render();
        if (state.frame) this.setFrame(Number(state.frame));
    }
}
