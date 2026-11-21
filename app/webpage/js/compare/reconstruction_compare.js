// Copyright (c) 2026 Yu Chen. All rights reserved.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

// Reconstruction comparison (MVRecon): each method's fused cloud, aligned to the GT as scored, in its own window.
// All windows are viewports of one renderer looking through one shared camera, so orbiting, zooming or a shortcut
// key in any window moves every window. Points show their fused image color or their error (distance to the nearest
// GT point), from the released MVRecon assets (artifacts/). Optionally each window also draws the fused views'
// camera frustums (cameras.json: the method's cameras as fused, aligned like its cloud; the GT window the GT cameras).
// Where the case has confidence.f32 (each point's predicted confidence, from the views that see it), the lowest
// percentile of every method's points can be hidden.

import * as THREE from 'three';
import { TrackballControls } from 'three/addons/controls/TrackballControls.js';
import {
    ASSET_ROOT, INDEX_ROOT, backgroundControlHtml, bindBackgroundControl, bindColorControl, bindFullscreen,
    bindWheelShortcuts, colorControlHtml, confidenceArray, confidenceControlHtml, confidenceThreshold, datasetLabel,
    enableConfidenceFilter, escapeHtml, fetchAssetJson, formatNumber, fullscreenButtonHtml, isLightColor,
    loadBackground, loadColor, methodLabel, moveItem, bindCtrlDragReorder, openAddDialog,
    pushFrustumSegments, renderLayerList, taskLabel,
    writeCompareState,
} from './compare_common.js';

const PRED_CAMERA_COLOR = '#f59e0b';
const CAMERA_DEPTH = 0.03;  // frustum depth as a fraction of the scene radius, at size 1
const GT_LAYER = '__gt__';
const UP_AXES = {
    '-y': new THREE.Vector3(0, -1, 0), y: new THREE.Vector3(0, 1, 0), z: new THREE.Vector3(0, 0, 1),
};
const KEY_ORBIT_RADIANS = Math.PI / 36;

// Turbo colormap (Mikhailov 2019, polynomial approximation), t in [0, 1] -> [r, g, b] in [0, 1].
function turbo(t) {
    const x = Math.min(1, Math.max(0, t));
    const r = 0.13572138 + x * (4.6153926 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))));
    const g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))));
    const b = 0.1066733 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))));
    return [Math.min(1, Math.max(0, r)), Math.min(1, Math.max(0, g)), Math.min(1, Math.max(0, b))];
}

// GT scans without colors (DTU's STL, NRGBD's mesh) are shaded by height along the up axis: low -> high.
const HEIGHT_STOPS = [[0.18, 0.25, 0.45], [0.30, 0.58, 0.62], [0.86, 0.80, 0.58]];

function heightColor(t) {
    const x = Math.min(1, Math.max(0, t)) * (HEIGHT_STOPS.length - 1);
    const index = Math.min(HEIGHT_STOPS.length - 2, Math.floor(x));
    const f = x - index;
    return HEIGHT_STOPS[index].map((value, channel) => value + (HEIGHT_STOPS[index + 1][channel] - value) * f);
}

async function fetchArray(url, Type) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
    return new Type(await response.arrayBuffer());
}

export class ReconstructionCompare {
    constructor(root) {
        this.root = root;
        this.scene = null;  // index entry
        this.manifest = null;  // scene.json
        this.panels = [];  // {id, label, color, cloud, data: {points, colors, error}, visible}, in window order
        this.colorCount = 1;  // a window keeps the color it was added with when it is moved (hue 0 was the GT's slot)
        this.colorMode = 'image';
        this.pointSize = 2;
        this.showGtOverlay = false;
        this.upAxis = '-y';
        this.showCameras = false;
        this.cameraScale = 1;
        this.cameraColor = loadColor('reconstruction.cameras', PRED_CAMERA_COLOR);
        this.confidencePercent = 0;
        this.background = loadBackground('reconstruction');
        this.buildDom();
        this.initThree();
    }

    buildDom() {
        this.root.innerHTML = `
            <div class="compare-toolbar">
                <div class="compare-segmented" data-role="color">
                    <button type="button" data-color="image" class="active">Original color</button>
                    <button type="button" data-color="error">Error map</button>
                </div>
                <label class="compare-inline" data-role="vmax-label">Error max
                    <input type="range" data-role="vmax" min="0.005" max="0.5" step="0.005" value="0.1">
                    <span data-role="vmax-value" class="compare-muted"></span>
                </label>
                <label class="compare-inline" title="Ctrl + wheel over the view">Point size
                    <input type="range" data-role="size" min="0.5" max="6" step="0.5" value="${this.pointSize}">
                </label>
                ${confidenceControlHtml()}
                <label class="compare-inline" title="Show the ground-truth window">
                    <input type="checkbox" data-role="gt-window" checked> GT window
                </label>
                <label class="compare-inline" title="Draw the GT points in every method window">
                    <input type="checkbox" data-role="overlay"> GT overlay
                </label>
                <label class="compare-inline" title="Camera frustums of the fused views (GT overlay adds the GT cameras)">
                    <input type="checkbox" data-role="cameras"> Cameras
                </label>
                <span class="compare-inline" data-role="camera-tools" hidden>
                    <label class="compare-inline" title="Alt + wheel over the view">Frustum size
                        <input type="range" data-role="camera-size" min="0.25" max="4" step="0.25" value="${this.cameraScale}">
                    </label>
                    ${colorControlHtml('camera-color', 'Frustum color', this.cameraColor, "The methods' camera frustums")}
                </span>
                ${backgroundControlHtml(this.background)}
                <label class="compare-inline">Up
                    <select data-role="up"><option value="-y">-Y</option><option value="y">+Y</option><option value="z">+Z</option></select>
                </label>
                <div class="compare-segmented" data-role="views">
                    <button type="button" data-view="fit" title="Fit (F)">Fit</button>
                    <button type="button" data-view="x" title="Look along X (1)">X</button>
                    <button type="button" data-view="y" title="Look along Y (2)">Y</button>
                    <button type="button" data-view="z" title="Look along Z (3)">Z</button>
                </div>
                ${fullscreenButtonHtml()}
            </div>
            <div class="compare-stage">
                <div class="compare-canvas-wrap recon-compare-wrap" tabindex="0" data-role="wrap">
                    <canvas class="compare-canvas"></canvas>
                    <div class="recon-compare-labels" data-role="labels"></div>
                    <div class="compare-canvas-hint" data-role="hint">Add methods to compare their reconstructions.</div>
                </div>
                <aside class="compare-layers" data-role="layers"></aside>
            </div>
            <div class="compare-readout compare-muted" data-role="camera-note" hidden></div>
            <div class="compare-readout compare-muted" data-role="confidence-note" hidden></div>
            <div class="compare-readout" data-role="readout">
                <span class="compare-muted">Drag to orbit, right-drag to pan, wheel to zoom, in any window; Ctrl + wheel point
                size, Alt + wheel frustum size; keys: arrows orbit, +/- zoom, F fit, 1/2/3 axis views. Ctrl + drag a window
                onto another to move it there (or use the arrows in the window list).</span>
            </div>
        `;
        this.wrap = this.root.querySelector('[data-role="wrap"]');
        this.canvas = this.root.querySelector('canvas');
        this.root.querySelector('[data-role="color"]').addEventListener('click', event => {
            const mode = event.target.closest('[data-color]')?.dataset.color;
            if (mode) { this.colorMode = mode; this.recolorAll(); }
        });
        this.vmaxInput = this.root.querySelector('[data-role="vmax"]');
        this.vmaxInput.addEventListener('input', () => this.recolorAll());
        this.root.querySelector('[data-role="size"]').addEventListener('input', event => {
            this.pointSize = Number(event.target.value);
            for (const panel of this.panels) panel.cloud.material.size = this.pointSize;
            if (this.gtOverlay) this.gtOverlay.material.size = Math.max(0.5, this.pointSize * 0.75);
            this.requestRender();
        });
        this.root.querySelector('[data-role="gt-window"]').addEventListener('change', event => {
            this.setPanelVisible(GT_LAYER, event.target.checked);
        });
        this.root.querySelector('[data-role="overlay"]').addEventListener('change', event => {
            this.showGtOverlay = event.target.checked;
            this.requestRender();
        });
        this.root.querySelector('[data-role="cameras"]').addEventListener('change', event => {
            this.showCameras = event.target.checked;
            this.root.querySelector('[data-role="camera-tools"]').hidden = !this.showCameras;
            this.rebuildCameras();
        });
        this.root.querySelector('[data-role="camera-size"]').addEventListener('input', event => {
            this.cameraScale = Number(event.target.value);
            this.rebuildCameras();
        });
        bindColorControl(this.root, 'camera-color', 'reconstruction.cameras', color => {
            this.cameraColor = color;
            this.rebuildCameras();
        });
        this.root.querySelector('[data-role="confidence"]').addEventListener('input', event => {
            this.confidencePercent = Number(event.target.value);
            this.applyConfidence();
        });
        bindBackgroundControl(this.root, 'reconstruction', color => {
            this.background = color;
            this.rebuildCameras();  // the GT cameras' ink follows the background
        });
        this.root.querySelector('[data-role="up"]').addEventListener('change', event => {
            this.upAxis = event.target.value;
            this.recolorAll();  // height shading follows the up axis
            this.fitView();
        });
        this.root.querySelector('[data-role="views"]').addEventListener('click', event => {
            const view = event.target.closest('[data-view]')?.dataset.view;
            if (view) this.fitView(view === 'fit' ? null : view);
        });
        this.wrap.addEventListener('keydown', event => this.handleKey(event));
        bindCtrlDragReorder(this.wrap, {
            pick: event => {
                const panel = this.panelAt(event);
                return panel ? { id: panel.id } : null;
            },
            onMove: (fromId, toId) => this.movePanelTo(fromId, toId),
            showDrop: id => this.showDrop(id),
        });
        bindFullscreen(this.root, this.root);
        bindWheelShortcuts(this.wrap, {
            ctrl: () => this.root.querySelector('[data-role="size"]'),
            alt: () => this.root.querySelector('[data-role="camera-size"]'),
        });
        this.renderLayers();
    }

    initThree() {
        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, preserveDrawingBuffer: true });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        this.renderer.setScissorTest(true);
        this.camera = new THREE.PerspectiveCamera(50, 1, 0.001, 1e5);
        this.createControls();
        this.resizeObserver = new ResizeObserver(() => this.resize());
        this.resizeObserver.observe(this.wrap);
        this.resize();
    }

    createControls() {
        // One set of controls on the whole canvas drives the shared camera, so every viewport follows. A trackball
        // turns the view about any axis (over the poles, and rolling it), unlike an orbit control, which keeps the up
        // axis up. Rebuilt on each fit, which also drops any rotation still coasting from the last drag.
        const target = this.controls ? this.controls.target.clone() : new THREE.Vector3();
        this.controls?.dispose();
        this.controls = new TrackballControls(this.camera, this.canvas);
        Object.assign(this.controls, { rotateSpeed: 2.5, zoomSpeed: 1.2, panSpeed: 0.8, dynamicDampingFactor: 0.15 });
        // No A/S/D modifier keys: typing elsewhere on the page must not change what a drag does.
        this.controls.keys = ['', '', ''];
        this.controls.target.copy(target);
        this.controls.addEventListener('change', () => this.requestRender());
        // The trackball moves the camera only in update(): keep rendering (and updating) while a drag lasts.
        this.controls.addEventListener('start', () => { this.dragging = true; this.requestRender(); });
        this.controls.addEventListener('end', () => { this.dragging = false; this.requestRender(); });
        this.dragging = false;
    }

    resize() {
        const width = Math.max(1, this.wrap.clientWidth);
        const height = Math.max(1, this.wrap.clientHeight);
        this.renderer.setSize(width, height, false);
        this.controls.handleResize();
        this.layoutViewports();
        this.requestRender();
    }

    visiblePanels() {
        return this.panels.filter(panel => panel.visible);
    }

    // Grid of viewports: as square as the canvas allows.
    layoutViewports() {
        const panels = this.visiblePanels();
        const width = Math.max(1, this.wrap.clientWidth);
        const height = Math.max(1, this.wrap.clientHeight);
        const count = Math.max(1, panels.length);
        let best = { columns: 1, score: -Infinity };
        for (let columns = 1; columns <= count; columns += 1) {
            const rows = Math.ceil(count / columns);
            const cell = Math.min(width / columns, height / rows);
            if (cell > best.score) best = { columns, score: cell };
        }
        const columns = best.columns;
        const rows = Math.ceil(count / columns);
        const cellWidth = width / columns;
        const cellHeight = height / rows;
        panels.forEach((panel, index) => {
            const column = index % columns;
            const row = Math.floor(index / columns);
            panel.viewport = { x: column * cellWidth, y: row * cellHeight, width: cellWidth, height: cellHeight };
        });
        this.camera.aspect = cellWidth / cellHeight;
        this.camera.updateProjectionMatrix();
        this.renderLabels();
    }

    requestRender() {
        if (this.renderQueued) return;
        this.renderQueued = true;
        requestAnimationFrame(() => {
            this.renderQueued = false;
            // The trackball reports its damped motion through 'change' events; during a drag, keep updating.
            this.controls.update();
            if (this.dragging) this.requestRender();
            this.renderFrame();
        });
    }

    renderFrame() {
        const height = Math.max(1, this.wrap.clientHeight);
        this.renderer.setScissor(0, 0, this.wrap.clientWidth, height);
        this.renderer.setClearColor(new THREE.Color(this.background));
        this.renderer.clear();
        // A window added since the last layout has no viewport yet; the layout that follows renders it.
        for (const panel of this.visiblePanels().filter(item => item.viewport)) {
            const { x, y, width, height: cellHeight } = panel.viewport;
            const bottom = height - y - cellHeight;  // WebGL viewports count from the bottom
            this.renderer.setViewport(x, bottom, width, cellHeight);
            this.renderer.setScissor(x + 1, bottom + 1, width - 2, cellHeight - 2);
            panel.scene.children.forEach(child => {
                if (child.userData.isOverlay) child.visible = this.showGtOverlay && panel.id !== GT_LAYER;
                if (child.userData.isCameras) child.visible = this.showCameras;
                if (child.userData.isGtCameras) child.visible = this.showCameras && this.showGtOverlay;
            });
            this.renderer.render(panel.scene, this.camera);
        }
    }

    renderLabels() {
        const labels = this.root.querySelector('[data-role="labels"]');
        labels.innerHTML = this.visiblePanels().filter(panel => panel.viewport).map(panel => `
            <div class="recon-compare-label" style="left:${panel.viewport.x + 8}px;top:${panel.viewport.y + 8}px">
                <strong>${escapeHtml(panel.label)}</strong>
                ${panel.detail ? `<span>${panel.detail}</span>` : ''}
            </div>
        `).join('');
    }

    // ---- window order ----------------------------------------------------------------------------------------

    // The visible window under the pointer (the canvas is split into the viewports layoutViewports computed).
    panelAt(event) {
        const rect = this.wrap.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        return this.visiblePanels().find(panel => panel.viewport && x >= panel.viewport.x && y >= panel.viewport.y
            && x < panel.viewport.x + panel.viewport.width && y < panel.viewport.y + panel.viewport.height) || null;
    }

    showDrop(id) {
        const labels = this.root.querySelector('[data-role="labels"]');
        labels.querySelector('.recon-compare-drop')?.remove();
        const panel = id && this.panels.find(item => item.id === id);
        if (!panel?.viewport) return;
        const { x, y, width, height } = panel.viewport;
        labels.insertAdjacentHTML('beforeend',
            `<div class="recon-compare-drop" style="left:${x}px;top:${y}px;width:${width}px;height:${height}px"></div>`);
    }

    // Put window ``fromId`` where window ``toId`` is (the others shift over).
    movePanelTo(fromId, toId) {
        const from = this.panels.findIndex(panel => panel.id === fromId);
        const to = this.panels.findIndex(panel => panel.id === toId);
        if (from < 0 || to < 0 || !moveItem(this.panels, from, to)) return;
        this.layoutViewports();
        this.renderLayers();
        this.requestRender();
        this.writeState();
    }

    // ---- loading ---------------------------------------------------------------------------------------------

    async loadIndex() {
        this.index = this.index || await fetchAssetJson(`${INDEX_ROOT}/recon.json`);
        return this.index;
    }

    async openAdd() {
        const index = await this.loadIndex();
        const result = await openAddDialog({
            title: 'Add reconstructions',
            scenes: index.scenes,
            current: this.scene,
            loadedMethods: this.panels.map(panel => panel.id),
            describeMethod: entry => `F-score ${formatNumber(entry.fscore)}`,
        });
        if (result) await this.addMethods(result.scene, result.methods);
    }

    async addMethods(scene, methods) {
        const sameScene = this.scene && this.scene.dataset_key === scene.dataset_key && this.scene.scene === scene.scene;
        const hint = this.root.querySelector('[data-role="hint"]');
        hint.hidden = false;
        hint.textContent = 'Loading point clouds...';
        if (!sameScene) {
            this.clear();
            this.scene = scene;
            this.manifest = { ...scene, methods: Object.fromEntries(scene.methods.map(entry => [entry.method, entry])) };
            const error = this.manifest.error;
            this.vmaxInput.max = String(Math.max(error.vmax * 4, 0.01));
            this.vmaxInput.step = String(error.vmax / 100);
            this.vmaxInput.value = String(error.vmax);
            const points = await fetchArray(`${ASSET_ROOT}/${scene.scene_path}/gt_points.f32`, Float32Array);
            // The GT scan's own colors (gt_colors.u8) where it has them; otherwise height shading (recolor).
            const colors = scene.gt_colors === false ? null
                : await fetchArray(`${ASSET_ROOT}/${scene.scene_path}/gt_colors.u8`, Uint8Array)
                    .then(array => (array.length === points.length ? array : null))
                    .catch(() => null);
            this.gtData = { points, colors };
            this.gtOverlay = this.makeCloud(points, null, 0x9aa4b2);
            this.addPanel(GT_LAYER, 'Ground truth', { points, colors, error: null }, `${points.length / 3} points`);
            this.root.querySelector('[data-role="gt-window"]').checked = true;
        }
        for (const method of methods) {
            if (this.panels.some(panel => panel.id === method) || !this.manifest.methods[method]) continue;
            const base = `${ASSET_ROOT}/${this.manifest.methods[method].path}`;
            // The released index says which optional files a case has (false: skip the request); otherwise try them.
            const { cameras: hasCameras, confidence: hasConfidence } = this.manifest.methods[method];
            try {
                const [points, colors, error, cameras, rawConfidence] = await Promise.all([
                    fetchArray(`${base}/points.f32`, Float32Array),
                    fetchArray(`${base}/colors.u8`, Uint8Array),
                    fetchArray(`${base}/error.f32`, Float32Array),
                    // Cases evaluated before cameras (or confidences) were saved have none.
                    hasCameras === false ? null
                        : fetch(`${base}/cameras.json`).then(response => (response.ok ? response.json() : null)).catch(() => null),
                    hasConfidence === false ? null : fetchArray(`${base}/confidence.f32`, Float32Array).catch(() => null),
                ]);
                const confidence = rawConfidence?.length === points.length / 3 ? confidenceArray(rawConfidence) : null;
                const metrics = this.manifest.methods[method];
                this.addPanel(method, methodLabel(method), { points, colors, error, cameras, confidence },
                    `F-score ${formatNumber(metrics.fscore)} · Acc ${formatNumber(metrics.accuracy)} · Comp ${formatNumber(metrics.completion)}`);
            } catch (loadError) {
                console.error(loadError);
                hint.textContent = `Could not load ${method}: ${loadError.message}`;
            }
        }
        hint.hidden = this.panels.length > 1;
        if (this.panels.length <= 1) hint.textContent = 'Add methods to compare their reconstructions.';
        this.afterLayersChanged(!sameScene);
    }

    makeCloud(points, colors, fallbackColor) {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(points, 3));
        const material = new THREE.PointsMaterial({ size: this.pointSize, sizeAttenuation: false });
        if (colors) {
            geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(points.length), 3));
            material.vertexColors = true;
        } else {
            material.color = new THREE.Color(fallbackColor);
        }
        return new THREE.Points(geometry, material);
    }

    addPanel(id, label, data, detail) {
        const cloud = this.makeCloud(data.points, true, 0xcbd5e1);
        if (data.confidence) {
            cloud.geometry.setAttribute('confidence', new THREE.BufferAttribute(data.confidence, 1));
            data.confidenceUniform = enableConfidenceFilter(cloud.material);
        }
        const scene = new THREE.Scene();
        scene.add(cloud);
        if (this.gtOverlay && id !== GT_LAYER) {
            const overlay = new THREE.Points(this.gtOverlay.geometry, this.gtOverlay.material);
            overlay.userData.isOverlay = true;
            scene.add(overlay);
        }
        const color = id === GT_LAYER ? '#cbd5e1' : `hsl(${(this.colorCount++ * 67) % 360} 70% 55%)`;
        const panel = { id, label, color, data, detail, cloud, scene, visible: true };
        this.panels.push(panel);
        this.recolor(panel);
    }

    recolor(panel) {
        const attribute = panel.cloud.geometry.getAttribute('color');
        const target = attribute.array;
        const { colors, error } = panel.data;
        const count = panel.data.points.length / 3;
        const vmax = Number(this.vmaxInput.value);
        if (this.colorMode === 'error' && error) {
            for (let i = 0; i < count; i += 1) {
                const [r, g, b] = turbo(error[i] / vmax);
                target[3 * i] = r; target[3 * i + 1] = g; target[3 * i + 2] = b;
            }
        } else if (colors) {
            for (let i = 0; i < 3 * count; i += 1) target[i] = colors[i] / 255;
        } else {
            const { points } = panel.data;
            const up = UP_AXES[this.upAxis];
            let low = Infinity;
            let high = -Infinity;
            for (let i = 0; i < count; i += 1) {
                const height = points[3 * i] * up.x + points[3 * i + 1] * up.y + points[3 * i + 2] * up.z;
                low = Math.min(low, height);
                high = Math.max(high, height);
            }
            const span = Math.max(high - low, 1e-9);
            for (let i = 0; i < count; i += 1) {
                const height = points[3 * i] * up.x + points[3 * i + 1] * up.y + points[3 * i + 2] * up.z;
                const [r, g, b] = heightColor((height - low) / span);
                target[3 * i] = r; target[3 * i + 1] = g; target[3 * i + 2] = b;
            }
        }
        attribute.needsUpdate = true;
    }

    recolorAll() {
        this.root.querySelectorAll('[data-color]').forEach(button => button.classList.toggle('active', button.dataset.color === this.colorMode));
        this.root.querySelector('[data-role="vmax-label"]').hidden = this.colorMode !== 'error';
        this.root.querySelector('[data-role="vmax-value"]').textContent = `${formatNumber(Number(this.vmaxInput.value))}`;
        for (const panel of this.panels) this.recolor(panel);
        this.requestRender();
    }

    // ---- cameras -----------------------------------------------------------------------------------------------

    // Frustum lines of a cameras.json pose list (pred or gt), each image's own aspect and field of view.
    cameraObject(cameras, poses, color, depth) {
        const values = [];
        poses.forEach((pose, index) => {
            const [width, height] = cameras.sizes[index];
            const [fx, fy] = cameras.intrinsics[index];
            pushFrustumSegments(values, pose.slice(0, 3), pose.slice(3), width / (2 * fx), height / (2 * fy), depth);
        });
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(values, 3));
        return new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: new THREE.Color(color) }));
    }

    // (Re)build every window's frustums: its own cameras, and the GT cameras for the GT overlay.
    rebuildCameras() {
        for (const panel of this.panels) {
            for (const child of [...panel.scene.children]) {
                if (child.userData.isCameras || child.userData.isGtCameras) {
                    panel.scene.remove(child);
                    child.geometry.dispose();
                    child.material.dispose();
                }
            }
        }
        if (this.showCameras && this.manifest?.bounds) {
            const [min, max] = this.manifest.bounds;
            const radius = Math.max(Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2, 1e-3);
            const depth = radius * CAMERA_DEPTH * this.cameraScale;
            const gtInk = isLightColor(this.background) ? '#334155' : '#e2e8f0';
            const gtSource = this.panels.find(panel => panel.data.cameras?.gt)?.data.cameras;
            for (const panel of this.panels) {
                const cameras = panel.id === GT_LAYER ? gtSource : panel.data.cameras;
                const poses = panel.id === GT_LAYER ? cameras?.gt : cameras?.pred;
                if (poses) {
                    const object = this.cameraObject(cameras, poses, panel.id === GT_LAYER ? gtInk : this.cameraColor, depth);
                    object.userData.isCameras = true;
                    panel.scene.add(object);
                }
                if (panel.id !== GT_LAYER && gtSource) {
                    const object = this.cameraObject(gtSource, gtSource.gt, gtInk, depth);
                    object.userData.isGtCameras = true;
                    panel.scene.add(object);
                }
            }
        }
        const missing = this.panels.filter(panel => panel.id !== GT_LAYER && !panel.data.cameras).map(panel => panel.label);
        const note = this.root.querySelector('[data-role="camera-note"]');
        note.hidden = !this.showCameras || !missing.length;
        note.textContent = `No saved cameras yet for ${missing.join(', ')} (evaluated before cameras were saved).`;
        this.requestRender();
    }

    // Hide each method's lowest-confidence points (its own percentile); windows without confidences are listed.
    applyConfidence() {
        const percent = this.confidencePercent;
        this.root.querySelector('[data-role="confidence-value"]').textContent = `${percent}%`;
        const methods = this.panels.filter(panel => panel.id !== GT_LAYER);
        for (const panel of methods) {
            if (panel.data.confidenceUniform) {
                panel.data.confidenceUniform.value = confidenceThreshold(panel.data.confidence, percent);
            }
        }
        const missing = methods.filter(panel => !panel.data.confidence).map(panel => panel.label);
        const note = this.root.querySelector('[data-role="confidence-note"]');
        note.hidden = !percent || !missing.length;
        note.textContent = `No saved confidences for ${missing.join(', ')}: all their points stay shown.`;
        this.requestRender();
    }

    // Show or hide one window (the GT window too: its checkbox and its layer row stay in step).
    setPanelVisible(id, visible) {
        const panel = this.panels.find(item => item.id === id);
        if (panel) panel.visible = visible;
        if (id === GT_LAYER) this.root.querySelector('[data-role="gt-window"]').checked = visible;
        this.layoutViewports();
        this.renderLayers();
        this.requestRender();
    }

    removePanel(id) {
        if (id === GT_LAYER) {
            this.setPanelVisible(GT_LAYER, false);  // the GT stays loaded: method windows overlay it
            return;
        }
        const panel = this.panels.find(item => item.id === id);
        if (!panel) return;
        panel.cloud.geometry.dispose();
        this.panels = this.panels.filter(item => item !== panel);
        if (this.panels.length <= 1) this.clear();
        this.afterLayersChanged(false);
    }

    clear() {
        for (const panel of this.panels) panel.cloud.geometry.dispose();
        this.gtOverlay?.geometry.dispose();
        this.panels = [];
        this.colorCount = 1;
        this.gtOverlay = null;
        this.scene = null;
        this.manifest = null;
    }

    afterLayersChanged(refit) {
        this.layoutViewports();
        this.renderLayers();
        this.recolorAll();
        if (refit) this.fitView();
        this.rebuildCameras();
        this.applyConfidence();
        this.writeState();
    }

    writeState() {
        // ``methods`` keeps the windows' order; ``gt`` is the GT window's place among them when it is not first.
        const gt = this.panels.findIndex(panel => panel.id === GT_LAYER);
        writeCompareState(this.scene ? {
            compare: 'reconstruction', task: 'MVRecon', dataset: this.scene.dataset_key, scene: this.scene.scene,
            methods: this.panels.filter(panel => panel.id !== GT_LAYER).map(panel => panel.id),
            gt: gt > 0 ? gt : null, color: this.colorMode,
        } : { compare: 'reconstruction' });
    }

    renderLayers() {
        const container = this.root.querySelector('[data-role="layers"]');
        const layers = this.panels.map(panel => ({
            id: panel.id, label: panel.label, visible: panel.visible, removable: panel.id !== GT_LAYER || panel.visible,
            color: panel.color, detail: panel.detail,
        }));
        const sceneLabel = this.scene
            ? `${taskLabel('MVRecon')} · ${datasetLabel(this.scene.dataset_key)} · ${this.scene.scene}` : '';
        renderLayerList(container, {
            title: 'Viewer windows', sceneLabel, layers, addLabel: 'Add reconstructions',
            onToggle: id => {
                const panel = this.panels.find(item => item.id === id);
                if (panel) this.setPanelVisible(id, !panel.visible);
            },
            onRemove: id => this.removePanel(id),
            onAdd: () => this.openAdd(),
            onMove: (id, delta) => {
                const at = this.panels.findIndex(panel => panel.id === id);
                const neighbour = this.panels[at + delta];
                if (neighbour) this.movePanelTo(id, neighbour.id);
            },
        });
        const error = this.manifest?.error;
        this.root.querySelector('[data-role="readout"]').querySelector('.recon-legend')?.remove();
        if (error) {
            this.root.querySelector('[data-role="readout"]').insertAdjacentHTML('afterbegin', `
                <span class="recon-legend depth-legend-item">
                    <span class="compare-muted">Error (distance to GT)</span> 0
                    <span class="depth-legend-bar" style="background:linear-gradient(90deg,#30123b,#4662d7,#36aaf9,#1ae4b6,#72fe5e,#c8ef34,#faba39,#f66b19,#ca2a04,#7a0403)"></span>
                    error max · metric threshold ${formatNumber(error.eval_threshold)}
                </span>
            `);
        }
    }

    // ---- camera -----------------------------------------------------------------------------------------------

    fitView(axis = null) {
        const bounds = this.manifest?.bounds;
        if (!bounds) return;
        const min = new THREE.Vector3(...bounds[0]);
        const max = new THREE.Vector3(...bounds[1]);
        const center = min.clone().add(max).multiplyScalar(0.5);
        const radius = Math.max(max.clone().sub(min).length() / 2, 1e-3);
        const up = UP_AXES[this.upAxis];
        this.camera.up.copy(up);
        let direction;
        if (axis === 'x') direction = new THREE.Vector3(1, 0, 0);
        else if (axis === 'y') direction = new THREE.Vector3(0, 1, 0);
        else if (axis === 'z') direction = new THREE.Vector3(0, 0, 1);
        else {
            // A three-quarter view from "above" along the chosen up axis.
            const side = Math.abs(up.x) > 0.5 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
            direction = side.clone().add(up.clone().multiplyScalar(0.6)).add(side.clone().cross(up).multiplyScalar(-0.8)).normalize();
        }
        if (Math.abs(direction.dot(up)) > 0.99) this.camera.up.copy(Math.abs(up.y) > 0.5 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(0, 1, 0));
        const distance = radius / Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) * 1.1;
        this.camera.position.copy(center).add(direction.multiplyScalar(distance));
        this.camera.near = radius / 1000;
        this.camera.far = radius * 1000;
        this.camera.updateProjectionMatrix();
        this.camera.lookAt(center);
        this.createControls();
        this.controls.target.copy(center);
        this.controls.update();
        this.requestRender();
    }

    handleKey(event) {
        const target = this.controls.target;
        const offset = this.camera.position.clone().sub(target);
        // Turns the view (camera.up with it), so the arrow keys go over the poles too, like a drag.
        const rotate = (axisVector, angle) => {
            const axis = axisVector.clone();
            offset.applyAxisAngle(axis, angle);
            this.camera.up.applyAxisAngle(axis, angle);
            this.camera.position.copy(target).add(offset);
            this.camera.lookAt(target);
        };
        const right = new THREE.Vector3().crossVectors(this.camera.up, offset).normalize();
        const handled = {
            ArrowLeft: () => rotate(this.camera.up, -KEY_ORBIT_RADIANS),
            ArrowRight: () => rotate(this.camera.up, KEY_ORBIT_RADIANS),
            ArrowUp: () => rotate(right, -KEY_ORBIT_RADIANS),
            ArrowDown: () => rotate(right, KEY_ORBIT_RADIANS),
            '+': () => { this.camera.position.copy(target).add(offset.multiplyScalar(0.85)); },
            '=': () => { this.camera.position.copy(target).add(offset.multiplyScalar(0.85)); },
            '-': () => { this.camera.position.copy(target).add(offset.multiplyScalar(1 / 0.85)); },
            f: () => this.fitView(),
            1: () => this.fitView('x'),
            2: () => this.fitView('y'),
            3: () => this.fitView('z'),
        }[event.key];
        if (!handled) return;
        event.preventDefault();
        handled();
        this.controls.update();
        this.requestRender();
    }

    async restore(state) {
        if (!state.dataset || !state.scene) return;
        const index = await this.loadIndex();
        const scene = index.scenes.find(item => item.dataset_key === state.dataset && item.scene === state.scene);
        if (!scene) return;
        if (state.color === 'error') this.colorMode = 'error';
        await this.addMethods(scene, state.methods || []);
        const gt = Number(state.gt);
        const at = this.panels.findIndex(panel => panel.id === GT_LAYER);
        if (gt > 0 && at >= 0 && moveItem(this.panels, at, gt)) {
            this.layoutViewports();
            this.renderLayers();
            this.requestRender();
            this.writeState();
        }
    }
}
