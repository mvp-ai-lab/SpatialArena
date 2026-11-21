// Copyright (c) 2026 Yu Chen. All rights reserved.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

// Trajectory comparison (SfM / Weak-Texture SfM / SLAM): the GT and several methods' aligned trajectories in one
// 3D view. Trajectories come from the released assets (artifacts/), aligned exactly as the ATE.
// Cameras show as trajectories, frustums or both; error lines link each predicted camera center to its GT center (the
// translation error), at the current frame or for every camera. The pose AUC chart plots each method's relative-pose
// accuracy against the angle threshold, computed as the SfM evaluator does (sparse_reconstruction_evaluator.py).

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TrackballControls } from 'three/addons/controls/TrackballControls.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import {
    ASSET_ROOT, GT_COLOR, INDEX_ROOT, backgroundControlHtml, bindBackgroundControl, bindFullscreen, bindWheelShortcuts,
    datasetLabel, escapeHtml, fetchAssetJson, formatNumber, fullscreenButtonHtml, isLightColor, loadBackground,
    methodLabel, nextSeriesColor, openAddDialog, pushFrustumSegments, renderLayerList, taskLabel, writeCompareState,
} from './compare_common.js';
import { AUC_CURVE_MAX, AUC_THRESHOLDS, poseAuc, relativePoseErrors } from './pose_auc.js';

const GT_INK_ON_LIGHT = '#334155';
// Frustums: at most this many per trajectory (every k-th camera), a 60 degree 4:3 image (trajectories carry no
// intrinsics), drawn FRUSTUM_DEPTH of the scene radius deep at size 1.
const FRUSTUM_LIMIT = 400;
const FRUSTUM_HALF_WIDTH = Math.tan(Math.PI / 6);
const FRUSTUM_HALF_HEIGHT = FRUSTUM_HALF_WIDTH * 0.75;
const FRUSTUM_DEPTH = 0.035;
// How a newly opened scene shows its cameras: SfM as frustums (a few hundred unordered views), SLAM as trajectories
// (long ordered sequences); the visitor can switch afterwards.
const DEFAULT_DISPLAY = { SfM: 'frustums', WeakTextureSfM: 'frustums', SLAM: 'trajectory' };
// Axis views look along an axis from its positive side, with a fixed screen "up".
const AXIS_VIEWS = {
    x: { direction: new THREE.Vector3(1, 0, 0), up: new THREE.Vector3(0, 0, 1) },
    y: { direction: new THREE.Vector3(0, 1, 0), up: new THREE.Vector3(0, 0, 1) },
    z: { direction: new THREE.Vector3(0, 0, 1), up: new THREE.Vector3(0, 1, 0) },
};

// Segments joining consecutive frames that have a position, across the frames without one (a method that returns
// keyframe poses only, e.g. VGGT-SLAM, or a GT shown on the predicted frames only, is still drawn as a path).
function segmentPositions(positions) {
    const values = [];
    let previous = null;
    for (const position of positions) {
        if (!position) continue;
        if (previous) values.push(previous[0], previous[1], previous[2], position[0], position[1], position[2]);
        previous = position;
    }
    return values;
}

export class TrajectoryCompare {
    constructor(root) {
        this.root = root;
        this.layers = [];  // {id: method, method, color, visible, data, object, marker}
        this.scene = null;  // index entry of the scene in the viewer
        this.gt = null;  // {object, marker, data}
        this.frame = 0;
        this.viewMode = 'free';
        this.lineWidth = 2.5;
        this.playing = false;
        this.display = 'trajectory';  // trajectory | frustums | both
        this.frustumScale = 1;
        this.background = loadBackground('trajectory');
        this.buildDom();
        this.initThree();
    }

    buildDom() {
        this.root.innerHTML = `
            <div class="compare-toolbar">
                <div class="compare-segmented" data-role="views">
                    <button type="button" data-view="free" class="active">Free</button>
                    <button type="button" data-view="z" title="Look down the Z axis (top view)">Z</button>
                    <button type="button" data-view="y" title="Look along the Y axis">Y</button>
                    <button type="button" data-view="x" title="Look along the X axis">X</button>
                </div>
                <button type="button" class="compare-button" data-action="fit"><i class="fas fa-expand"></i> Fit</button>
                <label class="compare-inline">Line
                    <input type="range" data-role="width" min="1" max="8" step="0.5" value="${this.lineWidth}">
                </label>
                <label class="compare-inline" title="Lines linking each predicted camera center to its GT center (the translation error)">Error lines
                    <select data-role="errors">
                        <option value="current" selected>Current frame</option>
                        <option value="all">All cameras</option>
                        <option value="off">Off</option>
                    </select>
                </label>
                <label class="compare-inline"><input type="checkbox" data-role="image" checked> Image</label>
                ${fullscreenButtonHtml()}
            </div>
            <div class="compare-toolbar">
                <div class="compare-segmented" data-role="display">
                    <button type="button" data-display="trajectory" class="active">Trajectory</button>
                    <button type="button" data-display="frustums">Frustums</button>
                    <button type="button" data-display="both">Both</button>
                </div>
                <label class="compare-inline" data-role="frustum-size-label" title="Alt + wheel over the view">Frustum size
                    <input type="range" data-role="frustum-size" min="0.25" max="4" step="0.25" value="${this.frustumScale}">
                </label>
                <label class="compare-inline" title="Relative pose accuracy vs. angle threshold, as the SfM evaluator's AUC">
                    <input type="checkbox" data-role="auc"> Pose AUC curves
                </label>
                ${backgroundControlHtml(this.background)}
                <div class="compare-frame-control">
                    <button type="button" class="compare-icon-button" data-action="play" title="Play"><i class="fas fa-play"></i></button>
                    <input type="range" data-role="frame" min="0" max="0" value="0">
                    <span data-role="frame-label" class="compare-muted">-</span>
                </div>
            </div>
            <div class="compare-stage">
                <div class="compare-canvas-wrap"><canvas class="compare-canvas"></canvas>
                    <figure class="trajectory-inset" data-role="inset" hidden>
                        <img alt="Input image of the selected frame" data-role="inset-image">
                        <figcaption data-role="inset-caption"></figcaption>
                    </figure>
                    <div class="compare-canvas-hint" data-role="hint">Add trajectories to compare methods.</div>
                </div>
                <aside class="compare-layers" data-role="layers"></aside>
            </div>
            <div class="trajectory-auc" data-role="auc-panel" hidden></div>
            <div class="compare-readout" data-role="readout"></div>
        `;
        this.canvas = this.root.querySelector('canvas');
        this.wrap = this.root.querySelector('.compare-canvas-wrap');
        this.frameInput = this.root.querySelector('[data-role="frame"]');
        this.root.querySelector('[data-role="views"]').addEventListener('click', event => {
            const view = event.target.closest('[data-view]')?.dataset.view;
            if (view) this.setViewMode(view);
        });
        this.root.querySelector('[data-action="fit"]').addEventListener('click', () => this.fitView());
        this.root.querySelector('[data-action="play"]').addEventListener('click', () => this.togglePlay());
        this.root.querySelector('[data-role="width"]').addEventListener('input', event => {
            this.lineWidth = Number(event.target.value);
            this.forEachLine(material => { material.linewidth = this.lineWidth; });
            this.requestRender();
        });
        this.root.querySelector('[data-role="errors"]').addEventListener('change', () => this.rebuildErrorLines());
        this.root.querySelector('[data-role="display"]').addEventListener('click', event => {
            const display = event.target.closest('[data-display]')?.dataset.display;
            if (display) this.setDisplay(display);
        });
        this.root.querySelector('[data-role="frustum-size"]').addEventListener('input', event => {
            this.frustumScale = Number(event.target.value);
            this.rebuildFrustums();
        });
        this.root.querySelector('[data-role="auc"]').addEventListener('change', () => this.renderAuc());
        bindBackgroundControl(this.root, 'trajectory', color => this.applyBackground(color));
        this.root.querySelector('[data-role="image"]').addEventListener('change', () => this.updateInset());
        this.insetImage = this.root.querySelector('[data-role="inset-image"]');
        this.insetImage.addEventListener('error', () => {
            this.root.querySelector('[data-role="inset-caption"]').textContent += ' · image unavailable';
        });
        this.frameInput.addEventListener('input', () => this.setFrame(Number(this.frameInput.value)));
        bindFullscreen(this.root, this.root);
        bindWheelShortcuts(this.wrap, { alt: () => this.root.querySelector('[data-role="frustum-size"]') });
        this.renderLayers();
    }

    initThree() {
        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, preserveDrawingBuffer: true });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        this.three = new THREE.Scene();
        this.three.background = new THREE.Color(this.background);
        this.perspective = new THREE.PerspectiveCamera(50, 1, 0.001, 1e6);
        this.perspective.up.set(0, 0, 1);
        this.orthographic = new THREE.OrthographicCamera(-1, 1, 1, -1, -1e6, 1e6);
        this.camera = this.perspective;
        // Free view: a trackball, which turns the view about any axis (over the poles, and rolling it), unlike an
        // orbit control that keeps +Z up. Axis views: an orbit control with rotation off (pan and zoom only).
        this.trackball = new TrackballControls(this.perspective, this.canvas);
        Object.assign(this.trackball, { rotateSpeed: 2.5, zoomSpeed: 1.2, panSpeed: 0.8, dynamicDampingFactor: 0.15 });
        // No A/S/D modifier keys: typing elsewhere on the page must not change what a drag does.
        this.trackball.keys = ['', '', ''];
        this.orbit = new OrbitControls(this.orthographic, this.canvas);
        Object.assign(this.orbit, { enableDamping: true, enableRotate: false, enabled: false });
        this.controls = this.trackball;
        this.group = new THREE.Group();
        this.three.add(this.group);
        // Translation error lines in each method's color: the current frame's (redrawn per frame) and every
        // camera's (rebuilt when the layers or the mode change).
        this.errorLines = new THREE.LineSegments(
            new THREE.BufferGeometry(),
            new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95 }),
        );
        this.allErrorLines = new THREE.LineSegments(
            new THREE.BufferGeometry(),
            new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55 }),
        );
        this.three.add(this.errorLines, this.allErrorLines);
        // The selected frame's frustum of every visible trajectory, drawn thick in its color.
        this.currentFrustums = new LineSegments2(new LineSegmentsGeometry(), new LineMaterial({
            vertexColors: true, linewidth: this.lineWidth + 1,
            resolution: new THREE.Vector2(this.wrap.clientWidth, this.wrap.clientHeight),
        }));
        this.currentFrustums.visible = false;
        this.three.add(this.currentFrustums);
        this.axes = new THREE.AxesHelper(1);
        this.three.add(this.axes);
        // Render on demand (camera moves, state changes), not every animation frame.
        for (const controls of [this.trackball, this.orbit]) {
            controls.addEventListener('change', () => this.requestRender());
        }
        // The trackball moves the camera only in update(): keep rendering (and updating) while a drag lasts.
        this.trackball.addEventListener('start', () => { this.dragging = true; this.requestRender(); });
        this.trackball.addEventListener('end', () => { this.dragging = false; this.requestRender(); });
        this.resizeObserver = new ResizeObserver(() => this.resize());
        this.resizeObserver.observe(this.wrap);
        this.resize();
    }

    requestRender() {
        if (this.renderQueued || this.disposed) return;
        this.renderQueued = true;
        this.animationId = requestAnimationFrame(() => {
            this.renderQueued = false;
            // The orbit's update() returns true while damping still moves the camera; the trackball's reports its
            // damped motion through 'change' events. Keep rendering until it settles.
            if (this.controls.update() || this.dragging) this.requestRender();
            this.renderer.render(this.three, this.camera);
        });
    }

    dispose() {
        this.disposed = true;
        cancelAnimationFrame(this.animationId);
        clearInterval(this.playTimer);
        this.resizeObserver.disconnect();
        this.trackball.dispose();
        this.orbit.dispose();
        this.renderer.dispose();
    }

    resize() {
        const width = Math.max(1, this.wrap.clientWidth);
        const height = Math.max(1, this.wrap.clientHeight);
        this.renderer.setSize(width, height, false);
        this.perspective.aspect = width / height;
        this.perspective.updateProjectionMatrix();
        this.updateOrthographicFrustum();
        this.trackball.handleResize();
        this.forEachLine(material => material.resolution.set(width, height));
        this.requestRender();
    }

    forEachLine(callback) {
        for (const item of [this.gt, ...this.layers]) {
            if (item?.object) callback(item.object.material);
        }
        if (this.currentFrustums) callback(this.currentFrustums.material);
    }

    gtColor() {
        return isLightColor(this.background) ? GT_INK_ON_LIGHT : GT_COLOR;
    }

    applyBackground(color) {
        this.background = color;
        this.three.background = new THREE.Color(color);
        if (this.gt) {
            this.gt.object.material.color.set(this.gtColor());
            this.gt.marker.material.color.set(this.gtColor());
            if (this.gt.frustums) this.gt.frustums.material.color.set(this.gtColor());
        }
        this.renderLayers();
        this.updateFrame();
        this.renderAuc();
    }

    // ---- frustums and display ------------------------------------------------------------------------------------

    setDisplay(display) {
        this.display = display;
        this.root.querySelectorAll('[data-display]').forEach(button => {
            button.classList.toggle('active', button.dataset.display === display);
        });
        this.rebuildFrustums();
    }

    frustumDepth() {
        const box = this.bounds();
        const radius = box.isEmpty() ? 1 : Math.max(box.getSize(new THREE.Vector3()).length() / 2, 1e-3);
        return radius * FRUSTUM_DEPTH * this.frustumScale;
    }

    frustumObject(positions, quaternions, color, depth) {
        const frames = positions.map((position, index) => (position && quaternions[index] ? index : -1)).filter(index => index >= 0);
        const stride = Math.max(1, Math.ceil(frames.length / FRUSTUM_LIMIT));
        const values = [];
        for (let k = 0; k < frames.length; k += stride) {
            const frame = frames[k];
            pushFrustumSegments(values, positions[frame], quaternions[frame], FRUSTUM_HALF_WIDTH, FRUSTUM_HALF_HEIGHT, depth);
        }
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(values, 3));
        const object = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: new THREE.Color(color) }));
        object.userData.stride = stride;
        return object;
    }

    // ---- GT shown --------------------------------------------------------------------------------------------------

    // Index of a frame name in a layer's own frame list.
    layerIndexOfName(layer, name) {
        layer.nameIndex = layer.nameIndex || new Map(layer.data.names.map((value, index) => [value, index]));
        return layer.nameIndex.get(name);
    }

    // The GT positions drawn: only the frames at least one visible method has a pose for (a method that registered
    // part of the images, or lost tracking, is compared on its own frames), all of them when every method is hidden.
    gtShown() {
        const { names } = this.gt.data;
        const positions = this.gt.data.gt.positions;
        const visible = this.layers.filter(layer => layer.visible);
        if (!visible.length) return positions;
        return positions.map((position, index) => (visible.some(layer => {
            const at = this.layerIndexOfName(layer, names[index]);
            return at !== undefined && layer.data.pred.positions[at];
        }) ? position : null));
    }

    // The GT trajectory line through the shown frames (consecutive shown frames are joined across the others).
    refreshGtLine() {
        if (!this.gt) return;
        const shown = this.gtShown();
        this.gt.shown = shown;
        this.gt.shownCount = shown.filter(Boolean).length;
        this.group.remove(this.gt.object);
        this.gt.object.geometry.dispose();
        this.gt.object = this.lineObject(shown, this.gtColor());
        this.group.add(this.gt.object);
    }

    // Rebuild every trajectory's frustums (display, size or the trajectories changed).
    rebuildFrustums() {
        this.refreshGtLine();
        const items = [this.gt, ...this.layers].filter(Boolean);
        for (const item of items) {
            if (item.frustums) {
                this.group.remove(item.frustums);
                item.frustums.geometry.dispose();
                item.frustums.material.dispose();
                item.frustums = null;
            }
        }
        if (this.display !== 'trajectory' && this.gt) {
            const depth = this.frustumDepth();
            this.gt.frustums = this.frustumObject(this.gtShown(), this.gt.data.gt.quaternions, this.gtColor(), depth);
            this.group.add(this.gt.frustums);
            for (const layer of this.layers) {
                layer.frustums = this.frustumObject(layer.data.pred.positions, layer.data.pred.quaternions, layer.color, depth);
                this.group.add(layer.frustums);
            }
        }
        this.applyVisibility();
    }

    updateCurrentFrustums() {
        const values = [];
        const colors = [];
        if (this.display !== 'trajectory' && this.gt) {
            const depth = this.frustumDepth() * 1.15;
            const add = (position, quaternion, color) => {
                if (!position || !quaternion) return;
                const before = values.length;
                pushFrustumSegments(values, position, quaternion, FRUSTUM_HALF_WIDTH, FRUSTUM_HALF_HEIGHT, depth);
                const rgb = new THREE.Color(color);
                for (let i = before; i < values.length; i += 3) colors.push(rgb.r, rgb.g, rgb.b);
            };
            const gtPosition = (this.gt.shown || this.gt.data.gt.positions)[this.frame];
            if (!this.gtHidden) add(gtPosition, this.gt.data.gt.quaternions[this.frame], this.gtColor());
            for (const layer of this.layers) {
                const index = this.layerFrameIndex(layer);
                if (layer.visible && index !== undefined) {
                    add(layer.data.pred.positions[index], layer.data.pred.quaternions[index], layer.color);
                }
            }
        }
        this.currentFrustums.visible = values.length > 0;
        if (!values.length) return;
        this.currentFrustums.geometry.dispose();
        this.currentFrustums.geometry = new LineSegmentsGeometry();
        this.currentFrustums.geometry.setPositions(values);
        this.currentFrustums.geometry.setColors(colors);
        this.currentFrustums.material.linewidth = this.lineWidth + 1;
    }

    // ---- pose AUC -----------------------------------------------------------------------------------------------

    renderAuc() {
        const panel = this.root.querySelector('[data-role="auc-panel"]');
        const layers = this.layers.filter(layer => layer.visible);
        panel.hidden = !this.root.querySelector('[data-role="auc"]').checked || !layers.length;
        if (panel.hidden) return;
        for (const layer of layers) layer.poseErrors = layer.poseErrors || relativePoseErrors(layer.data);
        const width = 640;
        const height = 260;
        const margin = { left: 46, right: 14, top: 12, bottom: 36 };
        const plotWidth = width - margin.left - margin.right;
        const plotHeight = height - margin.top - margin.bottom;
        const x = value => margin.left + (value / AUC_CURVE_MAX) * plotWidth;
        const y = value => margin.top + (1 - value) * plotHeight;
        const grid = [];
        for (let t = 0; t <= AUC_CURVE_MAX; t += 5) {
            grid.push(`<line x1="${x(t)}" x2="${x(t)}" y1="${y(0)}" y2="${y(1)}" class="auc-grid"/>`,
                `<text x="${x(t)}" y="${y(0) + 16}" text-anchor="middle">${t}</text>`);
        }
        for (let v = 0; v <= 1.0001; v += 0.2) {
            grid.push(`<line x1="${x(0)}" x2="${x(AUC_CURVE_MAX)}" y1="${y(v)}" y2="${y(v)}" class="auc-grid"/>`,
                `<text x="${x(0) - 6}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(1)}</text>`);
        }
        const curves = layers.map(layer => {
            const { errors } = layer.poseErrors;
            if (!errors.length) return '';
            const points = [];
            let below = 0;
            for (let step = 0; step <= AUC_CURVE_MAX * 10; step += 1) {
                const threshold = step / 10;
                while (below < errors.length && errors[below] <= threshold) below += 1;
                points.push(`${x(threshold).toFixed(1)},${y(below / errors.length).toFixed(1)}`);
            }
            return `<polyline points="${points.join(' ')}" fill="none" stroke="${escapeHtml(layer.color)}" stroke-width="2"/>`;
        });
        const legend = layers.map(layer => {
            const { errors, pairs, totalPairs } = layer.poseErrors;
            const aucs = AUC_THRESHOLDS.map(threshold => `@${threshold}° <strong>${formatNumber(poseAuc(errors, threshold))}</strong>`);
            const sampled = pairs < totalPairs ? ` (${pairs} of ${totalPairs} pairs)` : ` (${pairs} pairs)`;
            return `<span class="compare-readout-item"><span class="compare-swatch" style="background:${escapeHtml(layer.color)}"></span>
                ${escapeHtml(methodLabel(layer.method))} AUC ${aucs.join(' · ')}<span class="compare-muted">${sampled}</span></span>`;
        });
        panel.innerHTML = `
            <div class="trajectory-auc-header"><strong>Pose accuracy</strong>
                <span class="compare-muted">fraction of frame pairs whose relative pose error max(rotation, translation
                direction) is below the threshold</span></div>
            <svg viewBox="0 0 ${width} ${height}" class="trajectory-auc-chart" role="img" aria-label="Pose accuracy curves">
                ${grid.join('')}
                ${curves.join('')}
                <text x="${margin.left + plotWidth / 2}" y="${height - 4}" text-anchor="middle">Error threshold (°)</text>
                <text transform="translate(12 ${margin.top + plotHeight / 2}) rotate(-90)" text-anchor="middle">Accuracy</text>
            </svg>
            <div class="trajectory-auc-legend">${legend.join('')}</div>
        `;
    }

    // ---- loading ---------------------------------------------------------------------------------------------

    async loadIndex() {
        this.index = this.index || await fetchAssetJson(`${INDEX_ROOT}/trajectories.json`);
        return this.index;
    }

    async openAdd() {
        const index = await this.loadIndex();
        const result = await openAddDialog({
            title: 'Add trajectories',
            scenes: index.scenes,
            current: this.scene,
            loadedMethods: this.layers.map(layer => layer.method),
            describeMethod: entry => `ATE ${formatNumber(entry.ate_rmse)} m`,
        });
        if (result) await this.addMethods(result.scene, result.methods);
    }

    sameScene(scene) {
        return this.scene && scene && this.scene.task === scene.task
            && this.scene.dataset_key === scene.dataset_key && this.scene.scene === scene.scene;
    }

    async addMethods(scene, methods) {
        if (!this.sameScene(scene)) {
            this.clear();
            this.setDisplay(DEFAULT_DISPLAY[scene.task] || 'trajectory');
        }
        this.scene = scene;
        const hint = this.root.querySelector('[data-role="hint"]');
        hint.textContent = 'Loading trajectories...';
        hint.hidden = false;
        for (const method of methods) {
            if (this.layers.some(layer => layer.method === method)) continue;
            const entry = scene.methods.find(item => item.method === method);
            if (!entry) continue;
            try {
                const data = await fetchAssetJson(entry.path);
                this.addLayer(method, data, entry);
            } catch (error) {
                console.error(error);
                hint.textContent = `Could not load ${method}: ${error.message}`;
            }
        }
        hint.hidden = this.layers.length > 0;
        if (!this.layers.length) hint.textContent = 'Add trajectories to compare methods.';
        this.afterLayersChanged(true);
    }

    lineObject(positions, color, dashed = false) {
        const geometry = new LineSegmentsGeometry();
        geometry.setPositions(segmentPositions(positions));
        const material = new LineMaterial({
            color: new THREE.Color(color).getHex(),
            linewidth: this.lineWidth,
            dashed,
            dashSize: 0.05,
            gapSize: 0.03,
            resolution: new THREE.Vector2(this.wrap.clientWidth, this.wrap.clientHeight),
        });
        const line = new LineSegments2(geometry, material);
        if (dashed) line.computeLineDistances();
        return line;
    }

    marker(color) {
        const mesh = new THREE.Mesh(
            new THREE.SphereGeometry(1, 16, 12),
            new THREE.MeshBasicMaterial({ color: new THREE.Color(color), depthTest: false }),
        );
        mesh.renderOrder = 10;
        mesh.visible = false;
        this.group.add(mesh);
        return mesh;
    }

    addLayer(method, data, entry) {
        if (!this.gt) {
            // The GT of the first trajectory; every trajectory of a scene shares the same GT frames up to which
            // ones the method returned (SfM: the matched images), so the longest one is kept.
            this.gt = { data };
            this.gt.object = this.lineObject(data.gt.positions, this.gtColor());
            this.gt.marker = this.marker(this.gtColor());
            this.group.add(this.gt.object);
        } else if (data.gt.positions.length > this.gt.data.gt.positions.length) {
            this.group.remove(this.gt.object);
            this.gt.data = data;
            this.gt.object = this.lineObject(data.gt.positions, this.gtColor());
            this.group.add(this.gt.object);
        }
        const color = nextSeriesColor(this.layers);
        const object = this.lineObject(data.pred.positions, color);
        this.group.add(object);
        this.layers.push({ id: method, method, color, visible: true, data, entry, object, marker: this.marker(color) });
    }

    removeLayer(id) {
        const layer = this.layers.find(item => item.id === id);
        if (!layer) return;
        this.group.remove(layer.object, layer.marker);
        if (layer.frustums) this.group.remove(layer.frustums);
        layer.object.geometry.dispose();
        this.layers = this.layers.filter(item => item !== layer);
        if (!this.layers.length) this.clear();
        this.afterLayersChanged(false);
    }

    clear() {
        for (const layer of this.layers) this.group.remove(layer.object, layer.marker, ...(layer.frustums ? [layer.frustums] : []));
        if (this.gt) this.group.remove(this.gt.object, this.gt.marker, ...(this.gt.frustums ? [this.gt.frustums] : []));
        this.layers = [];
        this.gt = null;
        this.scene = null;
        this.frame = 0;
    }

    toggleLayer(id) {
        if (id === '__gt__') {
            this.gtHidden = !this.gtHidden;
        } else {
            const layer = this.layers.find(item => item.id === id);
            if (layer) layer.visible = !layer.visible;
        }
        this.rebuildFrustums();  // the GT follows the visible methods' frames; also applies the visibility
        this.renderLayers();
    }

    applyVisibility() {
        const lines = this.display !== 'frustums';
        if (this.gt) {
            this.gt.object.visible = !this.gtHidden && lines;
            if (this.gt.frustums) this.gt.frustums.visible = !this.gtHidden;
        }
        for (const layer of this.layers) {
            layer.object.visible = layer.visible && lines;
            if (layer.frustums) layer.frustums.visible = layer.visible;
        }
        this.root.querySelector('[data-role="frustum-size-label"]').hidden = this.display === 'trajectory';
        this.rebuildErrorLines();
        this.renderAuc();
    }

    errorMode() {
        return this.root.querySelector('[data-role="errors"]').value;
    }

    // Every camera's error line (mode "all"): predicted center to GT center, per visible method.
    rebuildErrorLines() {
        const positions = [];
        const colors = [];
        if (this.errorMode() === 'all') {
            for (const layer of this.layers) {
                if (!layer.visible) continue;
                const rgb = new THREE.Color(layer.color);
                const { pred, gt } = layer.data;
                pred.positions.forEach((position, index) => {
                    const target = gt.positions[index];
                    if (!position || !target) return;
                    positions.push(...position, ...target);
                    colors.push(rgb.r, rgb.g, rgb.b, rgb.r, rgb.g, rgb.b);
                });
            }
        }
        this.allErrorLines.geometry.dispose();
        this.allErrorLines.geometry = new THREE.BufferGeometry();
        this.allErrorLines.geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        this.allErrorLines.geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        this.allErrorLines.visible = positions.length > 0;
        this.updateFrame();
    }

    afterLayersChanged(refit) {
        const frames = this.gt ? this.gt.data.names.length : 0;
        this.frameInput.max = String(Math.max(0, frames - 1));
        this.frame = Math.min(this.frame, Math.max(0, frames - 1));
        this.frameInput.value = String(this.frame);
        if (refit) this.fitView();
        this.rebuildFrustums();  // sized by the scene; also applies the visibility
        this.renderLayers();
        writeCompareState(this.scene ? {
            compare: 'trajectory', task: this.scene.task, dataset: this.scene.dataset_key, scene: this.scene.scene,
            methods: this.layers.map(layer => layer.method),
        } : { compare: 'trajectory' });
    }

    renderLayers() {
        const container = this.root.querySelector('[data-role="layers"]');
        const gtFrames = this.gt && this.gt.shownCount !== undefined && this.gt.shownCount < this.gt.data.names.length
            ? `${this.gt.shownCount} of ${this.gt.data.names.length} frames (those with a predicted pose)`
            : `${this.gt?.data.names.length} frames`;
        const layers = this.gt ? [{ id: '__gt__', label: 'Ground truth', color: this.gtColor(), visible: !this.gtHidden,
            detail: gtFrames, removable: false }] : [];
        for (const layer of this.layers) {
            const coverage = layer.data.num_valid < layer.data.num_frames
                ? ` · ${layer.data.num_valid}/${layer.data.num_frames}` : '';
            layers.push({
                id: layer.id, label: methodLabel(layer.method), color: layer.color, visible: layer.visible,
                detail: `ATE ${formatNumber(layer.data.ate_rmse)} m${coverage}`,
            });
        }
        const sceneLabel = this.scene
            ? `${taskLabel(this.scene.task)} · ${datasetLabel(this.scene.dataset_key)} · ${this.scene.scene}` : '';
        renderLayerList(container, {
            title: 'Viewer layers',
            sceneLabel,
            layers,
            addLabel: 'Add trajectories',
            onToggle: id => this.toggleLayer(id),
            onRemove: id => this.removeLayer(id),
            onAdd: () => this.openAdd(),
        });
    }

    // ---- frame ------------------------------------------------------------------------------------------------

    // Frame index in a layer's own frame list (SfM layers may cover different image subsets than the GT shown).
    layerFrameIndex(layer) {
        if (layer.data === this.gt.data) return this.frame;
        return this.layerIndexOfName(layer, this.gt.data.names[this.frame]);
    }

    setFrame(frame) {
        this.frame = frame;
        this.updateFrame();
    }

    updateFrame() {
        const label = this.root.querySelector('[data-role="frame-label"]');
        const readout = this.root.querySelector('[data-role="readout"]');
        if (!this.gt) {
            label.textContent = '-';
            readout.innerHTML = '';
            this.errorLines.geometry.setAttribute('position', new THREE.Float32BufferAttribute([], 3));
            this.errorLines.geometry.setAttribute('color', new THREE.Float32BufferAttribute([], 3));
            this.currentFrustums.visible = false;
            this.requestRender();
            return;
        }
        const names = this.gt.data.names;
        label.textContent = `${this.frame + 1} / ${names.length} · ${names[this.frame]}`;
        const gtPosition = this.gt.data.gt.positions[this.frame];
        const markers = this.display === 'trajectory';
        this.gt.marker.position.fromArray(gtPosition);
        this.gt.marker.visible = !this.gtHidden && markers && Boolean((this.gt.shown || this.gt.data.gt.positions)[this.frame]);
        const showErrors = this.errorMode() === 'current';
        const errorPositions = [];
        const errorColors = [];
        const rows = [];
        for (const layer of this.layers) {
            const index = this.layerFrameIndex(layer);
            const position = index === undefined ? null : layer.data.pred.positions[index];
            layer.marker.visible = Boolean(position) && layer.visible && markers;
            if (position) {
                layer.marker.position.fromArray(position);
                if (showErrors && layer.visible) {
                    const rgb = new THREE.Color(layer.color);
                    errorPositions.push(...position, ...layer.data.gt.positions[index]);
                    errorColors.push(rgb.r, rgb.g, rgb.b, rgb.r, rgb.g, rgb.b);
                }
            }
            rows.push(`
                <span class="compare-readout-item">
                    <span class="compare-swatch" style="background:${escapeHtml(layer.color)}"></span>
                    ${escapeHtml(methodLabel(layer.method))}
                    <strong>${position ? `${formatNumber(layer.data.ate[index])} m` : 'no pose'}</strong>
                </span>
            `);
        }
        this.errorLines.geometry.setAttribute('position', new THREE.Float32BufferAttribute(errorPositions, 3));
        this.errorLines.geometry.setAttribute('color', new THREE.Float32BufferAttribute(errorColors, 3));
        this.updateCurrentFrustums();
        readout.innerHTML = rows.length
            ? `<span class="compare-muted">Position error at this frame:</span> ${rows.join('')}` : '';
        this.updateInset();
        this.requestRender();
    }

    // The input image of the selected camera, anchored top-left (thumbnails written with the trajectories).
    thumbnailUrl(name) {
        if (!this.scene?.thumbnail_dir || !name) return null;
        const encoded = name.split('/').map(encodeURIComponent).join('/');
        return `${ASSET_ROOT}/${this.scene.thumbnail_dir}/${encoded}.webp`;
    }

    updateInset() {
        const inset = this.root.querySelector('[data-role="inset"]');
        const name = this.gt?.data.names[this.frame];
        const url = this.thumbnailUrl(name);
        inset.hidden = !url || !this.root.querySelector('[data-role="image"]').checked;
        if (inset.hidden) return;
        this.root.querySelector('[data-role="inset-caption"]').textContent = `${this.frame + 1} · ${name}`;
        if (this.insetImage.getAttribute('src') !== url) this.insetImage.src = url;
        // Warm the cache for the next frames so playback does not wait on the network.
        this.preloaded = this.preloaded || new Set();
        for (let offset = 1; offset <= 8; offset += 1) {
            const next = this.thumbnailUrl(this.gt.data.names[this.frame + offset]);
            if (next && !this.preloaded.has(next)) {
                this.preloaded.add(next);
                new Image().src = next;
            }
        }
    }

    togglePlay() {
        this.playing = !this.playing;
        const icon = this.root.querySelector('[data-action="play"] i');
        icon.className = this.playing ? 'fas fa-pause' : 'fas fa-play';
        clearInterval(this.playTimer);
        if (!this.playing || !this.gt) return;
        const frames = this.gt.data.names.length;
        // About 20 s for a whole trajectory, at most 30 updates per second.
        const step = Math.max(1, Math.round(frames / 600));
        this.playTimer = setInterval(() => {
            const next = (this.frame + step) % frames;
            this.frameInput.value = String(next);
            this.setFrame(next);
        }, 33);
    }

    // ---- views ------------------------------------------------------------------------------------------------

    bounds() {
        const box = new THREE.Box3();
        const add = positions => positions.forEach(position => { if (position) box.expandByPoint(new THREE.Vector3(...position)); });
        if (this.gt) add(this.gtShown());
        for (const layer of this.layers) if (layer.visible) add(layer.data.pred.positions);
        return box;
    }

    setViewMode(mode) {
        this.viewMode = mode;
        this.root.querySelectorAll('[data-view]').forEach(button => {
            button.classList.toggle('active', button.dataset.view === mode);
        });
        const free = mode === 'free';
        this.camera = free ? this.perspective : this.orthographic;
        this.controls = free ? this.trackball : this.orbit;
        this.trackball.enabled = free;
        this.orbit.enabled = !free;
        this.dragging = false;
        this.fitView();
    }

    updateOrthographicFrustum() {
        const aspect = Math.max(1e-6, this.wrap.clientWidth / Math.max(1, this.wrap.clientHeight));
        const half = this.orthoHalfHeight || 1;
        Object.assign(this.orthographic, { left: -half * aspect, right: half * aspect, top: half, bottom: -half });
        this.orthographic.updateProjectionMatrix();
    }

    fitView() {
        const box = this.bounds();
        if (box.isEmpty()) return;
        const center = box.getCenter(new THREE.Vector3());
        const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 1e-3);
        const markerSize = radius * 0.02;
        for (const item of [this.gt, ...this.layers]) item?.marker?.scale.setScalar(markerSize);
        this.axes.scale.setScalar(radius * 0.15);
        this.axes.position.copy(box.min);
        this.controls.target.copy(center);
        if (this.viewMode === 'free') {
            this.perspective.up.set(0, 0, 1);  // the trackball rolls the camera; a fit starts upright again
            const offset = new THREE.Vector3(1, -1.2, 0.9).normalize().multiplyScalar(radius * 2.4);
            this.perspective.position.copy(center).add(offset);
            this.perspective.near = radius / 1000;
            this.perspective.far = radius * 1000;
            this.perspective.updateProjectionMatrix();
        } else {
            const { direction, up } = AXIS_VIEWS[this.viewMode];
            this.orthographic.up.copy(up);
            this.orthographic.position.copy(center).add(direction.clone().multiplyScalar(radius * 4));
            this.orthographic.zoom = 1;
            this.orthoHalfHeight = radius * 1.1;
            this.updateOrthographicFrustum();
        }
        this.camera.lookAt(center);
        this.controls.update();
        this.updateFrame();
    }

    // ---- URL state ----------------------------------------------------------------------------------------------

    async restore(state) {
        if (!state.task || !state.dataset || !state.scene) return;
        const index = await this.loadIndex();
        const scene = index.scenes.find(item => (
            item.task === state.task && item.dataset_key === state.dataset && item.scene === state.scene
        ));
        if (scene) await this.addMethods(scene, state.methods || []);
    }
}
