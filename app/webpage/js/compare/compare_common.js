// Copyright (c) 2026 Yu Chen. All rights reserved.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

// Shared pieces of the artifact viewer's comparison modes: asset fetching, method colors, the "add methods" dialog,
// the layer list and the URL state (#artifacts?compare=<mode>&...), so a comparison can be shared as a link.

// The released viewer assets, committed with the dashboard (python/export_release_artifacts.py): a few scenes per task
// with their best-ranked methods. The index lives in artifacts/index/, shared scene assets in artifacts/scenes/, each
// method's in artifacts/cases/; the files are those spatialarena/eval/viewer_assets.py writes, with capped frames and
// points.
export const ASSET_ROOT = 'artifacts';
export const INDEX_ROOT = 'index';

// Distinct, colorblind-aware series colors (Okabe-Ito order, then extras); the GT is drawn in a neutral color.
export const SERIES_COLORS = [
    '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7',
    '#8C6BB1', '#66C2A5', '#FC8D62', '#A6D854', '#E78AC3',
];
export const GT_COLOR = '#f5f5f5';

const jsonCache = new Map();

export async function fetchAssetJson(path) {
    if (!jsonCache.has(path)) {
        jsonCache.set(path, fetch(`${ASSET_ROOT}/${path}`).then(response => {
            if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
            return response.json();
        }).catch(error => {
            jsonCache.delete(path);
            throw error;
        }));
    }
    return jsonCache.get(path);
}

export function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, char => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]
    ));
}

export function methodLabel(method) {
    return window.benchmarkMethodLabel ? window.benchmarkMethodLabel(method) : method;
}

export function datasetLabel(datasetKey) {
    return window.benchmarkDatasetLabel ? window.benchmarkDatasetLabel(datasetKey) : datasetKey;
}

export function taskLabel(task) {
    return window.benchmarkTaskLabel ? window.benchmarkTaskLabel(task) : task;
}

export function formatNumber(value, digits = 3) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return '-';
    const number = Number(value);
    if (Math.abs(number) >= 100) return number.toFixed(1);
    if (Math.abs(number) >= 10) return number.toFixed(2);
    return number.toFixed(digits);
}

// The next color not used by any current layer.
export function nextSeriesColor(layers) {
    const used = new Set(layers.map(layer => layer.color));
    return SERIES_COLORS.find(color => !used.has(color)) || SERIES_COLORS[layers.length % SERIES_COLORS.length];
}

// ---- viewer background ----------------------------------------------------------------------------------------

// Each comparison mode's background color, remembered per viewer in the browser (a convenience, so it may be lost).
export const DEFAULT_BACKGROUND = '#111318';
const BACKGROUND_KEY = 'ffb.compare.background.';

export function loadBackground(mode) {
    try {
        const value = window.localStorage.getItem(BACKGROUND_KEY + mode);
        if (/^#[0-9a-f]{6}$/i.test(value || '')) return value;
    } catch (error) { /* storage unavailable: use the default */ }
    return DEFAULT_BACKGROUND;
}

export function backgroundControlHtml(value) {
    return `<label class="compare-inline" title="Viewer background color">Background
        <input type="color" class="compare-color-input" data-role="background" value="${escapeHtml(value)}"></label>`;
}

// Calls onChange(color) whenever the picker of `root` changes, and remembers the color for `mode`.
export function bindBackgroundControl(root, mode, onChange) {
    const input = root.querySelector('[data-role="background"]');
    input.addEventListener('input', () => {
        try { window.localStorage.setItem(BACKGROUND_KEY + mode, input.value); } catch (error) { /* not remembered */ }
        onChange(input.value);
    });
}

// A remembered color picker (e.g. the camera frustums' color), like the background: per viewer, in the browser.
const COLOR_KEY = 'ffb.compare.color.';

export function loadColor(key, fallback) {
    try {
        const value = window.localStorage.getItem(COLOR_KEY + key);
        if (/^#[0-9a-f]{6}$/i.test(value || '')) return value;
    } catch (error) { /* storage unavailable: use the default */ }
    return fallback;
}

export function colorControlHtml(role, label, value, title = '') {
    return `<label class="compare-inline" title="${escapeHtml(title)}">${escapeHtml(label)}
        <input type="color" class="compare-color-input" data-role="${escapeHtml(role)}" value="${escapeHtml(value)}"></label>`;
}

export function bindColorControl(root, role, key, onChange) {
    const input = root.querySelector(`[data-role="${role}"]`);
    input.addEventListener('input', () => {
        try { window.localStorage.setItem(COLOR_KEY + key, input.value); } catch (error) { /* not remembered */ }
        onChange(input.value);
    });
}

// ---- full screen --------------------------------------------------------------------------------------------

export function fullscreenButtonHtml() {
    return `<button type="button" class="compare-button" data-action="fullscreen" title="Full screen (Esc to leave)">
        <i class="fas fa-maximize"></i> Full screen</button>`;
}

// The button of `root` puts `target` (the whole mode: toolbars, viewer and layers) in full screen and back; the
// viewer's ResizeObserver resizes its canvas. onChange(isFullscreen) runs after each switch.
export function bindFullscreen(root, target, onChange = () => {}) {
    const button = root.querySelector('[data-action="fullscreen"]');
    if (!button) return;
    if (!target.requestFullscreen) {
        button.hidden = true;
        return;
    }
    button.addEventListener('click', () => {
        if (document.fullscreenElement === target) document.exitFullscreen();
        else target.requestFullscreen().catch(error => console.error(error));
    });
    document.addEventListener('fullscreenchange', () => {
        const active = document.fullscreenElement === target;
        button.innerHTML = active
            ? '<i class="fas fa-minimize"></i> Exit full screen'
            : '<i class="fas fa-maximize"></i> Full screen';
        onChange(active);
    });
}

// ---- wheel shortcuts ----------------------------------------------------------------------------------------

// Alt+wheel and Ctrl+wheel over `element` step a range input (frustum size, point size) instead of zooming: one
// step per wheel notch, through the input's own 'input' handler. Captured before OrbitControls sees the wheel, and
// prevented so the browser neither zooms the page (Ctrl) nor navigates (Alt).
export function bindWheelShortcuts(element, { alt = null, ctrl = null } = {}) {
    element.addEventListener('wheel', event => {
        const input = event.ctrlKey ? ctrl?.() : (event.altKey ? alt?.() : null);
        if (!input) return;
        event.preventDefault();
        event.stopImmediatePropagation();
        const delta = event.deltaY || event.deltaX;
        if (!delta || input.disabled) return;
        const step = Number(input.step) || 1;
        const next = Number(input.value) + (delta < 0 ? step : -step);
        const clamped = Math.min(Number(input.max), Math.max(Number(input.min), next));
        if (clamped === Number(input.value)) return;
        input.value = String(clamped);
        input.dispatchEvent(new Event('input', { bubbles: true }));
    }, { capture: true, passive: false });
}

// ---- confidence filter --------------------------------------------------------------------------------------

// Predicted confidence (confidence.f32, one value per point; NaN where unknown) hides unreliable points. Confidence
// scales differ by method (e.g. VGGT's 1 + exp, Pi3's sigmoid), so the threshold is a percentile of the cloud's own
// confidences: "hide the lowest p %". Points are discarded in the vertex shader, so moving the slider is instant.

export function confidenceArray(raw) {
    // Unknown (NaN) sorts below every threshold above 0 %: such points show only when nothing is filtered.
    const values = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i += 1) values[i] = Number.isFinite(raw[i]) ? raw[i] : -3.0e38;
    return values;
}

// Below every confidence, the unknown sentinel included (a finite float: GLSL uniforms need one).
export const NO_CONFIDENCE_THRESHOLD = -3.4e38;

// The confidence value below which `percent` % of the cloud's points lie (nothing hidden at 0 %).
export function confidenceThreshold(values, percent) {
    if (!values?.length || percent <= 0) return NO_CONFIDENCE_THRESHOLD;
    if (!values.sorted) {
        const stride = Math.max(1, Math.floor(values.length / 200000));
        const sample = [];
        for (let i = 0; i < values.length; i += stride) sample.push(values[i]);
        values.sorted = Float32Array.from(sample).sort();
    }
    const sorted = values.sorted;
    return sorted[Math.min(sorted.length - 1, Math.floor((percent / 100) * sorted.length))];
}

// Make a PointsMaterial discard points whose 'confidence' attribute is below material.userData.confidenceThreshold.
export function enableConfidenceFilter(material) {
    const uniform = { value: NO_CONFIDENCE_THRESHOLD };
    material.userData.confidenceUniform = uniform;
    material.onBeforeCompile = shader => {
        shader.uniforms.confidenceThreshold = uniform;
        shader.vertexShader = shader.vertexShader
            .replace('#include <common>', '#include <common>\nattribute float confidence;\nuniform float confidenceThreshold;')
            .replace('#include <project_vertex>',
                '#include <project_vertex>\nif (confidence < confidenceThreshold) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);');
    };
    material.customProgramCacheKey = () => 'confidence-filter';
    material.needsUpdate = true;
    return uniform;
}

export function confidenceControlHtml() {
    return `<label class="compare-inline" data-role="confidence-label" title="Hide each cloud's least confident points (a percentile of its own predicted confidence)">
        Hide low confidence
        <input type="range" data-role="confidence" min="0" max="95" step="5" value="0">
        <span data-role="confidence-value" class="compare-muted">0%</span></label>`;
}

// Whether a #rrggbb color is light (relative luminance), so neutral overlays can switch to a dark ink.
export function isLightColor(hex) {
    const value = parseInt(String(hex).slice(1), 16);
    const channel = shift => {
        const c = ((value >> shift) & 255) / 255;
        return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
    };
    return 0.2126 * channel(16) + 0.7152 * channel(8) + 0.0722 * channel(0) > 0.4;
}

// ---- camera frustums ------------------------------------------------------------------------------------------

function rotateByQuaternion(quaternion, [vx, vy, vz]) {
    // v' = v + 2 w (q x v) + 2 q x (q x v), q = (x, y, z); renormalized (the payloads round quaternions).
    const norm = Math.hypot(...quaternion) || 1;
    const [w, x, y, z] = quaternion.map(value => value / norm);
    const cx = y * vz - z * vy;
    const cy = z * vx - x * vz;
    const cz = x * vy - y * vx;
    return [
        vx + 2 * (w * cx + y * cz - z * cy),
        vy + 2 * (w * cy + z * cx - x * cz),
        vz + 2 * (w * cz + x * cy - y * cx),
    ];
}

// Line-segment endpoints (pushed onto `out` as x, y, z triples) of one camera frustum: the pyramid from the camera
// center to an image rectangle `depth` in front of it, plus a small triangle on the image's top edge so its roll
// reads at a glance. `position` and the camera-to-world quaternion `quaternion` (w, x, y, z) use OpenCV axes
// (x right, y down, z forward); halfWidth and halfHeight are the image half-extents per unit depth (w / 2f).
export function pushFrustumSegments(out, position, quaternion, halfWidth, halfHeight, depth) {
    const world = ([x, y, z]) => {
        const [rx, ry, rz] = rotateByQuaternion(quaternion, [x * depth, y * depth, z * depth]);
        return [position[0] + rx, position[1] + ry, position[2] + rz];
    };
    const corners = [[-halfWidth, -halfHeight, 1], [halfWidth, -halfHeight, 1], [halfWidth, halfHeight, 1],
        [-halfWidth, halfHeight, 1]].map(world);
    const apex = [position[0], position[1], position[2]];
    const top = [[-0.35 * halfWidth, -halfHeight, 1], [0.35 * halfWidth, -halfHeight, 1], [0, -1.45 * halfHeight, 1]].map(world);
    const segment = (a, b) => out.push(a[0], a[1], a[2], b[0], b[1], b[2]);
    corners.forEach((corner, index) => {
        segment(apex, corner);
        segment(corner, corners[(index + 1) % 4]);
    });
    segment(top[0], top[2]);
    segment(top[1], top[2]);
    return out;
}

// ---- URL state ------------------------------------------------------------------------------------------------

export function readCompareState() {
    const [view, query = ''] = window.location.hash.replace(/^#/, '').split('?');
    if (view !== 'artifacts') return {};
    const params = new URLSearchParams(query);
    const state = Object.fromEntries(params.entries());
    if (state.methods) state.methods = state.methods.split(',').filter(Boolean);
    return state;
}

export function writeCompareState(state) {
    // Only while the artifact viewer is the open view: the compare modes initialize before the dashboard reads the
    // view to open from the URL, and rewriting another view's hash (e.g. #home) here would open the viewer instead.
    const view = window.location.hash.replace(/^#/, '').split('?')[0];
    if (view !== 'artifacts') return;
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(state)) {
        if (value === undefined || value === null || value === '') continue;
        params.set(key, Array.isArray(value) ? value.join(',') : String(value));
    }
    const hash = `#artifacts${params.toString() ? `?${params.toString()}` : ''}`;
    if (window.location.hash !== hash) window.history.replaceState(null, '', hash);
}

// ---- "Add methods" dialog -------------------------------------------------------------------------------------

// scenes: [{task, dataset_key, scene, methods: [{method, ...}]}]. Resolves to {scene, methods} or null.
// describeMethod(entry) renders the extra text next to a method (e.g. its ATE).
export function openAddDialog({ title, scenes, current, loadedMethods = [], describeMethod = () => '' }) {
    return new Promise(resolve => {
        const overlay = document.createElement('div');
        overlay.className = 'compare-dialog-overlay';
        overlay.innerHTML = `
            <div class="compare-dialog" role="dialog" aria-modal="true" aria-label="${escapeHtml(title)}">
                <div class="compare-dialog-header">
                    <h3>${escapeHtml(title)}</h3>
                    <button type="button" class="compare-icon-button" data-action="cancel" aria-label="Close">&times;</button>
                </div>
                <div class="compare-dialog-body">
                    <label>Task <select data-field="task"></select></label>
                    <label>Dataset <select data-field="dataset"></select></label>
                    <label>Scene <select data-field="scene"></select></label>
                    <div class="compare-dialog-methods-header">
                        <span>Methods</span>
                        <span>
                            <button type="button" class="compare-link-button" data-action="all">all</button>
                            <button type="button" class="compare-link-button" data-action="none">none</button>
                        </span>
                    </div>
                    <div class="compare-dialog-methods" data-field="methods"></div>
                    <p class="compare-dialog-note" data-field="note"></p>
                </div>
                <div class="compare-dialog-footer">
                    <button type="button" class="compare-button" data-action="cancel">Cancel</button>
                    <button type="button" class="compare-button compare-button-primary" data-action="add">Add</button>
                </div>
            </div>
        `;
        document.body.appendChild(overlay);
        const field = name => overlay.querySelector(`[data-field="${name}"]`);
        const tasks = [...new Set(scenes.map(scene => scene.task))];
        const fill = (select, values, label, preferred) => {
            select.innerHTML = values.map(value => (
                `<option value="${escapeHtml(value)}">${escapeHtml(label(value))}</option>`
            )).join('');
            if (values.includes(preferred)) select.value = preferred;
        };
        const selectedScene = () => scenes.find(scene => (
            scene.task === field('task').value
            && scene.dataset_key === field('dataset').value
            && scene.scene === field('scene').value
        ));
        const renderMethods = () => {
            const scene = selectedScene();
            const sameScene = current && scene && current.task === scene.task
                && current.dataset_key === scene.dataset_key && current.scene === scene.scene;
            field('methods').innerHTML = (scene?.methods || []).map(entry => {
                const loaded = sameScene && loadedMethods.includes(entry.method);
                return `
                    <label class="compare-dialog-method${loaded ? ' is-loaded' : ''}">
                        <input type="checkbox" value="${escapeHtml(entry.method)}"${loaded ? ' checked disabled' : ''}>
                        <span>${escapeHtml(methodLabel(entry.method))}</span>
                        <span class="compare-muted">${loaded ? 'loaded' : describeMethod(entry)}</span>
                    </label>
                `;
            }).join('') || '<p class="compare-muted">No methods for this scene.</p>';
            field('note').textContent = current && scene && !sameScene && loadedMethods.length
                ? 'A different scene replaces the methods currently in the viewer.'
                : '';
        };
        const onTask = () => {
            const datasets = [...new Set(scenes.filter(scene => scene.task === field('task').value)
                .map(scene => scene.dataset_key))].sort();
            fill(field('dataset'), datasets, datasetLabel, current?.dataset_key);
            onDataset();
        };
        const onDataset = () => {
            const sceneNames = scenes
                .filter(scene => scene.task === field('task').value && scene.dataset_key === field('dataset').value)
                .map(scene => scene.scene).sort();
            fill(field('scene'), sceneNames, value => value, current?.scene);
            renderMethods();
        };
        fill(field('task'), tasks, taskLabel, current?.task);
        field('task').addEventListener('change', onTask);
        field('dataset').addEventListener('change', onDataset);
        field('scene').addEventListener('change', renderMethods);
        onTask();
        const close = result => {
            overlay.remove();
            document.removeEventListener('keydown', onKey);
            resolve(result);
        };
        const onKey = event => { if (event.key === 'Escape') close(null); };
        document.addEventListener('keydown', onKey);
        overlay.addEventListener('click', event => {
            const action = event.target.closest('[data-action]')?.dataset.action;
            if (event.target === overlay || action === 'cancel') close(null);
            const boxes = [...field('methods').querySelectorAll('input:not(:disabled)')];
            if (action === 'all') boxes.forEach(box => { box.checked = true; });
            if (action === 'none') boxes.forEach(box => { box.checked = false; });
            if (action === 'add') {
                const methods = boxes.filter(box => box.checked).map(box => box.value);
                close(methods.length ? { scene: selectedScene(), methods } : null);
            }
        });
    });
}

// ---- Layer list -----------------------------------------------------------------------------------------------

// layers: [{id, label, color, detail, visible, removable}]. Calls onToggle(id) / onRemove(id) / onAdd().
// Move one item of a list to another position (in place); returns whether anything moved.
export function moveItem(items, from, to) {
    const target = Math.max(0, Math.min(items.length - 1, to));
    if (from < 0 || from >= items.length || target === from) return false;
    const [item] = items.splice(from, 1);
    items.splice(target, 0, item);
    return true;
}

// The window list's left / right arrow buttons, moving one viewer window earlier or later (data-move="-1" / "1").
// Only in the side list: the windows' own titles stay clean for figures.
function moveButtonsHtml(id, { first = false, last = false } = {}) {
    const button = (delta, icon, title, disabled) => `<button type="button" class="compare-icon-button compare-move-button"
        data-move="${delta}" data-move-id="${escapeHtml(id)}" title="${title}"${disabled ? ' disabled' : ''}>
        <i class="fas fa-chevron-${icon}"></i></button>`;
    return `<span class="compare-move-buttons">${button(-1, 'left', 'Move window left (earlier)', first)}${
        button(1, 'right', 'Move window right (later)', last)}</span>`;
}

// Ctrl (or Cmd) + drag a viewer window onto another one to move it there. pick(event) names the window under the
// pointer (null for none; ``movable`` false for one that stays put); onMove(fromId, toId) reorders. The press is taken
// before the view's own handlers (orbit, pan), so a Ctrl drag never also rotates or pans; showDrop(id | null) draws
// the target while dragging.
export function bindCtrlDragReorder(element, { pick, onMove, showDrop = () => {} }) {
    let drag = null;
    const finish = () => {
        if (!drag) return;
        element.classList.remove('is-reordering');
        showDrop(null);
        drag = null;
    };
    element.addEventListener('pointerdown', event => {
        if (event.button !== 0 || !(event.ctrlKey || event.metaKey)) return;
        const source = pick(event);
        if (!source || source.movable === false) return;
        event.preventDefault();
        event.stopPropagation();  // capture phase: the view's orbit / pan handlers never see this press
        drag = { id: source.id, pointerId: event.pointerId };
        element.setPointerCapture?.(event.pointerId);
        element.classList.add('is-reordering');
        showDrop(source.id);
    }, true);
    element.addEventListener('pointermove', event => {
        if (!drag) return;
        event.stopPropagation();
        const over = pick(event);
        showDrop(over && over.movable !== false ? over.id : drag.id);
    }, true);
    element.addEventListener('pointerup', event => {
        if (!drag) return;
        event.stopPropagation();
        const over = pick(event);
        const from = drag.id;
        finish();
        if (over && over.movable !== false && over.id !== from) onMove(from, over.id);
    }, true);
    element.addEventListener('pointercancel', finish, true);
    element.addEventListener('lostpointercapture', finish);
}

// The side list of a mode's windows / layers. onMove(id, delta), when given, adds left / right arrows to each movable
// row (layer.movable !== false) that move it one place earlier or later.
export function renderLayerList(container, { title, sceneLabel, layers, addLabel, onToggle, onRemove, onAdd, onMove }) {
    const movable = layers.filter(layer => layer.movable !== false).map(layer => layer.id);
    container.innerHTML = `
        <div class="compare-layers-header">
            <strong>${escapeHtml(title)}</strong>
            ${sceneLabel ? `<span class="compare-muted">${escapeHtml(sceneLabel)}</span>` : ''}
            ${onMove && movable.length > 1 ? '<span class="compare-muted compare-move-hint">Arrows or Ctrl + drag a window to reorder</span>' : ''}
        </div>
        <ul class="compare-layer-list">
            ${layers.map(layer => `
                <li class="compare-layer${layer.visible === false ? ' is-hidden' : ''}${onMove ? ' has-move' : ''}" data-layer="${escapeHtml(layer.id)}">
                    <button type="button" class="compare-layer-toggle" data-action="toggle" title="Show or hide">
                        <span class="compare-swatch" style="background:${escapeHtml(layer.color)}"></span>
                    </button>
                    <span class="compare-layer-name">${escapeHtml(layer.label)}</span>
                    <span class="compare-layer-detail">${layer.detail || ''}</span>
                    ${onMove ? (layer.movable === false || movable.length < 2 ? '<span></span>' : moveButtonsHtml(layer.id, {
                        first: movable.indexOf(layer.id) === 0, last: movable.indexOf(layer.id) === movable.length - 1,
                    })) : ''}
                    ${layer.removable === false ? '' : '<button type="button" class="compare-icon-button" data-action="remove" title="Remove">&times;</button>'}
                </li>
            `).join('')}
        </ul>
        <button type="button" class="compare-button compare-button-primary compare-add-button" data-action="add">
            <i class="fas fa-plus"></i> ${escapeHtml(addLabel)}
        </button>
    `;
    container.onclick = event => {
        const move = event.target.closest('[data-move]');
        if (move && onMove) { onMove(move.dataset.moveId, Number(move.dataset.move)); return; }
        const action = event.target.closest('[data-action]')?.dataset.action;
        const id = event.target.closest('[data-layer]')?.dataset.layer;
        if (action === 'toggle' && id) onToggle(id);
        if (action === 'remove' && id) onRemove(id);
        if (action === 'add') onAdd();
    };
}
