// Copyright (c) 2026 Yu Chen. All rights reserved.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

// Mode switch of the artifact viewer: one comparison mode per kind of artifact. A mode is created the first time it is
// opened and kept, so switching back keeps its layers.

import { INDEX_ROOT, fetchAssetJson, readCompareState } from './compare_common.js';

const MODES = {
    trajectory: {
        index: 'trajectories.json',
        label: 'Trajectory comparison', hint: 'SfM and SLAM: aligned trajectories of several methods in one view',
        load: () => import('./trajectory_compare.js').then(module => module.TrajectoryCompare),
    },
    depth: {
        index: 'depth.json',
        label: 'Depth comparison', hint: 'MonoDepth and VideoDepth: swipe between methods, play videos',
        load: () => import('./depth_compare.js').then(module => module.DepthCompare),
    },
    reconstruction: {
        index: 'recon.json',
        label: 'Reconstruction comparison', hint: 'MVRecon: point clouds in linked views',
        load: () => import('./reconstruction_compare.js').then(module => module.ReconstructionCompare),
    },
};

const DEFAULT_MODE = Object.keys(MODES)[0];
// A mode opened without a shared link shows the first released scene with its best-ranked methods, so the viewer is
// never empty; the visitor adds or removes methods from there.
const DEFAULT_SCENE_METHODS = 4;

async function defaultState(mode) {
    const index = await fetchAssetJson(`${INDEX_ROOT}/${MODES[mode].index}`);
    const scene = index.scenes?.[0];
    if (!scene) return null;
    return {
        compare: mode, task: scene.task, dataset: scene.dataset_key, scene: scene.scene,
        methods: scene.methods.slice(0, DEFAULT_SCENE_METHODS).map(item => item.method),
    };
}
const instances = {};

function setupModeSwitch() {
    const card = document.querySelector('#benchmark-view-artifacts .card');
    if (!card || card.querySelector('.compare-mode-switch')) return;
    const modeSwitch = document.createElement('div');
    modeSwitch.className = 'compare-mode-switch compare-segmented';
    modeSwitch.setAttribute('role', 'tablist');
    modeSwitch.innerHTML = Object.entries(MODES).map(([mode, config]) => (
        `<button type="button" role="tab" data-compare-mode="${mode}" title="${config.hint}">${config.label}</button>`
    )).join('');
    const compareRoot = document.createElement('div');
    compareRoot.className = 'compare-root';
    card.append(modeSwitch, compareRoot);
    modeSwitch.addEventListener('click', event => {
        const mode = event.target.closest('[data-compare-mode]')?.dataset.compareMode;
        if (mode) activateMode(mode);
    });
}

async function activateMode(mode, state = null) {
    if (!MODES[mode]) mode = DEFAULT_MODE;
    document.querySelectorAll('[data-compare-mode]').forEach(button => {
        const active = button.dataset.compareMode === mode;
        button.classList.toggle('active', active);
        button.setAttribute('aria-selected', active ? 'true' : 'false');
    });
    const compareRoot = document.querySelector('.compare-root');
    for (const [name, instance] of Object.entries(instances)) instance.container.hidden = name !== mode;
    if (!instances[mode]) {
        const container = document.createElement('div');
        container.className = `compare-mode compare-mode-${mode}`;
        compareRoot.appendChild(container);
        try {
            const ModeClass = await MODES[mode].load();
            instances[mode] = { container, viewer: new ModeClass(container) };
        } catch (error) {
            console.error(error);
            container.innerHTML = `<div class="benchmark-empty">This comparison mode failed to load: ${error.message}</div>`;
            instances[mode] = { container, viewer: null };
            return;
        }
        const restore = state?.compare === mode ? state : await defaultState(mode).catch(() => null);
        if (restore) await instances[mode].viewer.restore?.(restore);
    }
    instances[mode].container.hidden = false;
    instances[mode].viewer?.resize?.();
    if (!state) instances[mode].viewer?.afterLayersChanged?.(false);
}

function init() {
    setupModeSwitch();
    const state = readCompareState();
    activateMode(state.compare || DEFAULT_MODE, state);
}

// Set up the first time the Artifact Viewer is shown (the dashboard unhides its panel), so other views never fetch
// the viewer's assets.
function initWhenShown() {
    const panel = document.getElementById('benchmark-view-artifacts');
    if (!panel || !panel.hidden) {
        init();
        return;
    }
    const observer = new MutationObserver(() => {
        if (panel.hidden) return;
        observer.disconnect();
        init();
    });
    observer.observe(panel, { attributes: true, attributeFilter: ['hidden'] });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initWhenShown);
} else {
    initWhenShown();
}
