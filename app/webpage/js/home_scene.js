// Copyright (c) 2026 Yu Chen. All rights reserved.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

// The SpatialArena home tab's first slide background: one of our captures, its LiDAR map as a point cloud, streamed
// in chunks (assets/scene/*.sap2, packed from the scene's gt_points.ply). The points are shuffled, so the first chunk
// is already the whole scene at a lower density: it is assembled from the ground up while the camera flies in, and
// every later chunk fades in where it belongs as it arrives. Drag to orbit, right-drag to pan, Ctrl/Cmd + scroll,
// pinch or the +/- buttons to zoom; plain scrolling still moves to the next slide. The canvas is transparent, so the
// page's own (light or dark) background shows through.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const SCENE = {
    url: 'assets/scene/yorkminster_chapter_house.sap2',
    name: 'York Minster Chapter House',
};
const FIRST_CHUNK = 400000;
const LATER_CHUNK = 500000;
const ASSEMBLY_SECONDS = 3.4;
const FADE_SECONDS = 0.9;
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
// Small screens, data-saver connections and low-memory devices stop after two chunks (about 0.9 M points).
const lightBudget = window.matchMedia('(max-width: 760px)').matches || navigator.connection?.saveData
    || (navigator.deviceMemory && navigator.deviceMemory <= 4);
const MAX_CHUNKS = lightBudget ? 2 : Infinity;

const host = document.getElementById('home-hero-scene');
if (host) start(host);

function start(container) {
    const canvas = container.querySelector('canvas');
    const loader = document.getElementById('home-hero-loader');
    const loaderText = loader?.querySelector('.home-hero-loader-text');
    const loaderBar = loader?.querySelector('.home-hero-loader-bar i');
    const caption = document.getElementById('home-hero-caption');
    const setStatus = (text, fraction) => {
        if (loaderText) loaderText.textContent = text;
        if (loaderBar && fraction != null) loaderBar.style.width = `${Math.round(100 * fraction)}%`;
    };
    const fmt = n => n.toLocaleString('en-US');

    let renderer;
    try {
        renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: true, powerPreference: 'high-performance' });
    } catch (_error) {
        setStatus('3D preview unavailable (WebGL is off)', null);
        return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x000000, 0);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, 1, 0.05, 1000);
    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.55;
    controls.zoomSpeed = 0.8;
    controls.enabled = false;

    // Plain wheel scrolls the page (to the next slide); Ctrl/Cmd + wheel (and trackpad pinch, which arrives as a
    // Ctrl + wheel) zooms the scene. Registered in the capture phase so it runs before OrbitControls' own handler.
    canvas.addEventListener('wheel', event => {
        if (!event.ctrlKey && !event.metaKey) event.stopImmediatePropagation();
    }, { capture: true });

    // flyT0: when the current camera fly-in started (the first chunk's arrival, then every return to the home view).
    const state = { header: null, points: null, home: null, t0: null, flyT0: null, visible: true, idleSince: 0, loaded: 0 };
    const seconds = () => (performance.now() - state.t0) / 1000;
    let positions, colors, random, born, geometry;

    function resize() {
        const width = container.clientWidth, height = container.clientHeight;
        if (!width || !height) return;
        renderer.setSize(width, height, false);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        if (state.points) {
            state.points.material.uniforms.uScale.value = height / 2 / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
        }
    }
    new ResizeObserver(resize).observe(container);

    const homePanel = document.getElementById('benchmark-view-home');
    new IntersectionObserver(entries => { state.visible = entries.some(entry => entry.isIntersecting); }, { threshold: 0.02 })
        .observe(container);

    stream().catch(error => setStatus(`Could not load the scene (${error.message})`, null));

    // ---- streaming: the header, then each chunk as soon as its bytes have arrived ----
    async function stream() {
        setStatus(`Loading the ${SCENE.name} LiDAR map…`, 0);
        const response = await fetch(SCENE.url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const reader = response.body.getReader();
        const firstChunkBytes = 36 + 4 + FIRST_CHUNK * 9;
        let pending = new Uint8Array(0);
        let received = 0;
        let chunksRead = 0;
        for (;;) {
            const { done, value } = await reader.read();
            if (value) {
                const merged = new Uint8Array(pending.length + value.length);
                merged.set(pending);
                merged.set(value, pending.length);
                pending = merged;
                received += value.length;
            }
            if (!state.header && pending.length >= 36) {
                setup(readHeader(pending));
                pending = pending.slice(36);
            }
            while (state.header && pending.length >= 4 && chunksRead < Math.min(MAX_CHUNKS, state.header.chunks)) {
                const count = new DataView(pending.buffer, pending.byteOffset, 4).getUint32(0, true);
                const size = 4 + count * 9 + (-(count * 9) & 3);
                if (pending.length < size) break;
                addChunk(pending.slice(4, 4 + count * 9), count, chunksRead);
                pending = pending.slice(size);
                chunksRead += 1;
            }
            if (chunksRead === 0) {
                setStatus(`Loading the ${SCENE.name} LiDAR map · ${(received / 1e6).toFixed(1)} MB`,
                    Math.min(received / firstChunkBytes, 1));
            }
            if (state.header && chunksRead >= Math.min(MAX_CHUNKS, state.header.chunks)) {
                if (!done) reader.cancel().catch(() => {});
                break;
            }
            if (done) break;
        }
        const wait = state.t0 === null ? 0 : Math.max(0, ASSEMBLY_SECONDS * 1000 - (performance.now() - state.t0)) + 700;
        window.setTimeout(() => loader?.classList.add('is-done'), reducedMotion ? 0 : wait);
    }

    function readHeader(bytes) {
        if (String.fromCharCode(...bytes.subarray(0, 4)) !== 'SAP2') throw new Error('not a SAP2 file');
        const view = new DataView(bytes.buffer, bytes.byteOffset, 36);
        const header = {
            total: view.getUint32(4, true),
            chunks: view.getUint32(8, true),
            lo: [0, 1, 2].map(i => view.getFloat32(12 + 4 * i, true)),
            span: [0, 1, 2].map(i => view.getFloat32(24 + 4 * i, true)),
        };
        header.center = header.lo.map((value, i) => value + header.span[i] / 2);
        header.radius = Math.hypot(...header.span.map(value => value / 2));
        return header;
    }

    // ---- geometry: one buffer for every point to be loaded; each chunk fills its range and widens the draw range ----
    function setup(header) {
        state.header = header;
        const capacity = Number.isFinite(MAX_CHUNKS)
            ? Math.min(header.total, FIRST_CHUNK + LATER_CHUNK * (MAX_CHUNKS - 1)) : header.total;
        state.capacity = capacity;
        positions = new Float32Array(capacity * 3);
        colors = new Uint8Array(capacity * 3);
        random = new Float32Array(capacity);
        born = new Float32Array(capacity);
        geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('aColor', new THREE.BufferAttribute(colors, 3, true));
        geometry.setAttribute('aRandom', new THREE.BufferAttribute(random, 1));
        geometry.setAttribute('aBorn', new THREE.BufferAttribute(born, 1));
        geometry.setDrawRange(0, 0);
        const material = new THREE.ShaderMaterial({
            transparent: true,
            uniforms: {
                uTime: { value: 0 },
                uSize: { value: 0.05 },
                uScale: { value: 400 },
                uRadius: { value: header.radius },
                uMinY: { value: -header.span[2] / 2 },
                uRangeY: { value: header.span[2] },
                uAssembly: { value: reducedMotion ? 0 : 1 },
            },
            vertexShader: `
                attribute vec3 aColor;
                attribute float aRandom;
                attribute float aBorn;
                uniform float uTime, uSize, uScale, uRadius, uMinY, uRangeY, uAssembly;
                varying vec3 vColor;
                varying float vAlpha;
                void main() {
                    vec3 p = position;
                    float eased;
                    if (aBorn < 0.0) {
                        // first chunk: from the ground up, each point after a height delay with some jitter...
                        float height = (position.y - uMinY) / uRangeY;
                        float t = clamp((uTime - height * 1.7 - aRandom * 0.7) / 1.1, 0.0, 1.0);
                        eased = mix(1.0, 1.0 - pow(1.0 - t, 3.0), uAssembly);
                        // ...converging from a scattered shell around the scene
                        float theta = aRandom * 6.2831853 * 13.0;
                        float phi = acos(2.0 * fract(aRandom * 7.13) - 1.0);
                        vec3 shell = vec3(sin(phi) * cos(theta), cos(phi) * 0.6 + 0.4, sin(phi) * sin(theta)) * uRadius * 1.5;
                        p = mix(shell, position, eased);
                    } else {
                        // later chunks: fade in where they belong
                        eased = mix(1.0, clamp((uTime - aBorn) / ${FADE_SECONDS.toFixed(1)}, 0.0, 1.0), uAssembly);
                    }
                    vec4 mv = modelViewMatrix * vec4(p, 1.0);
                    gl_Position = projectionMatrix * mv;
                    gl_PointSize = clamp(uSize * uScale / -mv.z, 1.0, 3.5);
                    vColor = aColor;
                    vAlpha = eased;
                }`,
            fragmentShader: `
                varying vec3 vColor;
                varying float vAlpha;
                void main() {
                    vec2 d = gl_PointCoord - 0.5;
                    if (dot(d, d) > 0.25 || vAlpha < 0.01) discard;
                    gl_FragColor = vec4(vColor, vAlpha);
                }`,
        });
        state.points = new THREE.Points(geometry, material);
        state.points.frustumCulled = false;
        scene.add(state.points);
        // Inside the chapter house, below the vault: the orbit circles the room's centre, looking across it.
        const r = header.radius;
        state.home = { position: new THREE.Vector3(r * 0.34, -r * 0.16, r * 0.34), target: new THREE.Vector3(0, -r * 0.02, 0) };
        controls.target.copy(state.home.target);
        controls.minDistance = r * 0.05;
        controls.maxDistance = r * 2.4;
        state.flyStart = new THREE.Vector3(r * 0.4, r * 2.6, r * 0.9);  // the fly-in starts high above the roof
        camera.position.copy(reducedMotion ? state.home.position : state.flyStart);
        camera.lookAt(state.home.target);
        resize();
        renderer.setAnimationLoop(frame);
    }

    function addChunk(bytes, count, index) {
        const { lo, span, center, total } = state.header;
        count = Math.min(count, state.capacity - state.loaded);
        if (count <= 0) return;
        const quantized = new Uint16Array(bytes.buffer, bytes.byteOffset, count * 3);
        const rgb = bytes.subarray(bytes.length - (bytes.length / 9) * 3);
        const start = state.loaded;
        const t = index === 0 ? -1 : seconds();
        // The capture is z-up; three.js is y-up: (x, y, z) -> (x, z, -y), centred on the bounding box.
        for (let i = 0; i < count; i += 1) {
            const k = start + i;
            const x = lo[0] + quantized[3 * i] / 65535 * span[0] - center[0];
            const y = lo[1] + quantized[3 * i + 1] / 65535 * span[1] - center[1];
            const z = lo[2] + quantized[3 * i + 2] / 65535 * span[2] - center[2];
            positions[3 * k] = x; positions[3 * k + 1] = z; positions[3 * k + 2] = -y;
            colors[3 * k] = rgb[3 * i]; colors[3 * k + 1] = rgb[3 * i + 1]; colors[3 * k + 2] = rgb[3 * i + 2];
            random[k] = (Math.imul(k + 1, 2654435761) >>> 0) / 4294967296;
            born[k] = t;
        }
        for (const name of ['position', 'aColor', 'aRandom', 'aBorn']) {
            const attribute = geometry.getAttribute(name);
            attribute.addUpdateRange(start * attribute.itemSize, count * attribute.itemSize);
            attribute.needsUpdate = true;
        }
        state.loaded += count;
        geometry.setDrawRange(0, state.loaded);
        const goal = Math.min(total, state.capacity);
        if (index === 0) {
            state.t0 = state.flyT0 = performance.now();
            setStatus(`Assembling ${fmt(count)} LiDAR points`, state.loaded / goal);
            state.controlsTimer = window.setTimeout(enableControls, reducedMotion ? 0 : ASSEMBLY_SECONDS * 1000);
        } else {
            setStatus(`Refining · ${fmt(state.loaded)} of ${fmt(goal)} points`, state.loaded / goal);
        }
        if (caption) {
            caption.textContent = `${SCENE.name} · SpatialArena`;
        }
    }

    // Coming back to the home view replays the fly-in from above the roof (the points are already assembled), with
    // the controls off until it lands, as on the first visit.
    function replayFlyIn() {
        if (reducedMotion || !state.home || state.t0 === null) return;
        window.clearTimeout(state.controlsTimer);
        controls.enabled = false;
        controls.autoRotate = false;
        controls.target.copy(state.home.target);
        camera.position.copy(state.flyStart);
        camera.lookAt(state.home.target);
        flyFrom.set(0, 0, 0);
        state.flyT0 = performance.now();
        state.controlsTimer = window.setTimeout(enableControls, ASSEMBLY_SECONDS * 1000);
    }
    if (homePanel) {
        let wasHidden = homePanel.hidden;
        new MutationObserver(() => {
            if (wasHidden && !homePanel.hidden) replayFlyIn();
            wasHidden = homePanel.hidden;
        }).observe(homePanel, { attributes: true, attributeFilter: ['hidden'] });
    }

    function enableControls() {
        controls.enabled = true;
        controls.autoRotate = !reducedMotion;
        controls.autoRotateSpeed = 0.4;
        container.classList.add('is-ready');
    }

    const flyFrom = new THREE.Vector3();
    function frame() {
        if (!state.visible || !homePanel || homePanel.hidden || document.hidden || !state.points) return;
        if (state.t0 !== null && !reducedMotion) {
            state.points.material.uniforms.uTime.value = seconds();
        }
        if (state.flyT0 !== null && !reducedMotion) {
            // The last frame lands the camera exactly, even when the fly-in ran out while the hero was off-screen.
            const elapsed = (performance.now() - state.flyT0) / 1000;
            if (!flyFrom.lengthSq()) flyFrom.copy(camera.position);
            const k = 1 - Math.pow(1 - Math.min(elapsed / ASSEMBLY_SECONDS, 1), 3);
            camera.position.lerpVectors(flyFrom, state.home.position, k);
            camera.lookAt(state.home.target);
            if (elapsed >= ASSEMBLY_SECONDS) state.flyT0 = null;
        }
        if (controls.enabled && !controls.autoRotate && !reducedMotion && performance.now() - state.idleSince > 8000) {
            controls.autoRotate = true;
        }
        if (controls.enabled) controls.update();
        renderer.render(scene, camera);
    }

    // Interaction pauses the slow orbit; it resumes after 8 s without input.
    controls.addEventListener('start', () => { controls.autoRotate = false; state.idleSince = Infinity; });
    controls.addEventListener('end', () => { state.idleSince = performance.now(); });

    function dolly(factor) {
        if (!state.home) return;
        const offset = camera.position.clone().sub(controls.target).multiplyScalar(factor);
        const distance = THREE.MathUtils.clamp(offset.length(), controls.minDistance, controls.maxDistance);
        camera.position.copy(controls.target).add(offset.setLength(distance));
        controls.autoRotate = false;
        state.idleSince = performance.now();
    }
    document.getElementById('home-hero-zoom-in')?.addEventListener('click', () => dolly(0.8));
    document.getElementById('home-hero-zoom-out')?.addEventListener('click', () => dolly(1.25));
    document.getElementById('home-hero-reset')?.addEventListener('click', () => {
        if (!state.home) return;
        camera.position.copy(state.home.position);
        controls.target.copy(state.home.target);
        controls.autoRotate = !reducedMotion;
    });
}
