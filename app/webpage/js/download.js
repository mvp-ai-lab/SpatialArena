// Copyright (c) 2026 Yu Chen. All rights reserved.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

// Download view (#download): how to fetch the benchmark data from its Hugging Face release. The task picker builds
// the `hf download` command for the chosen tasks, the release-contents table lists every dataset with its size, and
// each code block has a copy button.

(function () {
    const REPO_ID = "mvp-lab/SpatialArena-Dataset";
    const ROOT = "/data/spatialarena";
    const TASKS = ["MonoDepth", "VideoDepth", "MVRecon", "SfM", "SLAM"];

    // Packed size of each <Task>/<dataset>/ folder in the release, in GB. Datasets two tasks share are packed once:
    // `pointer` names the folder holding the data (the other one only has SEE_INSTEAD.md and LICENSE_NOTICE.md).
    const RELEASE = {
        MonoDepth: [
            ["aerialmetric", "AerialMetric", 13.6], ["hammer", "HAMMER", 5.0], ["ibims1", "iBims-1", 0.1],
            ["marigold", "Marigold", 56.8], ["spatialarena", "SpatialArena", 18.4], ["spring", "Spring", 8.9],
            ["synth4k", "Synth4K", 89.8],
        ],
        VideoDepth: [
            ["bonn", "Bonn", 18.5], ["dycheck", "DyCheck", 8.1], ["kitti", "KITTI", 32.0], ["sintel", "Sintel", 2.4],
            ["spatialarena", "SpatialArena", 49.7],
        ],
        MVRecon: [
            ["7scenes", "7-Scenes", 47.3], ["dtu", "DTU", 24.4], ["eth3d", "ETH3D", 0, "SfM/eth3d"],
            ["hiroom", "HiRoom", 0.8], ["nrgbd", "Neural RGB-D", 5.9], ["scannetpp", "ScanNet++", 40.6],
            ["spatialarena", "SpatialArena", 56.9], ["tanks_and_temples", "Tanks and Temples", 0, "SfM/tanks_and_temples"],
        ],
        SfM: [
            ["dtu", "DTU", 0, "MVRecon/dtu"], ["eth3d", "ETH3D", 100.0], ["hiroom", "HiRoom", 0, "MVRecon/hiroom"],
            ["imc", "IMC", 19.1], ["scannetpp", "ScanNet++", 0, "MVRecon/scannetpp"],
            ["spatialarena", "SpatialArena", 18.6], ["tanks_and_temples", "Tanks and Temples", 16.8],
            ["texturepoorsfm", "Weak-Texture SfM", 0.2],
        ],
        SLAM: [
            ["7scenes", "7-Scenes", 6.1], ["kitti_odometry", "KITTI Odometry", 3.4], ["oxford_spires", "Oxford Spires", 10.9],
            ["spatialarena", "SpatialArena", 2.9], ["tum_dynamics", "TUM Dynamics", 4.4], ["vbr", "VBR", 18.2],
        ],
    };
    // Folders a task needs from another task: the data its pointers name, and MVRecon/7scenes, which SLAM/7scenes
    // links into.
    const COMPANIONS = {
        MVRecon: ["SfM/eth3d", "SfM/tanks_and_temples"],
        SfM: ["MVRecon/dtu", "MVRecon/hiroom", "MVRecon/scannetpp"],
        SLAM: ["MVRecon/7scenes"],
    };

    const selected = new Set(TASKS);
    let spatialArenaOnly = false;

    const sizeOf = path => {
        const [task, dataset] = path.split("/");
        return (RELEASE[task] || []).find(row => row[0] === dataset)?.[2] || 0;
    };
    const formatSize = gb => (gb >= 1000 ? `${(gb / 1000).toFixed(2)} TB` : gb >= 10 ? `${Math.round(gb)} GB` : `${gb.toFixed(1)} GB`);
    const escapeHtml = value => String(value).replace(/[&<>"']/g, char => (
        { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));

    // The include patterns and total size of the current choice.
    function plan() {
        const tasks = TASKS.filter(task => selected.has(task));
        if (spatialArenaOnly) {
            const folders = tasks.map(task => `${task}/spatialarena`);
            return { tasks, folders, size: folders.reduce((sum, path) => sum + sizeOf(path), 0), all: false };
        }
        const folders = new Set();
        for (const task of tasks) {
            for (const [dataset] of RELEASE[task]) folders.add(`${task}/${dataset}`);
            for (const path of COMPANIONS[task] || []) folders.add(path);
        }
        const size = [...folders].reduce((sum, path) => sum + sizeOf(path), 0);
        return { tasks, folders: [...folders], size, all: tasks.length === TASKS.length };
    }

    function downloadCommand({ tasks, folders, all }) {
        const lines = [`hf download ${REPO_ID} --repo-type dataset --local-dir ${ROOT}`];
        if (!all) {
            const patterns = spatialArenaOnly || tasks.length === 0
                ? folders.map(path => `${path}/*`)
                : [...tasks.map(task => `${task}/*`),
                    ...folders.filter(path => !tasks.includes(path.split("/")[0])).map(path => `${path}/*`)];
            patterns.push("extract_benchmark_archives.py", "README.md");
            for (const pattern of patterns) lines.push(`--include "${pattern}"`);
        }
        return lines.join(" \\\n    ");
    }

    function renderPicker() {
        const picker = document.getElementById("download-task-picker");
        if (!picker) return;
        picker.innerHTML = TASKS.map(task => {
            const on = selected.has(task);
            return `<button type="button" class="download-chip${on ? " active" : ""}" data-download-task="${task}"
                aria-pressed="${on}">${on ? '<i class="fas fa-check" aria-hidden="true"></i>' : ""}${task}</button>`;
        }).join("");
        const scope = document.getElementById("download-scope");
        if (scope) {
            scope.querySelectorAll("[data-download-scope]").forEach(button => {
                const on = (button.dataset.downloadScope === "spatialarena") === spatialArenaOnly;
                button.classList.toggle("active", on);
                button.setAttribute("aria-pressed", String(on));
            });
        }
    }

    function renderCommand() {
        const current = plan();
        const code = document.getElementById("download-command");
        const size = document.getElementById("download-size");
        if (code) {
            code.textContent = current.tasks.length ? downloadCommand(current) : "# Pick at least one task above.";
        }
        if (size) {
            const what = spatialArenaOnly ? "SpatialArena tracks" : current.all ? "the full release" : "selected tasks";
            const borrowed = current.folders.some(path => !current.tasks.includes(path.split("/")[0]));
            const extra = !spatialArenaOnly && borrowed ? ", including the shared datasets they need" : "";
            size.textContent = current.tasks.length ? `About ${formatSize(current.size)} for ${what}${extra}.` : "";
        }
    }

    function renderContents() {
        const body = document.getElementById("download-contents-body");
        if (!body) return;
        let total = 0;
        body.innerHTML = TASKS.map(task => {
            const rows = RELEASE[task];
            const taskSize = rows.reduce((sum, row) => sum + row[2], 0);
            total += taskSize;
            const chips = rows.map(([dataset, label, gb, pointer]) => {
                const ours = dataset === "spatialarena";
                const note = pointer ? `stored in ${pointer}` : formatSize(gb);
                return `<span class="download-dataset${ours ? " is-ours" : ""}" title="${escapeHtml(`${task}/${dataset}`)}">`
                    + `${escapeHtml(label)}<small>${escapeHtml(note)}</small></span>`;
            }).join("");
            return `<tr><th scope="row">${task}</th><td><div class="download-datasets">${chips}</div></td>`
                + `<td class="numeric">${formatSize(taskSize)}</td></tr>`;
        }).join("");
        const totalCell = document.getElementById("download-contents-total");
        if (totalCell) totalCell.textContent = formatSize(total);
    }

    function copyText(text, button) {
        const done = () => {
            const label = button.querySelector("span");
            const before = label?.textContent;
            button.classList.add("copied");
            if (label) label.textContent = "Copied";
            window.setTimeout(() => {
                button.classList.remove("copied");
                if (label) label.textContent = before;
            }, 1400);
        };
        if (navigator.clipboard?.writeText) {
            navigator.clipboard.writeText(text).then(done).catch(() => fallbackCopy(text) && done());
        } else if (fallbackCopy(text)) {
            done();
        }
    }

    function fallbackCopy(text) {
        const area = document.createElement("textarea");
        area.value = text;
        area.setAttribute("readonly", "");
        area.style.position = "fixed";
        area.style.opacity = "0";
        document.body.appendChild(area);
        area.select();
        let ok = false;
        try {
            ok = document.execCommand("copy");
        } catch (_error) {
            ok = false;
        }
        area.remove();
        return ok;
    }

    function init() {
        const view = document.getElementById("benchmark-view-download");
        if (!view) return;
        view.addEventListener("click", event => {
            const task = event.target.closest("[data-download-task]")?.dataset.downloadTask;
            if (task) {
                if (selected.has(task)) selected.delete(task); else selected.add(task);
                renderPicker();
                renderCommand();
                return;
            }
            const scope = event.target.closest("[data-download-scope]")?.dataset.downloadScope;
            if (scope) {
                spatialArenaOnly = scope === "spatialarena";
                renderPicker();
                renderCommand();
                return;
            }
            const copy = event.target.closest("[data-copy]");
            if (copy) {
                const source = document.getElementById(copy.dataset.copy);
                if (source) copyText(source.textContent, copy);
            }
        });
        renderPicker();
        renderCommand();
        renderContents();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
