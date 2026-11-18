// Copyright (c) 2026 Yu Chen. All rights reserved.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

// SpatialArena home tab: the slide show, the scene carousel and the "evaluate your model" walkthrough. The alignment
// simulations are in js/home_align.js, the 3D background in js/home_scene.js, the statistics in js/home_atlas.js.

(function () {
    "use strict";

    const TEASER_DATASETS = [
        ["spatialarena", "SpatialArena (ours)"], ["eth3d", "ETH3D"], ["tanks_and_temples", "Tanks and Temples"],
        ["oxford_spires", "Oxford Spires"], ["dycheck", "DyCheck"], ["scannetpp", "ScanNet++"], ["dtu", "DTU"],
        ["kitti_odometry", "KITTI Odometry"], ["imc", "IMC Phototourism"], ["7scenes", "7-Scenes"],
        ["sintel", "Sintel"], ["vbr", "VBR"], ["hammer", "HAMMER"], ["tum_dynamics", "TUM Dynamics"],
        ["hiroom", "HiRoom"], ["spring", "Spring"], ["nrgbd", "NRGBD"], ["ibims1", "iBims-1"],
        ["bonn", "Bonn"], ["aerialmetric", "AerialMetric"], ["texturepoorsfm", "TexturePoorSfM"],
        ["synth4k", "Synth4K"], ["marigold", "Marigold suite"], ["kitti", "KITTI"],
    ];

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function homeVisible() {
        const panel = document.getElementById("benchmark-view-home");
        return panel && !panel.hidden && !document.hidden;
    }

    // ---------------------------------------------------------------- coding walkthrough

    const PYTHON_KEYWORDS = new Set(["from", "import", "as", "def", "return", "if", "not", "for", "in", "assert", "raise",
        "True", "False", "None"]);

    function highlightPython(text) {
        const segments = [];
        const pattern = /(#[^\n]*)|(f?"[^"\n]*")|(\b\d+(?:\.\d+)?\b)|([A-Za-z_][A-Za-z0-9_]*)(?=\()|([A-Za-z_][A-Za-z0-9_]*)|(\s+|.)/g;
        let match;
        while ((match = pattern.exec(text)) !== null) {
            if (match[1]) segments.push(["com", match[1]]);
            else if (match[2]) segments.push(["str", match[2]]);
            else if (match[3]) segments.push(["num", match[3]]);
            else if (match[4]) segments.push(["fn", match[4]]);
            else if (match[5]) segments.push([PYTHON_KEYWORDS.has(match[5]) ? "key" : "", match[5]]);
            else segments.push(["", match[6]]);
        }
        return segments;
    }

    // Each step: the window title and a list of blocks -- typed commands ("cmd"), typed code ("code") and printed
    // output ("out" / "ok").
    const CODE_STEPS = [
        {
            file: "terminal",
            blocks: [
                ["cmd", "git clone <this repository> SpatialArena && cd SpatialArena"],
                ["cmd", "pip install -e ."],
                ["out", "Successfully installed spatialarena-0.1.0"],
                ["com", "# The models themselves run in one shared environment (docs/benchmark/evaluator_quickstart.md)."],
            ],
        },
        {
            file: "evaluate.py",
            blocks: [
                ["code", `from spatialarena import BenchmarkEvaluator

bench = BenchmarkEvaluator(
    benchmark_root="/data/spatialarena",      # the downloaded benchmark data
    experiment_name="my_model",
    job_system="native",                       # or "slurm"
)

# SpatialArena's own captures, on three of the five tasks
bench.select_datasets([
    {"task": "MonoDepth", "dataset": "spatialarena"},
    {"task": "SfM", "dataset": "spatialarena"},
    {"task": "SLAM", "dataset": "spatialarena"},
])

# Your model is one command that writes depths/ and sparse/0 (COLMAP)
bench.select_models([{
    "name": "MyModel",
    "track": "rgb",
    "command": "python infer.py --data {data_path} --output {output_path}",
}])

report = bench.validate()
print(f"ok={report.ok}  errors={report.summary['error_count']}")
plan = bench.plan()
print(f"{len(plan.selected_runs)} runs planned")

result = bench.evaluate()          # inference, alignment and scoring
bench.export_report("./report")`],
            ],
        },
        {
            file: "terminal",
            blocks: [
                ["cmd", "python evaluate.py"],
                ["out", "ok=True  errors=0"],
                ["out", "3 runs planned"],
                ["out", "  MonoDepth  spatialarena  MyModel  rgb   inference ... scored"],
                ["out", "  SfM        spatialarena  MyModel  rgb   inference ... scored"],
                ["out", "  SLAM       spatialarena  MyModel  rgb   inference ... scored"],
                ["ok", "report written to ./report"],
                ["com", ""],
                ["com", "# The full benchmark, task by task, from the shipped configs:"],
                ["cmd", "python -m spatialarena.run_config configs/benchmark/full_slam.yaml plan"],
                ["out", "SLAM 7scenes DA3-L rgb short"],
                ["out", "SLAM 7scenes DA3-L rgb long"],
                ["out", "..."],
                ["out", "# 233 runs: SLAM/rgb/large=36, SLAM/rgb/long=63, SLAM/rgb/short=116, ..."],
            ],
        },
        {
            file: "terminal",
            blocks: [
                ["cmd", "app/webpage/run.sh 8020"],
                ["out", "collecting every report root ... results.json"],
                ["ok", "serving the dashboard on port 8020"],
                ["com", ""],
                ["com", "# Leaderboard: your model ranked per task and overall"],
                ["com", "# Artifact viewer: its depth maps, trajectories and point clouds next to the others"],
            ],
        },
    ];

    function createCodeDemo() {
        const body = document.getElementById("home-code-body");
        const file = document.getElementById("home-code-file");
        const stepList = document.getElementById("home-code-steps");
        if (!body || !file || !stepList) return;
        let run = 0;

        const sleep = ms => new Promise(resolve => window.setTimeout(resolve, ms));
        const waitVisible = async () => {
            while (!homeVisible()) await sleep(250);
        };
        const cursor = () => {
            const element = document.createElement("span");
            element.className = "home-code-cursor";
            return element;
        };

        async function typeSegments(segments, perChar, token) {
            const caret = cursor();
            body.appendChild(caret);
            for (const [cls, text] of segments) {
                const span = document.createElement("span");
                if (cls) span.className = `tok-${cls}`;
                body.insertBefore(span, caret);
                if (reducedMotion || perChar === 0) {
                    span.textContent = text;
                    continue;
                }
                for (const char of text) {
                    if (token !== run) return false;
                    await waitVisible();
                    span.textContent += char;
                    body.scrollTop = body.scrollHeight;
                    await sleep(char === "\n" ? perChar * 6 : perChar);
                }
            }
            caret.remove();
            return token === run;
        }

        async function playStep(index, token, instant) {
            const step = CODE_STEPS[index];
            body.textContent = "";
            file.textContent = step.file;
            stepList.querySelectorAll("li").forEach(item => {
                item.classList.toggle("is-active", Number(item.dataset.codeStep) === index);
            });
            for (const [kind, text] of step.blocks) {
                if (token !== run) return false;
                if (kind === "cmd") {
                    const ok = await typeSegments([["prompt", "$ "], ["", text], ["", "\n"]], instant ? 0 : 26, token);
                    if (!ok) return false;
                    if (!instant) await sleep(350);
                } else if (kind === "code") {
                    const ok = await typeSegments(highlightPython(text), instant ? 0 : 9, token);
                    if (!ok) return false;
                } else {
                    const span = document.createElement("span");
                    span.className = kind === "ok" ? "tok-ok" : (kind === "com" ? "tok-com" : "tok-out");
                    span.textContent = `${kind === "ok" ? "✓ " : ""}${text}\n`;
                    body.appendChild(span);
                    body.scrollTop = body.scrollHeight;
                    if (!instant) {
                        await waitVisible();
                        await sleep(kind === "com" ? 180 : 260);
                    }
                }
            }
            return true;
        }

        async function playFrom(start) {
            run += 1;
            const token = run;
            if (reducedMotion) {
                await playStep(start, token, true);
                return;
            }
            for (let index = start; ; index = (index + 1) % CODE_STEPS.length) {
                const finished = await playStep(index, token, false);
                if (!finished || token !== run) return;
                await sleep(index === CODE_STEPS.length - 1 ? 5000 : 2200);
                await waitVisible();
                if (token !== run) return;
            }
        }

        stepList.querySelectorAll("li").forEach(item => {
            item.tabIndex = 0;
            const go = () => playFrom(Number(item.dataset.codeStep));
            item.addEventListener("click", go);
            item.addEventListener("keydown", event => {
                if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    go();
                }
            });
        });
        document.getElementById("home-code-replay")?.addEventListener("click", () => playFrom(0));
        playFrom(0);
    }

    // ---------------------------------------------------------------- teaser and wiring

    function buildTeaser() {
        const track = document.getElementById("home-teaser-track");
        if (!track || track.childElementCount) return;
        const items = TEASER_DATASETS.map(([key, label]) => {
            const item = document.createElement("div");
            item.className = `home-teaser-item${key === "spatialarena" ? " is-ours" : ""}`;
            const image = document.createElement("img");
            image.src = `assets/datasets/${key}.jpg`;
            image.alt = `${label}: sample frame`;
            image.loading = "lazy";
            image.addEventListener("error", () => item.remove());
            const caption = document.createElement("span");
            caption.textContent = label;
            item.append(image, caption);
            return item;
        });
        // Two copies make the scroll seamless; the second is hidden from assistive technology.
        items.forEach(item => track.appendChild(item));
        items.forEach(item => {
            const copy = item.cloneNode(true);
            copy.setAttribute("aria-hidden", "true");
            copy.querySelector("img").alt = "";
            track.appendChild(copy);
        });
    }

    // ---------------------------------------------------------------- SpatialArena scene carousel

    const GALLERY_TASK_COLORS = { MonoDepth: "--primary", VideoDepth: "--success", MVRecon: "--secondary", SfM: "--warning",
        SLAM: "--accent" };
    const GALLERY_LAYERS = [["rgb", "RGB"], ["lidar", "LiDAR depth"], ["dense", "Dense depth"]];
    const CAROUSEL_INTERVAL = 6000;
    const carousel = { scenes: [], task: "all", index: 0, timer: null, hovered: false };

    const sceneTitle = name => name.replace(/([a-z])([A-Z0-9])/g, "$1 $2").replace(/([A-Z])([A-Z][a-z])/g, "$1 $2");
    const carouselScenes = () => carousel.scenes.filter(scene => carousel.task === "all" || scene.tasks.includes(carousel.task));

    function renderCarouselView() {
        const view = document.getElementById("home-carousel-view");
        const scenes = carouselScenes();
        if (!view || !scenes.length) return;
        carousel.index = (carousel.index + scenes.length) % scenes.length;
        const scene = scenes[carousel.index];
        const media = document.createElement("div");
        media.className = "home-scene-media";
        GALLERY_LAYERS.forEach(([key, label]) => {
            const layer = document.createElement("div");
            layer.className = "home-scene-layer";
            if (scene[key]) {
                const image = document.createElement("img");
                image.src = scene[key];
                image.alt = `${sceneTitle(scene.scene)}: ${label}`;
                layer.appendChild(image);
            } else {
                const missing = document.createElement("div");
                missing.className = "home-scene-missing";
                missing.textContent = key === "dense" && scene.lidar ? "no dense depth (LiDAR only)" : "no depth (pose-only sequence)";
                layer.appendChild(missing);
            }
            const tag = document.createElement("span");
            tag.textContent = label;
            layer.appendChild(tag);
            media.appendChild(layer);
        });
        const body = document.createElement("div");
        body.className = "home-scene-body";
        const title = document.createElement("div");
        title.innerHTML = `<b></b><small>${carousel.index + 1} / ${scenes.length}</small>`;
        title.querySelector("b").textContent = sceneTitle(scene.scene);
        const tags = document.createElement("div");
        tags.className = "home-scene-tags";
        scene.tasks.forEach(task => {
            const chip = document.createElement("span");
            chip.textContent = task;
            chip.style.background = `var(${GALLERY_TASK_COLORS[task] || "--gray"})`;
            tags.appendChild(chip);
        });
        body.append(title, tags);
        view.replaceChildren(media, body);
        const strip = document.getElementById("home-carousel-strip");
        strip?.querySelectorAll("button").forEach((button, i) => {
            const active = i === carousel.index;
            button.setAttribute("aria-selected", String(active));
            button.tabIndex = active ? 0 : -1;
            if (active && strip.scrollWidth > strip.clientWidth) {
                strip.scrollTo({ left: button.offsetLeft - strip.clientWidth / 2 + button.offsetWidth / 2, behavior: "smooth" });
            }
        });
    }

    function renderCarousel() {
        const scenes = carouselScenes();
        const strip = document.getElementById("home-carousel-strip");
        if (strip) {
            strip.replaceChildren(...scenes.map((scene, i) => {
                const button = document.createElement("button");
                button.type = "button";
                button.setAttribute("role", "tab");
                button.title = sceneTitle(scene.scene);
                button.setAttribute("aria-label", sceneTitle(scene.scene));
                button.innerHTML = `<img alt="" loading="lazy" src="${scene.rgb}">`;
                button.addEventListener("click", () => goToScene(i));
                return button;
            }));
        }
        const count = document.getElementById("home-gallery-count");
        if (count) {
            const dense = scenes.filter(scene => scene.dense).length;
            const lidar = scenes.filter(scene => scene.lidar).length;
            count.textContent = `${scenes.length} scenes · ${lidar} with LiDAR depth · ${dense} with dense depth`;
        }
        renderCarouselView();
    }

    function goToScene(index) {
        carousel.index = index;
        renderCarouselView();
        restartCarouselTimer();
    }

    function carouselInView() {
        const host = document.getElementById("home-carousel");
        if (!host || !homeVisible()) return false;
        const box = host.getBoundingClientRect();
        return box.bottom > 0 && box.top < window.innerHeight;
    }

    function restartCarouselTimer() {
        window.clearInterval(carousel.timer);
        if (reducedMotion) return;
        carousel.timer = window.setInterval(() => {
            if (!carousel.hovered && carouselInView()) {
                carousel.index += 1;
                renderCarouselView();
            }
        }, CAROUSEL_INTERVAL);
    }

    async function buildCarousel() {
        const host = document.getElementById("home-carousel");
        if (!host) return;
        try {
            const response = await fetch("assets/spatialarena/manifest.json", { cache: "no-cache" });
            // scenes with all three layers first, then LiDAR-only, then pose-only; alphabetical within each
            const rank = scene => (scene.dense ? 0 : scene.lidar ? 1 : 2);
            carousel.scenes = (await response.json()).sort((a, b) => rank(a) - rank(b) || a.scene.localeCompare(b.scene));
        } catch (_error) {
            document.getElementById("home-carousel-view").textContent = "The scene gallery could not be loaded.";
            return;
        }
        const taskGroup = document.getElementById("home-gallery-tasks");
        if (taskGroup) {
            const tasks = Object.keys(GALLERY_TASK_COLORS).filter(task => carousel.scenes.some(scene => scene.tasks.includes(task)));
            taskGroup.innerHTML = ["all", ...tasks].map(task => `<button type="button" role="radio" aria-checked="${task === "all"}"`
                + ` data-task="${task}">${task === "all" ? "All tasks" : task}</button>`).join("");
            taskGroup.addEventListener("click", event => {
                const button = event.target.closest("button");
                if (!button) return;
                taskGroup.querySelectorAll("button").forEach(item => item.setAttribute("aria-checked", String(item === button)));
                carousel.task = button.dataset.task;
                carousel.index = 0;
                renderCarousel();
                restartCarouselTimer();
            });
        }
        document.getElementById("home-carousel-prev")?.addEventListener("click", () => goToScene(carousel.index - 1));
        document.getElementById("home-carousel-next")?.addEventListener("click", () => goToScene(carousel.index + 1));
        host.addEventListener("mouseenter", () => { carousel.hovered = true; });
        host.addEventListener("mouseleave", () => { carousel.hovered = false; });
        host.addEventListener("focusin", () => { carousel.hovered = true; });
        host.addEventListener("focusout", () => { carousel.hovered = false; });
        host.addEventListener("keydown", event => {
            if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
                event.preventDefault();
                event.stopPropagation();
                goToScene(carousel.index + (event.key === "ArrowRight" ? 1 : -1));
            }
        });
        renderCarousel();
        restartCarouselTimer();
    }

    // ---------------------------------------------------------------- slide show

    function slides() {
        return [...document.querySelectorAll("#benchmark-view-home .home-slide")];
    }

    function currentSlideIndex() {
        const offset = (document.querySelector(".navbar")?.offsetHeight || 0) + 4;
        const list = slides();
        let index = 0;
        list.forEach((slide, i) => {
            if (slide.getBoundingClientRect().top - offset <= window.innerHeight * 0.4) index = i;
        });
        return index;
    }

    function goToSlide(index) {
        const list = slides();
        const target = list[Math.max(0, Math.min(index, list.length - 1))];
        target?.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "start" });
    }

    function updateDots() {
        const index = currentSlideIndex();
        document.querySelectorAll("#home-dots button").forEach((dot, i) => {
            dot.classList.toggle("is-active", i === index);
            dot.setAttribute("aria-current", i === index ? "step" : "false");
        });
    }

    function syncSlideMode() {
        const navbar = document.querySelector(".navbar");
        if (navbar) document.documentElement.style.setProperty("--home-nav-h", `${navbar.offsetHeight}px`);
        document.documentElement.classList.toggle("home-slides", Boolean(homeVisible()));
        updateDots();
    }

    // Slide transitions: a slide's content animates in (staggered, see css/home.css) each time the slide comes into
    // view, and resets once it has fully left, so it plays again on the way back.
    function animateSlides(list) {
        if (reducedMotion || !("IntersectionObserver" in window)) return;
        list.forEach(slide => {
            const parts = slide.classList.contains("home-slide-hero")
                ? slide.querySelectorAll(".home-hero > *:not(.home-title)")
                : slide.querySelectorAll(":scope > .home-slide-inner > *");
            parts.forEach((part, i) => part.style.setProperty("--home-stagger", String(i)));
        });
        const observer = new IntersectionObserver(entries => entries.forEach(entry => {
            if (entry.intersectionRatio >= 0.25) entry.target.classList.add("is-in-view");
            else if (!entry.isIntersecting) entry.target.classList.remove("is-in-view");
        }), { threshold: [0, 0.25] });
        list.forEach(slide => observer.observe(slide));
        document.documentElement.classList.add("home-anim");
    }

    function buildSlideShow() {
        const dots = document.getElementById("home-dots");
        const list = slides();
        if (!dots || !list.length) return;
        dots.replaceChildren(...list.map((slide, i) => {
            const dot = document.createElement("button");
            dot.type = "button";
            dot.setAttribute("aria-label", `Slide ${i + 1}: ${slide.dataset.slideTitle}`);
            dot.innerHTML = "<span></span>";
            dot.querySelector("span").textContent = slide.dataset.slideTitle;
            dot.addEventListener("click", () => goToSlide(i));
            return dot;
        }));
        const hero = list[0];
        if (hero && !hero.querySelector(".home-slide-next")) {
            const next = document.createElement("button");
            next.type = "button";
            next.className = "home-slide-next";
            next.setAttribute("aria-label", "Next slide");
            next.innerHTML = '<i class="fas fa-chevron-down" aria-hidden="true"></i>';
            next.addEventListener("click", () => goToSlide(1));
            hero.appendChild(next);
        }
        document.addEventListener("keydown", event => {
            if (!homeVisible() || event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) return;
            const tag = event.target.tagName;
            if (["INPUT", "TEXTAREA", "SELECT"].includes(tag) || event.target.isContentEditable) return;
            const step = { ArrowDown: 1, PageDown: 1, ArrowUp: -1, PageUp: -1 }[event.key];
            if (event.key === " " && tag !== "BUTTON" && tag !== "A") {
                event.preventDefault();
                goToSlide(currentSlideIndex() + (event.shiftKey ? -1 : 1));
            } else if (step) {
                event.preventDefault();
                goToSlide(currentSlideIndex() + step);
            }
        });
        animateSlides(list);
        window.addEventListener("scroll", updateDots, { passive: true });
        window.addEventListener("resize", syncSlideMode);
        const panel = document.getElementById("benchmark-view-home");
        if (panel) new MutationObserver(syncSlideMode).observe(panel, { attributes: true, attributeFilter: ["hidden"] });
        syncSlideMode();
        // Deep link to a slide: #home?slide=N (1-based), e.g. to share the scene gallery.
        const [view, query = ""] = window.location.hash.replace(/^#/, "").split("?");
        const wanted = Number(new URLSearchParams(query).get("slide"));
        if (view === "home" && wanted > 1) {
            window.setTimeout(() => list[Math.min(wanted, list.length) - 1]?.scrollIntoView({ block: "start" }), 150);
        }
    }

    window.renderSpatialArenaHome = function () {
        if (typeof window.renderSpatialArenaAtlas === "function") window.renderSpatialArenaAtlas();
    };

    document.addEventListener("DOMContentLoaded", () => {
        buildTeaser();
        buildCarousel();
        buildSlideShow();
        document.querySelectorAll("[data-home-scroll]").forEach(button => {
            button.addEventListener("click", () => {
                const target = document.getElementById(button.dataset.homeScroll);
                const index = slides().indexOf(target);
                if (index >= 0) goToSlide(index);
            });
        });
        createCodeDemo();
    });
})();
