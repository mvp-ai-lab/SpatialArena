// Copyright (c) 2026 Yu Chen. All rights reserved.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

let benchmarkData = null;
let benchmarkRadarChart = null;
const benchmarkRadarCharts = {};
const benchmarkBarCharts = {};
const benchmarkScatterCharts = {};
// Zoomed region of each score-vs-date chart, by task ({ xMin, xMax, yMin, yMax }); kept across re-renders.
const benchmarkReleaseZoom = {};
let benchmarkVisibleRankingRows = [];
let benchmarkVisibleTaskSections = [];
let benchmarkRankingSortState = { key: "score", direction: "desc", type: "base" };
let benchmarkActiveTaskMetricTab = "";
// The Leaderboard view shows the overall ranking (true) or the metric tables of benchmarkActiveTaskMetricTab (false).
let benchmarkLeaderboardOverall = true;
// The dataset tab chosen in each task ("all" = the Overall tab), so switching tasks keeps each one's choice.
let benchmarkActiveDatasetTabs = {};
const benchmarkTaskSortState = {};
const benchmarkColors = [
    "rgba(67, 97, 238, 0.28)",
    "rgba(247, 37, 133, 0.24)",
    "rgba(16, 185, 129, 0.24)",
    "rgba(245, 158, 11, 0.24)",
    "rgba(14, 165, 233, 0.24)",
    "rgba(139, 92, 246, 0.24)",
    "rgba(239, 68, 68, 0.22)",
    "rgba(20, 184, 166, 0.24)",
];

const benchmarkBorderColors = [
    "rgb(67, 97, 238)",
    "rgb(247, 37, 133)",
    "rgb(16, 185, 129)",
    "rgb(245, 158, 11)",
    "rgb(14, 165, 233)",
    "rgb(139, 92, 246)",
    "rgb(239, 68, 68)",
    "rgb(20, 184, 166)",
];
// released: the paper's first public date ("YYYY-MM": arXiv v1 first, else the official release), for the
// score-vs-date plots.
const benchmarkMethodInfo = {
    "DA3-L": {
        released: "2025-11",
        venue: "ICLR 2026",
        links: [["Hugging Face", "https://huggingface.co/depth-anything/DA3-LARGE-1.1"]],
    },
    "DA3-G": {
        released: "2025-11",
        venue: "ICLR 2026",
        links: [["Hugging Face", "https://huggingface.co/depth-anything/DA3-GIANT-1.1"]],
    },
    "DA3-NGL": {
        released: "2025-11",
        venue: "ICLR 2026",
        links: [["Hugging Face", "https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1"]],
    },
    VGGT: {
        released: "2025-03",
        venue: "CVPR 2025",
        links: [["Project", "https://vgg-t.github.io/"], ["Paper", "https://arxiv.org/abs/2503.11651"]],
    },
    PI3_X: {
        released: "2025-07",
        venue: "ICLR 2026",
        links: [["Project", "https://yyfz.github.io/pi3/"], ["Paper", "https://arxiv.org/abs/2507.13347"]],
    },
    FASTVGGT: {
        released: "2025-09",
        venue: "CVPR 2026",
        links: [["Paper", "https://arxiv.org/abs/2509.02560"]],
    },
    COME: {
        released: "2025-11",
        venue: "CVPR 2026",
        links: [["Project", "https://co-me-tokens.github.io/"], ["Paper", "https://arxiv.org/abs/2511.14751"]],
    },
    CUT3R: {
        released: "2025-01",
        venue: "CVPR 2025 Oral",
        links: [["Project", "https://cut3r.github.io/"], ["Paper", "https://arxiv.org/abs/2501.12387"]],
    },
    FAST3R: {
        released: "2025-01",
        venue: "CVPR 2025",
        links: [["Project", "https://fast3r-3d.github.io/"], ["Paper", "https://arxiv.org/abs/2501.13928"]],
    },
    VGGT_OMEGA: {
        released: "2026-05",
        venue: "CVPR 2026",
        links: [["Project", "http://vggt-omega.github.io/"], ["Paper", "https://arxiv.org/abs/2605.15195"]],
    },
    MAPANYTHING: {
        released: "2025-09",
        venue: "3DV 2026",
        links: [["Project", "https://map-anything.github.io/"], ["Paper", "https://arxiv.org/abs/2509.13414"]],
    },
    OMNIVGGT: {
        released: "2025-11",
        venue: "CVPR 2026 Highlight",
        links: [["Project", "https://livioni.github.io/OmniVGGT-official"], ["Paper", "https://arxiv.org/abs/2511.10560"]],
    },
    DEJAVIEW: {
        released: "2026-05",
        venue: "arXiv 2026",
        links: [["Code", "https://github.com/NVlabs/dvlt"], ["Paper", "https://arxiv.org/abs/2605.30215"]],
    },
    IGGT: {
        released: "2025-10",
        venue: "CVPR 2026",
        links: [["Code", "https://github.com/lifuguan/IGGT"], ["Paper", "https://arxiv.org/abs/2510.22706"]],
    },
    IGGT4D: {
        released: "2026-07",
        venue: "arXiv 2026",
        links: [["Code", "https://github.com/lifuguan/IGGT4D"], ["Paper", "https://arxiv.org/abs/2607.19228"]],
    },
    POW3R: {
        released: "2025-03",
        venue: "CVPR 2025",
        links: [["Project", "https://europe.naverlabs.com/pow3r"], ["Paper", "https://arxiv.org/abs/2503.17316"]],
    },
    DUST3R: {
        released: "2023-12",
        venue: "CVPR 2024",
        links: [["Project", "https://dust3r.europe.naverlabs.com/"], ["Paper", "https://arxiv.org/abs/2312.14132"]],
    },
    MAST3R: {
        released: "2024-06",
        venue: "ECCV 2024",
        links: [["Project", "https://europe.naverlabs.com/blog/mast3r-matching-and-stereo-3d-reconstruction/"], ["Paper", "https://arxiv.org/abs/2406.09756"]],
    },
    ROBUSTVGGT: {
        released: "2025-12",
        venue: "CVPR 2026",
        links: [["Code", "https://github.com/facebookresearch/RobustVGGT"], ["Paper", "https://arxiv.org/abs/2512.04012"]],
    },
    DEPTHVLM: {
        released: "2026-05",
        venue: "arXiv 2026",
        links: [["Project", "https://depthvlm.github.io/"], ["Paper", "https://arxiv.org/abs/2605.15876"]],
    },
    VLM3: {
        venue: "arXiv 2026",
        links: [["Code", "https://github.com/facebookresearch/VLM3"]],
    },
    DEPTHLM: {
        released: "2025-09",
        venue: "ICLR 2026 Oral",
        links: [["Code", "https://github.com/facebookresearch/DepthLM_Official"], ["Paper", "https://arxiv.org/abs/2509.25413"]],
    },
    SENSENOVA_VISION: {
        released: "2026-07",
        venue: "Technical Report 2026",
        links: [["Code", "https://github.com/SenseNova/SenseNova-Vision"], ["Model", "https://huggingface.co/sensenova/SenseNova-Vision-7B-MoT"]],
    },
    QWEN3_VL: {
        released: "2025-11",
        venue: "Technical Report 2025",
        links: [["Code", "https://github.com/QwenLM/Qwen3-VL"], ["Paper", "https://arxiv.org/abs/2511.21631"], ["Model", "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct"]],
    },
    MOGE: {
        released: "2024-10",
        venue: "CVPR 2025 Oral",
        links: [["Code", "https://github.com/microsoft/MoGe"], ["Model", "https://huggingface.co/Ruicheng/moge-vitl"]],
    },
    MOGE2: {
        released: "2025-07",
        links: [["Code", "https://github.com/microsoft/MoGe"], ["Model", "https://huggingface.co/Ruicheng/moge-2-vitl"]],
    },
    MOGE3: {
        released: "2026-07",
        links: [["Code", "https://github.com/microsoft/MoGe"], ["Paper", "https://arxiv.org/abs/2607.17967"], ["Model", "https://huggingface.co/Ruicheng/moge-3-vitl"]],
    },
    MARIGOLD: {
        released: "2023-12",
        venue: "CVPR 2024",
        links: [["Code", "https://github.com/prs-eth/Marigold"], ["Model", "https://huggingface.co/prs-eth/marigold-depth-v1-0"]],
    },
    MARIGOLD_V2: {  // the diffusion-transformer Marigold V2, not the v1-1 checkpoint (MARIGOLD_V1_1)
        released: "2026-09",
        links: [["Paper", "https://arxiv.org/abs/2609.08084"]],
    },
    LOTUS_D: {
        released: "2024-09",
        venue: "ICLR 2025",
        links: [["Code", "https://github.com/EnVision-Research/Lotus"], ["Model", "https://huggingface.co/jingheya/lotus-depth-d-v2-0-disparity"]],
    },
    DEPTH_PRO: {
        released: "2024-10",
        venue: "ICLR 2025",
        links: [["Code", "https://github.com/apple/ml-depth-pro"], ["Model", "https://huggingface.co/apple/DepthPro"]],
    },
    UNIDEPTH_V2: {
        released: "2025-02",
        links: [["Code", "https://github.com/lpiccinelli-eth/UniDepth"], ["Model", "https://huggingface.co/lpiccinelli/unidepth-v2-vitl14"]],
    },
    INFINIDEPTH: {
        released: "2026-01",
        venue: "CVPR 2026",
        links: [["Code", "https://github.com/zju3dv/InfiniDepth"], ["Paper", "https://arxiv.org/abs/2601.03252"], ["Model", "https://huggingface.co/ritianyu/InfiniDepth"]],
    },
    DVD: {
        released: "2026-03",
        links: [["Code", "https://github.com/EnVision-Research/DVD"], ["Paper", "https://arxiv.org/abs/2603.12250"], ["Model", "https://huggingface.co/FayeHongfeiZhang/DVD"]],
    },
    DEPTHCRAFTER: {
        released: "2024-09",
        venue: "CVPR 2025",
        links: [["Code", "https://github.com/Tencent/DepthCrafter"], ["Paper", "https://arxiv.org/abs/2409.02095"], ["Model", "https://huggingface.co/tencent/DepthCrafter"]],
    },
    CHRONODEPTH: {
        released: "2024-06",
        links: [["Code", "https://github.com/jiahao-shao1/ChronoDepth"], ["Paper", "https://arxiv.org/abs/2406.01493"], ["Model", "https://huggingface.co/jhshao/ChronoDepth-v1"]],
    },
    AMB3R: {
        released: "2025-11",
        links: [["Code", "https://github.com/HengyiWang/amb3r"], ["Paper", "https://arxiv.org/abs/2511.20343"]],
    },
    GENCEPTION_DEPTH: {
        released: "2026-07",
        links: [["Paper", "https://arxiv.org/abs/2607.09024"]],
    },
    MARIGOLD_V1_1: {
        released: "2025-05",
        links: [["Code", "https://github.com/prs-eth/Marigold"], ["Paper", "https://arxiv.org/abs/2505.09358"], ["Model", "https://huggingface.co/prs-eth/marigold-depth-v1-1"]],
    },
    GEONEXT_OFFICIAL_SVD: {
        released: "2026-08",
        links: [["Code", "https://github.com/Creative-Intelligence-Studio/GeoNeXt"], ["Paper", "https://arxiv.org/abs/2608.28549"]],
    },
    VGGT_LONG: {
        released: "2025-07",
        links: [["Code", "https://github.com/DengKaiCQ/VGGT-Long"], ["Paper", "https://arxiv.org/abs/2507.16443"]],
    },
    PI_LONG: {
        released: "2025-09",
        links: [["Code", "https://github.com/DengKaiCQ/Pi-Long"]],
    },
    VGGT_SLAM: {
        released: "2025-05",
        links: [["Code", "https://github.com/MIT-SPARK/VGGT-SLAM"], ["Paper", "https://arxiv.org/abs/2505.12549"]],
    },
    LOGER: {
        released: "2026-03",
        links: [["Code", "https://github.com/Junyi42/LoGeR"], ["Paper", "https://arxiv.org/abs/2603.03269"]],
    },
    ZIPMAP: {
        released: "2026-03",
        links: [["Code", "https://github.com/Haian-Jin/ZipMap"], ["Paper", "https://arxiv.org/abs/2603.04385"]],
    },
    LINGBOT_MAP: {
        released: "2026-04",
        links: [["Code", "https://github.com/robbyant/lingbot-map"], ["Paper", "https://arxiv.org/abs/2604.14141"]],
    },
    R3: {
        released: "2026-05",
        links: [["Code", "https://github.com/KevinXu02/R3"], ["Paper", "https://arxiv.org/abs/2605.26519"]],
    },
    SCAL3R: {
        released: "2026-04",
        links: [["Code", "https://github.com/zju3dv/Scal3R"], ["Paper", "https://arxiv.org/abs/2604.08542"]],
    },
    ABOT_RECON: {
        released: "2026-08",
        links: [["Code", "https://github.com/amap-cvlab/ABot-Recon"], ["Paper", "https://arxiv.org/abs/2608.27529"]],
    },
    CLAUDE_OPUS_5_5: {
        released: "2026-09",
        links: [],
    },
    GPT_6_ASTRA: {
        released: "2026-09",
        links: [],
    },
    GEMINI_4_ARGON: {
        released: "2026-09",
        links: [],
    },
};

const benchmarkMetricDefinitions = {
    absrel_percent: { group: "Depth", equation: "100 / N * sum_i |d_i - d_hat_i| / d_i" },
    squared_relative_difference: { group: "Depth", equation: "1 / N * sum_i (d_i - d_hat_i)^2 / d_i" },
    rmse_linear: { group: "Depth", equation: "sqrt(1 / N * sum_i (d_i - d_hat_i)^2)" },
    rmse_log: { group: "Depth", equation: "sqrt(1 / N * sum_i (log d_i - log d_hat_i)^2)" },
    log10: { group: "Depth", equation: "1 / N * sum_i |log10 d_i - log10 d_hat_i|" },
    i_rmse: { group: "Depth", equation: "sqrt(1 / N * sum_i (1 / d_i - 1 / d_hat_i)^2)" },
    silog_rmse: { group: "Depth", equation: "sqrt(mean(e_i^2) - mean(e_i)^2), e_i = log d_hat_i - log d_i" },
    delta125_percent: { group: "Depth", equation: "100 * mean(max(d_hat_i / d_i, d_i / d_hat_i) < 1.25)" },
    delta125_2_percent: { group: "Depth", equation: "100 * mean(max(d_hat_i / d_i, d_i / d_hat_i) < 1.25^2)" },
    delta125_3_percent: { group: "Depth", equation: "100 * mean(max(d_hat_i / d_i, d_i / d_hat_i) < 1.25^3)" },
    delta103_percent: { group: "Depth", equation: "100 * mean(max(d_hat_i / d_i, d_i / d_hat_i) < 1.03)" },
    mae_normal_deg: { group: "Depth Geometry", equation: "1 / N * sum_i acos(n_i · n_hat_i), in degrees" },
    accuracy: { group: "MVRecon", equation: "1 / |P_hat| * sum_p min_q ||p - q||_2" },
    completion: { group: "MVRecon", equation: "1 / |P| * sum_q min_p ||p - q||_2" },
    overall: { group: "MVRecon", equation: "(accuracy + completion) / 2" },
    precision: { group: "MVRecon", equation: "mean_p[min_q ||p - q||_2 < tau]" },
    recall: { group: "MVRecon", equation: "mean_q[min_p ||p - q||_2 < tau]" },
    fscore: { group: "MVRecon", equation: "2 * precision * recall / (precision + recall)" },
    ate_rmse: { group: "SfM", equation: "sqrt(1 / N * sum_i ||t_i - aligned(t_hat_i)||_2^2)" },
    slam_ate_rmse: { group: "SLAM", equation: "sqrt(1 / N * sum_i ||t_i - Sim3(t_hat_i)||_2^2), metres of the GT frame" },
    slam_drift_pct: { group: "SLAM", equation: "100 * sum_w RPE_t,1s(w) / sum_w GT path length(w)" },
    slam_rpe_trans_1s: { group: "SLAM", equation: "RMSE translational relative pose error over 1 s windows (m)" },
    slam_rpe_rot_1s: { group: "SLAM", equation: "RMSE rotational relative pose error over 1 s windows (deg)" },
    slam_rpe_trans_step: { group: "SLAM", equation: "RMSE translational relative pose error between consecutive inputs (m)" },
    slam_rpe_rot_step: { group: "SLAM", equation: "RMSE rotational relative pose error between consecutive inputs (deg)" },
    slam_auc_1: { group: "SLAM", equation: "area under the pairwise pose-accuracy curve (max of relative rotation and translation-direction error) up to 1 deg" },
    slam_auc_3: { group: "SLAM", equation: "area under the pairwise pose-accuracy curve (max of relative rotation and translation-direction error) up to 3 deg" },
    slam_auc_5: { group: "SLAM", equation: "area under the pairwise pose-accuracy curve (max of relative rotation and translation-direction error) up to 5 deg" },
    slam_auc_10: { group: "SLAM", equation: "area under the pairwise pose-accuracy curve (max of relative rotation and translation-direction error) up to 10 deg" },
    slam_auc_20: { group: "SLAM", equation: "area under the pairwise pose-accuracy curve (max of relative rotation and translation-direction error) up to 20 deg" },
    slam_auc_30: { group: "SLAM", equation: "area under the pairwise pose-accuracy curve (max of relative rotation and translation-direction error) up to 30 deg" },
    slam_completeness: { group: "SLAM", equation: "fraction of input frames with a returned pose" },
    ate_mean: { group: "SfM", equation: "1 / N * sum_i ||t_i - aligned(t_hat_i)||_2" },
    ate_median: { group: "SfM", equation: "median_i ||t_i - aligned(t_hat_i)||_2" },
    relative_pose_error_deg_mean: { group: "SfM", equation: "mean relative pose angular error in degrees" },
    relative_pose_axis_error_deg_mean: { group: "SfM", equation: "mean relative pose axis angular error in degrees" },
    relative_rotation_error_deg_mean: { group: "SfM", equation: "mean angle(R_ij, R_hat_ij) in degrees" },
    relative_rotation_error_deg_median: { group: "SfM", equation: "median angle(R_ij, R_hat_ij) in degrees" },
    relative_translation_error_deg_mean: { group: "SfM", equation: "mean angle(t_ij, t_hat_ij) in degrees" },
    relative_translation_error_deg_median: { group: "SfM", equation: "median angle(t_ij, t_hat_ij) in degrees" },
    relative_translation_axis_error_deg_mean: { group: "SfM", equation: "mean translation-axis angular error in degrees" },
    relative_translation_axis_error_deg_median: { group: "SfM", equation: "median translation-axis angular error in degrees" },
    RRA_at_5: { group: "SfM", equation: "mean(relative rotation error < 5 deg)" },
    RRA_at_15: { group: "SfM", equation: "mean(relative rotation error < 15 deg)" },
    RRA_at_30: { group: "SfM", equation: "mean(relative rotation error < 30 deg)" },
    RTA_at_5: { group: "SfM", equation: "mean(relative translation-axis error < 5 deg)" },
    RTA_at_15: { group: "SfM", equation: "mean(relative translation-axis error < 15 deg)" },
    RTA_at_30: { group: "SfM", equation: "mean(relative translation-axis error < 30 deg)" },
    auc_1: { group: "SfM", equation: "area under pose-accuracy curve up to 1 deg" },
    auc_3: { group: "SfM", equation: "area under pose-accuracy curve up to 3 deg" },
    auc_5: { group: "SfM", equation: "area under pose-accuracy curve up to 5 deg" },
    auc_10: { group: "SfM", equation: "area under pose-accuracy curve up to 10 deg" },
    auc_20: { group: "SfM", equation: "area under pose-accuracy curve up to 20 deg" },
    auc_30: { group: "SfM", equation: "area under pose-accuracy curve up to 30 deg" },
    mAA_30: { group: "SfM", equation: "mean average accuracy over pose thresholds up to 30 deg" },
};

const benchmarkViewNames = ["home", "overview", "download", "leaderboard", "artifacts"];
// Hashes of views that were removed or merged into another, so old links still land somewhere. Task Metrics is the
// Leaderboard's task tabs now (#tasks opens the first task).
const benchmarkViewAliases = { datasets: "download", coverage: "overview", tasks: "leaderboard", contributors: "home" };

function benchmarkViewFromHash() {
    // The hash may carry view state after "?" (e.g. #artifacts?compare=trajectory&..., #leaderboard?task=SfM).
    const view = window.location.hash.replace(/^#/, "").split("?")[0];
    return benchmarkViewAliases[view] || view;
}

// #leaderboard?task=<Task>[&dataset=<dataset>] opens that task's tables; #leaderboard alone the overall ranking.
function applyBenchmarkLeaderboardHash() {
    const [view, query = ""] = window.location.hash.replace(/^#/, "").split("?");
    if (view !== "leaderboard" && view !== "tasks") return;
    const params = new URLSearchParams(query);
    const task = params.get("task");
    benchmarkLeaderboardOverall = view === "leaderboard" && !task;
    if (task) benchmarkActiveTaskMetricTab = task;
    if (task && params.get("dataset")) setBenchmarkDatasetTab(task, params.get("dataset"));
}

function writeBenchmarkLeaderboardHash() {
    if (benchmarkActiveViewName() !== "leaderboard") return;
    const params = new URLSearchParams();
    if (!benchmarkLeaderboardOverall) {
        params.set("task", activeBenchmarkTaskMetricTab());
        const dataset = activeBenchmarkDatasetTab();
        if (dataset !== "all") params.set("dataset", dataset);
    }
    const hash = `#leaderboard${params.toString() ? `?${params}` : ""}`;
    if (window.location.hash !== hash) window.history.replaceState(null, "", hash);
}

document.addEventListener("DOMContentLoaded", function() {
    initBenchmarkThemeToggle();
    initBenchmarkControls();
    initBenchmarkOverviewFilters();
    if (typeof initChartExport === "function") initChartExport({ redraw: renderBenchmarkActiveView, prefix: "ffb" });
    const initialView = benchmarkViewFromHash();
    applyBenchmarkLeaderboardHash();
    if (benchmarkViewNames.includes(initialView)) {
        switchBenchmarkView(initialView, { updateHash: false });
    }
    window.addEventListener("hashchange", () => {
        const view = benchmarkViewFromHash();
        applyBenchmarkLeaderboardHash();
        if (benchmarkViewNames.includes(view) && view !== benchmarkActiveViewName()) {
            switchBenchmarkView(view, { updateHash: false });
        } else if (view === "leaderboard") {
            renderBenchmarkActiveView();
        }
    });
    loadBenchmarkData();
});

function initBenchmarkThemeToggle() {
    const toggle = document.getElementById("theme-toggle");
    if (!toggle) return;
    toggle.checked = document.documentElement.getAttribute("data-theme") === "dark";
    toggle.addEventListener("change", () => {
        const theme = toggle.checked ? "dark" : "light";
        document.documentElement.setAttribute("data-theme", theme);
        try {
            localStorage.setItem("benchmark-dashboard-theme", theme);
        } catch (_error) {
            // Storage can be unavailable (private windows); the toggle still works for this page view.
        }
        // Chart.js bakes text colors in at render time, so redraw the visible view.
        renderBenchmarkActiveView();
    });
}

function initBenchmarkControls() {
    initBenchmarkReleaseWidth();
    const radarMode = document.getElementById("benchmark-radar-mode");
    const radarLimit = document.getElementById("benchmark-radar-limit");
    const familyFilter = document.getElementById("benchmark-family-filter");
    const methodSearch = document.getElementById("benchmark-method-search");
    const reloadButton = document.getElementById("benchmark-reload");
    const rankingSearch = document.getElementById("benchmark-ranking-search");
    const rankingSort = document.getElementById("benchmark-ranking-sort");
    const exportRanking = document.getElementById("benchmark-export-ranking");
    const exportTask = document.getElementById("benchmark-export-task");

    if (radarMode) radarMode.addEventListener("change", renderBenchmarkRadar);
    if (radarLimit) radarLimit.addEventListener("change", renderBenchmarkRadar);
    if (familyFilter) familyFilter.addEventListener("change", renderBenchmarkTaskTables);
    if (methodSearch) methodSearch.addEventListener("input", renderBenchmarkTaskTables);
    if (reloadButton) {
        reloadButton.addEventListener("click", loadBenchmarkData);
    }
    if (rankingSearch) rankingSearch.addEventListener("input", renderBenchmarkRankingTable);
    if (rankingSort) {
        rankingSort.addEventListener("change", () => {
            benchmarkRankingSortState = sortStateFromBenchmarkRankingSelect(rankingSort.value);
            renderBenchmarkRankingTable();
        });
    }
    const rankingHead = document.getElementById("benchmark-ranking-head");
    if (rankingHead) rankingHead.addEventListener("click", handleBenchmarkRankingHeaderClick);
    const taskTables = document.getElementById("benchmark-task-tables");
    if (taskTables) taskTables.addEventListener("click", handleBenchmarkTaskMetricHeaderClick);
    const taskTabs = document.getElementById("benchmark-task-tabs");
    if (taskTabs) taskTabs.addEventListener("click", handleBenchmarkTaskTabClick);
    const datasetTabs = document.getElementById("benchmark-dataset-tabs");
    if (datasetTabs) datasetTabs.addEventListener("click", handleBenchmarkDatasetTabClick);
    if (exportRanking) exportRanking.addEventListener("click", exportBenchmarkRankingCsv);
    if (exportTask) exportTask.addEventListener("click", exportBenchmarkTaskCsv);
    document.querySelectorAll("[data-benchmark-view]").forEach(tab => {
        tab.addEventListener("click", event => {
            event.preventDefault();
            switchBenchmarkView(tab.dataset.benchmarkView || "overview");
        });
    });
    // The navbar logo is the way home (there is no Home tab): it switches view and scrolls to the top like a tab.
    document.querySelectorAll("[data-benchmark-home]").forEach(link => {
        link.addEventListener("click", event => {
            event.preventDefault();
            switchBenchmarkView("home");
        });
    });
}

function loadBenchmarkData() {
    Promise.all([
        fetchBenchmarkJson("data/results.json"),
        fetchBenchmarkJson("data/results_live.json", true),
    ])
        .then(([baseData, liveData]) => {
            benchmarkData = liveData ? mergeBenchmarkData(baseData, liveData) : baseData;
            renderBenchmarkDashboard();
        })
        .catch(error => {
            console.error("Error loading benchmark data:", error);
            const container = document.getElementById("benchmark-task-tables");
            if (container) {
                container.innerHTML = `<div class="benchmark-empty">${escapeBenchmarkHtml(error.message)}</div>`;
            }
        });
}

function fetchBenchmarkJson(path, optional = false) {
    return fetch(path, { cache: "no-cache" }).then(response => {
        if (!response.ok) {
            if (optional) return null;
            throw new Error(`Benchmark results JSON is missing: ${path}`);
        }
        return response.json();
    }).catch(error => {
        if (optional) return null;
        throw error;
    });
}

function mergeBenchmarkData(baseData, liveData) {
    if (!baseData) return liveData;
    if (!liveData) return baseData;
    const merged = JSON.parse(JSON.stringify(baseData));
    const union = (a, b) => Array.from(new Set([...(a || []), ...(b || [])]));

    merged.generated_at = liveData.generated_at || merged.generated_at;
    merged.source_tsv = baseData.source_tsv;
    merged.source_report_roots = union(baseData.source_report_roots, liveData.source_report_roots);
    merged.summary = { ...(baseData.summary || {}) };
    merged.methods = union(baseData.methods, liveData.methods);
    merged.tasks = union(baseData.tasks, liveData.tasks);
    merged.datasets = union(baseData.datasets, liveData.datasets);
    merged.method_display_names = { ...(baseData.method_display_names || {}), ...(liveData.method_display_names || {}) };
    merged.task_display_names = { ...(baseData.task_display_names || {}), ...(liveData.task_display_names || {}) };
    merged.dataset_display_names = { ...(baseData.dataset_display_names || {}), ...(liveData.dataset_display_names || {}) };
    merged.metric_catalog = { ...(baseData.metric_catalog || {}), ...(liveData.metric_catalog || {}) };
    merged.method_types = { ...(baseData.method_types || {}), ...(liveData.method_types || {}) };
    merged.method_type_labels = { ...(baseData.method_type_labels || {}), ...(liveData.method_type_labels || {}) };
    merged.task_catalog = { ...(baseData.task_catalog || {}), ...(liveData.task_catalog || {}) };
    merged.method_statuses = liveData.method_statuses || merged.method_statuses || [];

    if (liveData.task_tables) {
        merged.task_tables = merged.task_tables || {};
        for (const [task, taskTable] of Object.entries(liveData.task_tables)) {
            const baseTaskTable = merged.task_tables[task] || {};
            const mergedMethods = new Map((baseTaskTable.methods || []).map(row => [row.method, row]));
            for (const row of taskTable.methods || []) {
                if (row.missing && mergedMethods.has(row.method)) continue;  // a placeholder never hides a result
                mergedMethods.set(row.method, row);
            }
            const mergedDatasetTables = { ...(baseTaskTable.dataset_tables || {}) };
            for (const [dataset, datasetTable] of Object.entries(taskTable.dataset_tables || {})) {
                const baseDatasetTable = mergedDatasetTables[dataset] || {};
                const datasetMethods = new Map((baseDatasetTable.methods || []).map(row => [row.method, row]));
                for (const row of datasetTable.methods || []) {
                    if (row.missing && datasetMethods.has(row.method)) continue;
                    datasetMethods.set(row.method, row);
                }
                mergedDatasetTables[dataset] = {
                    ...baseDatasetTable,
                    ...datasetTable,
                    methods: Array.from(datasetMethods.values()),
                };
            }
            merged.task_tables[task] = {
                ...baseTaskTable,
                ...taskTable,
                datasets: union(baseTaskTable.datasets, taskTable.datasets),
                methods: Array.from(mergedMethods.values()),
                dataset_tables: mergedDatasetTables,
            };
        }
    }

    if (liveData.ranking?.rows?.length) {
        const rowsByMethod = new Map((merged.ranking?.rows || []).map(row => [row.method, row]));
        for (const row of liveData.ranking.rows || []) {
            rowsByMethod.set(row.method, row);
        }
        merged.ranking = {
            ...(merged.ranking || {}),
            ...liveData.ranking,
            rows: Array.from(rowsByMethod.values()),
            tasks: union(baseData.ranking?.tasks, liveData.ranking.tasks),
            datasets: union(baseData.ranking?.datasets, liveData.ranking.datasets),
        };
    }
    if (liveData.task_ranking?.rows?.length) {
        const rowsByMethod = new Map((merged.task_ranking?.rows || []).map(row => [row.method, row]));
        for (const row of liveData.task_ranking.rows || []) {
            rowsByMethod.set(row.method, row);
        }
        merged.task_ranking = {
            ...(merged.task_ranking || {}),
            ...liveData.task_ranking,
            rows: Array.from(rowsByMethod.values()),
            tasks: union(baseData.task_ranking?.tasks, liveData.task_ranking.tasks),
            datasets: union(baseData.task_ranking?.datasets, liveData.task_ranking.datasets),
        };
    }
    if (liveData.radar?.method_scores) {
        merged.radar = {
            ...(merged.radar || {}),
            ...liveData.radar,
            method_scores: {
                ...(merged.radar?.method_scores || {}),
                ...(liveData.radar.method_scores || {}),
            },
        };
    }
    // A live export that carries the coverage matrix (and its collection record) replaces the base one.
    if (liveData.coverage) merged.coverage = liveData.coverage;
    if (liveData.collection) merged.collection = liveData.collection;
    return merged;
}

function parseBenchmarkJsonResponse(response, fallbackMessage) {
    return response.text().then(text => {
        let payload = null;
        if (text) {
            try {
                payload = JSON.parse(text);
            } catch (_error) {
                payload = null;
            }
        }
        if (!response.ok) {
            const message = payload?.error || payload?.message || text || fallbackMessage || `HTTP ${response.status}`;
            throw new Error(message.replace(/^Error:\s*/i, ""));
        }
        if (payload === null) {
            throw new Error(text || fallbackMessage || "Invalid JSON response");
        }
        if (payload.ok === false) {
            const message = payload.error || payload.message || fallbackMessage || "Request failed";
            throw new Error(message.replace(/^Error:\s*/i, ""));
        }
        return payload;
    });
}

function renderBenchmarkDashboard() {
    if (!benchmarkData) return;
    renderBenchmarkSummary();
    populateBenchmarkFilters();
    renderBenchmarkRankingTable();
    renderBenchmarkRadar();
    renderBenchmarkOverallBarChart();
    renderBenchmarkTaskScopedViews();
    renderBenchmarkReleaseScatterCharts();  // home page slide
    renderBenchmarkFlagNotes();
}

function renderBenchmarkTaskScopedViews() {
    renderBenchmarkOverviewFilters();
    renderBenchmarkSummary();
    renderBenchmarkTrackBarCharts();
    renderBenchmarkTaskTables();
}

function renderBenchmarkSummary() {
    const summary = benchmarkData.summary || {};
    setBenchmarkText("benchmark-radar-badge", `${summary.ranked_methods || summary.methods || 0} methods`);
    setBenchmarkText("benchmark-task-table-count", `${summary.tasks || 0} tasks`);
}

function benchmarkBaseMethodName(method) {
    return String(method || "")
        .replace(/_rgb_sparse_depth_cameras$/, "")
        .replace(/_rgb_sparse_depth$/, "")
        .replace(/_rgb_masked_depth_cameras$/, "")
        .replace(/_rgb_masked_depth$/, "")
        .replace(/_rgb_depth$/, "")
        .replace(/_rgb_cameras$/, "");
}

function benchmarkResultRows(rows) {
    // Table rows with results: a "missing" row stands for a required method with none (its cells show "-").
    return (rows || []).filter(row => !row?.missing);
}

function benchmarkTaskMethods(taskTable) {
    // SLAM has no cross-dataset method list (raw metrics are not averaged over datasets); use its range tables.
    if (!taskTable?.range_tables) return benchmarkResultRows(taskTable?.methods);
    const byMethod = new Map();
    for (const rangeTable of Object.values(taskTable.range_tables)) {
        for (const row of rangeTable.methods || []) byMethod.set(row.method, row);
    }
    return Array.from(byMethod.values());
}

// SLAM range tables use "<metric>::<dataset>" columns; they share the base metric's catalog entry.
const BENCHMARK_SLAM_COLUMN_SEPARATOR = "::";

function benchmarkMetricInfo(metricKey) {
    const [baseKey] = String(metricKey || "").split(BENCHMARK_SLAM_COLUMN_SEPARATOR);
    return benchmarkData?.metric_catalog?.[metricKey] || benchmarkData?.metric_catalog?.[baseKey] || {};
}

// Hover text of a metric table cell: what the number is (the rank is already shown as the badge).
function benchmarkMetricCellTitle(metricKey) {
    const info = benchmarkMetricInfo(metricKey);
    const [baseKey] = String(metricKey || "").split(BENCHMARK_SLAM_COLUMN_SEPARATOR);
    const family = document.querySelector(`#benchmark-family-filter option[value="${CSS.escape(info.family || "")}"]`)?.textContent
        || String(info.family || "").replace(/_/g, " ");
    const dataset = benchmarkSlamColumnDataset(metricKey);
    const parts = [
        info.label || baseKey,
        family,
        dataset ? benchmarkDatasetLabel(dataset) : "",
        info.direction === "higher" ? "higher is better" : info.direction === "lower" ? "lower is better" : "",
    ].filter(Boolean);
    const equation = benchmarkMetricDefinitions[baseKey]?.equation;
    return parts.join(" · ") + (equation ? `\n${equation}` : "");
}

function benchmarkSlamColumnDataset(metricKey) {
    const parts = String(metricKey || "").split(BENCHMARK_SLAM_COLUMN_SEPARATOR);
    return parts.length > 1 ? parts[1] : "";
}

function switchBenchmarkView(viewName, { updateHash = true } = {}) {
    const nextView = benchmarkViewNames.includes(viewName) ? viewName : "home";
    if (updateHash && window.location.hash.split("?")[0] !== `#${nextView}`) {
        window.history.replaceState(null, "", `#${nextView}`);
    }
    if (updateHash) window.scrollTo({ top: 0, left: 0, behavior: "auto" });
    document.querySelectorAll("[data-benchmark-view]").forEach(tab => {
        const isActive = tab.dataset.benchmarkView === nextView;
        tab.classList.toggle("active", isActive);
        tab.setAttribute("aria-selected", isActive ? "true" : "false");
    });
    document.querySelectorAll(".benchmark-view-panel").forEach(panel => {
        const isActive = panel.id === `benchmark-view-${nextView}`;
        panel.classList.toggle("active", isActive);
        panel.hidden = !isActive;
    });
    window.setTimeout(renderBenchmarkActiveView, 0);
}

function benchmarkActiveViewName() {
    // From the shown panel, not a navbar tab: Home has no tab (the logo leads there).
    return document.querySelector(".benchmark-view-panel.active")?.id.replace(/^benchmark-view-/, "") || "home";
}

function renderBenchmarkActiveView() {
    if (!benchmarkData) return;
    const view = benchmarkActiveViewName();
    if (view === "home") {
        if (typeof renderSpatialArenaHome === "function") renderSpatialArenaHome(benchmarkData);
        renderBenchmarkReleaseScatterCharts();
    } else if (view === "overview") {
        renderBenchmarkOverviewFilters();
            renderBenchmarkRadar();
        renderBenchmarkOverallBarChart();
        renderBenchmarkTrackBarCharts();
        } else if (view === "leaderboard") {
        renderBenchmarkTaskMetricTabs();
        renderBenchmarkLeaderboardTab();
        writeBenchmarkLeaderboardHash();
    }
}

function populateBenchmarkFilters() {
    renderBenchmarkTaskMetricTabs();
    renderBenchmarkDatasetTabs();
}

function handleBenchmarkTaskTabClick(event) {
    const tab = event.target.closest("[data-benchmark-task-tab]");
    if (!tab) return;
    const key = tab.dataset.benchmarkTaskTab || "";
    benchmarkLeaderboardOverall = key === "overall";
    if (!benchmarkLeaderboardOverall) benchmarkActiveTaskMetricTab = key;
    renderBenchmarkTaskMetricTabs();
    renderBenchmarkLeaderboardTab();
    writeBenchmarkLeaderboardHash();
}

// Show the overall ranking or the selected task's dataset tabs and metric tables.
function renderBenchmarkLeaderboardTab() {
    const overall = document.getElementById("benchmark-leaderboard-overall");
    const task = document.getElementById("benchmark-leaderboard-task");
    if (overall) overall.hidden = !benchmarkLeaderboardOverall;
    if (task) task.hidden = benchmarkLeaderboardOverall;
    if (benchmarkLeaderboardOverall) {
        renderBenchmarkRankingTable();
    } else {
        renderBenchmarkDatasetTabs();
        renderBenchmarkTaskTables();
    }
}

function activeBenchmarkTaskMetricTab() {
    const tasks = benchmarkData?.tasks || [];
    if (!tasks.length) return "";
    if (!benchmarkActiveTaskMetricTab || !tasks.includes(benchmarkActiveTaskMetricTab)) {
        benchmarkActiveTaskMetricTab = tasks[0];
    }
    return benchmarkActiveTaskMetricTab;
}

function renderBenchmarkTaskMetricTabs() {
    const container = document.getElementById("benchmark-task-tabs");
    if (!container || !benchmarkData) return;
    const tasks = benchmarkData.tasks || [];
    const activeTask = benchmarkLeaderboardOverall ? "" : activeBenchmarkTaskMetricTab();
    const overallActive = benchmarkLeaderboardOverall;
    const overallCount = (benchmarkData.ranking?.rows || benchmarkData.task_ranking?.rows || []).length;
    container.innerHTML = `
        <button type="button" class="benchmark-task-tab benchmark-overall-tab${overallActive ? " active" : ""}" data-benchmark-task-tab="overall" role="tab" aria-selected="${overallActive ? "true" : "false"}">
            <i class="fas fa-medal" aria-hidden="true"></i>
            <span>Overall</span>
            <span class="benchmark-task-tab-count">${overallCount}</span>
        </button>
    ` + tasks.map(task => {
        const isActive = task === activeTask;
        const taskTable = benchmarkData.task_tables?.[task] || {};
        const methodCount = benchmarkTaskMethods(taskTable).length;
        return `
            <button type="button" class="benchmark-task-tab${isActive ? " active" : ""}" data-benchmark-task-tab="${escapeBenchmarkAttr(task)}" role="tab" aria-selected="${isActive ? "true" : "false"}">
                <span>${escapeBenchmarkHtml(benchmarkTaskLabel(task))}</span>
                <span class="benchmark-task-tab-count">${methodCount}</span>
            </button>
        `;
    }).join("");
}

// SLAM dataset tables are keyed "<dataset>@<range>" (benchmark_analyzer.SLAM_RANGE_SEPARATOR); one tab per dataset.
const BENCHMARK_SLAM_RANGE_SEPARATOR = "@";

function benchmarkBaseDataset(datasetKey) {
    return String(datasetKey || "").split(BENCHMARK_SLAM_RANGE_SEPARATOR)[0];
}

function benchmarkDatasetTabKeys(taskTable) {
    // SLAM: datasets in protocol order (as the range tables list them); other tasks: alphabetical.
    if (taskTable?.range_tables) {
        const ordered = [];
        for (const slamRange of taskTable.ranges || Object.keys(taskTable.range_tables)) {
            for (const dataset of taskTable.range_tables[slamRange]?.datasets || []) {
                if (!ordered.includes(dataset)) ordered.push(dataset);
            }
        }
        for (const key of taskTable.datasets || []) {
            const base = benchmarkBaseDataset(key);
            if (!ordered.includes(base)) ordered.push(base);
        }
        return ordered;
    }
    return Array.from(new Set(taskTable?.datasets || [])).sort();
}

function benchmarkDatasetTableKeys(taskTable, datasetTab) {
    // The dataset tables a dataset tab shows: the dataset itself, or (SLAM) one per range it takes part in.
    if (!taskTable?.range_tables) return taskTable?.dataset_tables?.[datasetTab] ? [datasetTab] : [];
    const ranges = taskTable.ranges || Object.keys(taskTable.range_tables);
    return Object.keys(taskTable.dataset_tables || {})
        .filter(key => benchmarkBaseDataset(key) === datasetTab)
        .sort((a, b) => ranges.indexOf(a.split(BENCHMARK_SLAM_RANGE_SEPARATOR)[1])
            - ranges.indexOf(b.split(BENCHMARK_SLAM_RANGE_SEPARATOR)[1]));
}

function activeBenchmarkDatasetTab() {
    const task = activeBenchmarkTaskMetricTab();
    const taskTable = benchmarkData?.task_tables?.[task];
    const chosen = benchmarkActiveDatasetTabs[task] || "all";
    return chosen === "all" || benchmarkDatasetTabKeys(taskTable).includes(chosen) ? chosen : "all";
}

function setBenchmarkDatasetTab(task, datasetTab) {
    benchmarkActiveDatasetTabs[task] = datasetTab || "all";
}

function renderBenchmarkDatasetTabs() {
    const container = document.getElementById("benchmark-dataset-tabs");
    if (!container || !benchmarkData) return;
    const task = activeBenchmarkTaskMetricTab();
    const taskTable = benchmarkData.task_tables?.[task];
    const active = activeBenchmarkDatasetTab();
    const countMethods = rows => (rows || []).length;
    const tab = (key, label, count) => `
        <button type="button" class="benchmark-task-tab benchmark-dataset-tab${key === active ? " active" : ""}" data-benchmark-dataset-tab="${escapeBenchmarkAttr(key)}" role="tab" aria-selected="${key === active ? "true" : "false"}">
            <span>${escapeBenchmarkHtml(label)}</span>
            <span class="benchmark-task-tab-count">${count}</span>
        </button>
    `;
    const keys = benchmarkDatasetTabKeys(taskTable);
    container.innerHTML = taskTable
        ? tab("all", "Overall", countMethods(benchmarkTaskMethods(taskTable))) + keys.map(key => {
            const methods = new Set();
            for (const tableKey of benchmarkDatasetTableKeys(taskTable, key)) {
                for (const row of benchmarkResultRows(taskTable.dataset_tables[tableKey].methods)) {
                    methods.add(row.method);
                }
            }
            return tab(key, benchmarkDatasetLabel(key), methods.size);
        }).join("")
        : "";
}

function handleBenchmarkDatasetTabClick(event) {
    const tab = event.target.closest("[data-benchmark-dataset-tab]");
    if (!tab) return;
    setBenchmarkDatasetTab(activeBenchmarkTaskMetricTab(), tab.dataset.benchmarkDatasetTab);
    renderBenchmarkDatasetTabs();
    renderBenchmarkTaskTables();
    writeBenchmarkLeaderboardHash();
}

function renderBenchmarkRankingTable() {
    const head = document.getElementById("benchmark-ranking-head");
    const body = document.getElementById("benchmark-ranking-body");
    if (!head || !body || !benchmarkData) return;

    const tasks = benchmarkData.ranking?.tasks || benchmarkData.task_ranking?.tasks || [];
    const rows = getBenchmarkRankingRows();
    benchmarkVisibleRankingRows = rows;
    setBenchmarkText("benchmark-ranking-count", `${rows.length} methods`);

    head.innerHTML = `
        <tr>
            ${renderBenchmarkSortableHeader("Rank", "rank", "base", "numeric")}
            ${renderBenchmarkSortableHeader("Method", "method", "base")}
            ${renderBenchmarkSortableHeader("Score", "score", "base", "numeric")}
            ${renderBenchmarkSortableHeader("Avg Rank", "average_rank", "base", "numeric")}
            ${renderBenchmarkSortableHeader("Coverage", "coverage", "base", "numeric")}
            ${renderBenchmarkSortableHeader("Wins", "first_place_wins", "base", "numeric")}
            ${tasks.map(task => renderBenchmarkSortableHeader(benchmarkTaskLabel(task), task, "task", "numeric")).join("")}
        </tr>
    `;

    body.innerHTML = rows.map((row, index) => {
        const displayRank = index + 1;
        const taskCells = tasks.map(task => {
            const taskScore = row.task_scores?.[task];
            const rankBadge = renderBenchmarkRankBadge(taskScore?.rank);
            const scoreLabel = isFiniteNumber(taskScore?.score) ? `${taskScore.score.toFixed(1)}` : "-";
            return `<td class="numeric">${scoreLabel} ${rankBadge}</td>`;
        }).join("");
        return `
            <tr>
                <td class="numeric">${renderBenchmarkRankBadge(displayRank)}</td>
                <td class="benchmark-method-name">${benchmarkMethodNameHtml(row)}</td>
                <td class="numeric">${formatBenchmarkNumber(row.score, 2)}</td>
                <td class="numeric">${formatBenchmarkNumber(row.average_rank, 2)}</td>
                <td class="numeric">${row.coverage ?? 0}</td>
                <td class="numeric">${row.first_place_wins ?? 0}</td>
                ${taskCells}
            </tr>
        `;
    }).join("");
}

function getBenchmarkRankingRows() {
    const methodSearch = (document.getElementById("benchmark-ranking-search")?.value || "").trim().toLowerCase();
    const rows = (benchmarkData.ranking?.rows || benchmarkData.task_ranking?.rows || [])
        .filter(row => !methodSearch || benchmarkMethodSearchText(row).includes(methodSearch));

    rows.sort((a, b) => {
        return compareBenchmarkRankingRows(a, b, benchmarkRankingSortState)
            || compareFiniteAsc(a.rank, b.rank)
            || benchmarkMethodLabel(a).localeCompare(benchmarkMethodLabel(b));
    });
    return rows;
}

function sortStateFromBenchmarkRankingSelect(value) {
    switch (value) {
        case "average_rank_asc":
            return { key: "average_rank", direction: "asc", type: "base" };
        case "wins_desc":
            return { key: "first_place_wins", direction: "desc", type: "base" };
        case "coverage_desc":
            return { key: "coverage", direction: "desc", type: "base" };
        case "method_asc":
            return { key: "method", direction: "asc", type: "base" };
        case "score_desc":
        default:
            return { key: "score", direction: "desc", type: "base" };
    }
}

function handleBenchmarkRankingHeaderClick(event) {
    const header = event.target.closest("[data-sort-key]");
    if (!header) return;
    const key = header.dataset.sortKey;
    const type = header.dataset.sortType || "base";
    const sameColumn = benchmarkRankingSortState.key === key && benchmarkRankingSortState.type === type;
    const defaultDirection = defaultBenchmarkRankingDirection(key, type);
    benchmarkRankingSortState = {
        key,
        type,
        direction: sameColumn
            ? (benchmarkRankingSortState.direction === "asc" ? "desc" : "asc")
            : defaultDirection,
    };
    renderBenchmarkRankingTable();
}

function defaultBenchmarkRankingDirection(key, type) {
    if (type === "task") return "desc";
    if (key === "method" || key === "rank" || key === "average_rank") return "asc";
    return "desc";
}

function compareBenchmarkRankingRows(a, b, state) {
    const key = state?.key || "score";
    const direction = state?.direction || "desc";
    if (state?.type === "task") {
        return compareFiniteDirection(
            a.task_scores?.[key]?.score,
            b.task_scores?.[key]?.score,
            direction,
        );
    }
    if (key === "method") {
        return compareBenchmarkStrings(benchmarkMethodLabel(a), benchmarkMethodLabel(b), direction);
    }
    return compareFiniteDirection(a[key], b[key], direction);
}

function renderBenchmarkSortableHeader(label, key, type = "base", className = "") {
    return `
        <th class="${escapeBenchmarkAttr(`${className} benchmark-sortable-th`.trim())}" data-sort-key="${escapeBenchmarkAttr(key)}" data-sort-type="${escapeBenchmarkAttr(type)}">
            <button type="button" class="benchmark-sort-header">
                <span>${escapeBenchmarkHtml(label)}</span>
                ${renderBenchmarkSortIndicator(benchmarkRankingSortState, key, type)}
            </button>
        </th>
    `;
}

function renderBenchmarkSortIndicator(state, key, type) {
    if (!state || state.key !== key || state.type !== type) {
        return '<i class="fas fa-sort benchmark-sort-icon benchmark-sort-icon-muted"></i>';
    }
    const icon = state.direction === "asc" ? "fa-sort-up" : "fa-sort-down";
    return `<i class="fas ${icon} benchmark-sort-icon"></i>`;
}

function renderBenchmarkRadar() {
    if (!benchmarkData || typeof Chart === "undefined") return;
    renderBenchmarkRadarChart("benchmarkTrackRadarChart", "tasks");
    renderBenchmarkRadarChart("benchmarkDatasetRadarChart", "datasets");
    const legacyCanvas = document.getElementById("benchmarkRadarChart");
    if (legacyCanvas) {
        const legacyMode = document.getElementById("benchmark-radar-mode")?.value || "tasks";
        renderBenchmarkRadarChart("benchmarkRadarChart", legacyMode);
    }
}

function renderBenchmarkRadarChart(canvasId, mode) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const limitValue = document.getElementById("benchmark-radar-limit")?.value || "8";
    const rankingRows = benchmarkOverviewRankingRows().filter(row => isFiniteNumber(row.score));
    const axes = benchmarkRadarAxes(mode, rankingRows);
    const labels = axes.map(axis => axis.label);
    const limit = limitValue === "all" ? rankingRows.length : Number(limitValue);
    const selectedRows = rankingRows.slice(0, Number.isFinite(limit) ? limit : rankingRows.length);
    setBenchmarkText("benchmark-radar-badge", `${rankingRows.length} methods`);
    // A radar needs three axes to show a shape; with fewer, point to the bar charts instead.
    const wrap = canvas.parentElement;
    let empty = wrap.querySelector(".benchmark-radar-empty");
    if (axes.length < 3) {
        if (benchmarkRadarCharts[canvasId]) {
            benchmarkRadarCharts[canvasId].destroy();
            delete benchmarkRadarCharts[canvasId];
        }
        canvas.hidden = true;
        if (!empty) {
            empty = document.createElement("div");
            empty.className = "benchmark-radar-empty";
            wrap.appendChild(empty);
        }
        empty.hidden = false;
        empty.textContent = `${axes.length} ${mode === "datasets" ? "dataset" : "track"}${axes.length === 1 ? "" : "s"} selected: `
            + "a radar needs at least three. The Top-10 scoreboards below show these scores.";
        return;
    }
    canvas.hidden = false;
    if (empty) empty.hidden = true;

    const datasets = selectedRows.map((row, index) => {
        const methodScores = axes.map(axis => axis.value(row));
        return {
            label: benchmarkChartMethodLabel(row),
            data: methodScores.map(value => isFiniteNumber(value) ? Number(value.toFixed(3)) : 0),
            backgroundColor: benchmarkColors[index % benchmarkColors.length],
            borderColor: benchmarkBorderColors[index % benchmarkBorderColors.length],
            borderWidth: 2,
            pointRadius: mode === "datasets" ? 2 : 3,
            fill: true,
        };
    });

    if (benchmarkRadarCharts[canvasId]) {
        benchmarkRadarCharts[canvasId].destroy();
    }

    benchmarkRadarCharts[canvasId] = new Chart(canvas.getContext("2d"), {
        type: "radar",
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: { boxWidth: 14 },
                },
                tooltip: {
                    callbacks: {
                        label: context => `${context.dataset.label}: ${context.formattedValue}`,
                    },
                },
            },
            scales: {
                r: {
                    min: 0,
                    max: 100,
                    ticks: {
                        stepSize: 20,
                        showLabelBackdrop: false,
                    },
                    pointLabels: {
                        font: {
                            size: mode === "datasets" ? 10 : 12,
                        },
                    },
                },
            },
        },
    });
    if (canvasId === "benchmarkRadarChart") benchmarkRadarChart = benchmarkRadarCharts[canvasId];
}

// Radar axes over the filtered rows: the selected tracks, or the selected (track, dataset) pairs. A dataset under
// several selected tracks (e.g. DTU in MVRecon and SfM) gets one axis per track.
function benchmarkRadarAxes(mode, rows) {
    if (mode === "datasets") {
        const selected = new Set(benchmarkSelectedTasks());
        const pairs = benchmarkFilterDatasetPairs()
            .filter(pair => selected.has(pair.task) && benchmarkDatasetSelected(pair.task, pair.dataset))
            .filter(pair => rows.some(row => isFiniteNumber(
                row[pair.excluded ? "excluded_dataset_scores" : "task_dataset_scores"]?.[pair.task]?.[pair.dataset]?.score)));
        const tracksPerDataset = {};
        for (const pair of pairs) tracksPerDataset[pair.dataset] = (tracksPerDataset[pair.dataset] || 0) + 1;
        return pairs.map(pair => {
            const name = tracksPerDataset[pair.dataset] > 1
                ? `${benchmarkDatasetLabel(pair.dataset)} (${benchmarkTaskLabel(pair.task)})`
                : benchmarkDatasetLabel(pair.dataset);
            const scores = pair.excluded ? "excluded_dataset_scores" : "task_dataset_scores";
            return {
                key: pair.key,
                label: pair.excluded ? `${name} \u2020` : name,  // display-only, see benchmarkFilterDatasetPairs
                value: row => row[scores]?.[pair.task]?.[pair.dataset]?.score,
            };
        });
    }
    return benchmarkSelectedTasks()
        .filter(task => rows.some(row => row.task_scores[task]))
        .map(task => ({ key: task, label: benchmarkTaskLabel(task), value: row => row.task_scores[task]?.score }));
}

function renderBenchmarkBarCharts() {
    if (!benchmarkData || typeof Chart === "undefined") return;
    renderBenchmarkOverallBarChart();
    renderBenchmarkTrackBarCharts();
}

function renderBenchmarkOverallBarChart() {
    const rows = benchmarkOverviewRankingRows()
        .filter(row => isFiniteNumber(row.score))
        .slice()
        .sort((a, b) => compareFiniteDesc(a.score, b.score) || compareFiniteAsc(a.rank, b.rank))
        .slice(0, 10);
    renderBenchmarkHorizontalBarChart(
        "benchmarkOverallBarChart",
        rows.map(row => benchmarkChartMethodLabel(row)),
        rows.map(row => Number(row.score.toFixed(3))),
        "Overall score",
    );
}

function renderBenchmarkTrackBarCharts() {
    const container = document.getElementById("benchmark-track-bar-grid");
    if (!container) return;
    const tasks = benchmarkSelectedTasks();
    const taskRows = benchmarkOverviewTaskRows();
    container.innerHTML = tasks.map(task => `
        <div class="benchmark-bar-panel">
            <div class="benchmark-radar-title">${escapeBenchmarkHtml(benchmarkTaskLabel(task))}</div>
            <div class="benchmark-bar-wrap benchmark-bar-wrap-small">
                <canvas id="benchmarkTrackBarChart-${escapeBenchmarkAttr(task)}"></canvas>
            </div>
        </div>
    `).join("");

    for (const task of tasks) {
        const rows = taskRows
            .filter(row => isFiniteNumber(row.task_scores?.[task]?.score))
            .slice()
            .sort((a, b) => {
                const scoreCompare = compareFiniteDesc(a.task_scores?.[task]?.score, b.task_scores?.[task]?.score);
                return scoreCompare || compareFiniteAsc(a.task_scores?.[task]?.rank, b.task_scores?.[task]?.rank);
            })
            .slice(0, 10);
        renderBenchmarkHorizontalBarChart(
            `benchmarkTrackBarChart-${task}`,
            rows.map(row => benchmarkChartMethodLabel(row)),
            rows.map(row => Number(row.task_scores[task].score.toFixed(3))),
            `${benchmarkTaskLabel(task)} score`,
        );
    }
}

// A method's paper date ("YYYY-MM" in benchmarkMethodInfo) as a month index (year * 12 + month - 1), or null.
function benchmarkMethodReleaseMonth(method) {
    const released = benchmarkMethodInfo[benchmarkBaseMethodName(method)]?.released;
    const match = /^(\d{4})-(\d{2})$/.exec(released || "");
    return match ? Number(match[1]) * 12 + Number(match[2]) - 1 : null;
}

// Month index -> "yy-mm".
function benchmarkMonthLabel(month) {
    const year = Math.floor(month / 12);
    return `${String(year % 100).padStart(2, "0")}-${String(month - year * 12 + 1).padStart(2, "0")}`;
}

// Model families of the score-vs-date trend lines, built from the method types; each gets its own color and dash.
const benchmarkReleaseFamilies = [
    { key: "3d", label: "3D/4D models", types: ["feedforward_3d", "streaming"],
      light: "#2a78d6", dark: "#3987e5", dash: [] },
    { key: "depth", label: "Depth models", types: ["mono_depth", "video_generative"],
      light: "#eb6834", dark: "#d95926", dash: [7, 4] },
    { key: "vlm", label: "VLMs", types: ["vlm", "api_vlm"],
      light: "#1baf7a", dark: "#199e70", dash: [2, 3] },
];

function benchmarkReleaseFamily(method) {
    const type = benchmarkMethodType(method);
    return benchmarkReleaseFamilies.find(family => family.types.includes(type)) || null;
}

// Least-squares line score = a + b * month over the points, or null without 3 points over 2 dates.
function benchmarkLinearFit(points) {
    if (points.length < 3 || new Set(points.map(point => point.x)).size < 2) return null;
    const meanX = points.reduce((sum, point) => sum + point.x, 0) / points.length;
    const meanY = points.reduce((sum, point) => sum + point.y, 0) / points.length;
    let sxx = 0, sxy = 0;
    for (const point of points) {
        sxx += (point.x - meanX) ** 2;
        sxy += (point.x - meanX) * (point.y - meanY);
    }
    const slope = sxy / sxx;
    return { slope, at: x => meanY + slope * (x - meanX) };
}

// One color per base model, the same in every scatter plot ([+C]/[+D] runs share their model's color):
// hues spread by the golden angle over the models sorted by name, lighter in the dark theme.
function benchmarkModelColor(method, alpha = 1) {
    const bases = [...new Set((benchmarkData?.methods || []).map(benchmarkBaseMethodName))].sort();
    const index = Math.max(0, bases.indexOf(benchmarkBaseMethodName(method)));
    const dark = document.documentElement.getAttribute("data-theme") === "dark";
    const hue = (index * 137.508) % 360;
    // Yellow-green hues read faint on a light background, so they are darker there.
    const faint = hue > 40 && hue < 170;
    const lightness = dark ? 66 - (index % 3) * 6 : (faint ? 30 : 38) + (index % 3) * 5;
    return `hsla(${hue.toFixed(1)}, 70%, ${lightness}%, ${alpha})`;
}

// Drag a box over a score-vs-date plot to zoom into that region (only the methods inside stay drawn and labelled);
// double-click resets. Native pointer listeners: Chart.js coalesces its own events per animation frame, which can
// drop the press or release of a quick drag. The zoom is stored in benchmarkReleaseZoom[options.zoomKey].
function benchmarkAttachReleaseZoom(chart) {
    const canvas = chart.canvas;
    const position = event => {
        const rect = canvas.getBoundingClientRect();
        return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    };
    const insideArea = ({ x, y }) => {
        const area = chart.chartArea;
        return x >= area.left && x <= area.right && y >= area.top && y <= area.bottom;
    };
    const clamp = ({ x, y }) => {
        const area = chart.chartArea;
        return { x: Math.min(Math.max(x, area.left), area.right), y: Math.min(Math.max(y, area.top), area.bottom) };
    };
    const move = event => {
        const state = chart.$releaseDrag;
        if (!state) return;
        const point = clamp(position(event));
        state.x1 = point.x;
        state.y1 = point.y;
        chart.draw();
    };
    const up = event => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
        const state = chart.$releaseDrag;
        chart.$releaseDrag = null;
        if (!state) return;
        const end = clamp(position(event));
        chart.draw();
        // A click (or a sliver) is not a zoom.
        if (Math.abs(end.x - state.x0) < 8 || Math.abs(end.y - state.y0) < 8) return;
        const xs = [state.x0, end.x].map(x => chart.scales.x.getValueForPixel(x));
        const ys = [state.y0, end.y].map(y => chart.scales.y.getValueForPixel(y));
        benchmarkReleaseZoom[chart.options.zoomKey] = {
            xMin: Math.min(...xs), xMax: Math.max(...xs), yMin: Math.min(...ys), yMax: Math.max(...ys),
        };
        benchmarkApplyReleaseZoom(chart);
    };
    canvas.addEventListener("pointerdown", event => {
        const point = position(event);
        if (event.button !== 0 || !insideArea(point)) return;
        event.preventDefault();
        chart.$releaseDrag = { x0: point.x, y0: point.y, x1: point.x, y1: point.y };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
    });
    canvas.addEventListener("pointermove", event => {
        canvas.style.cursor = chart.$releaseDrag || insideArea(position(event)) ? "crosshair" : "";
    });
    canvas.addEventListener("dblclick", () => {
        if (!benchmarkReleaseZoom[chart.options.zoomKey]) return;
        delete benchmarkReleaseZoom[chart.options.zoomKey];
        benchmarkApplyReleaseZoom(chart);
    });
    canvas.style.touchAction = "pan-y";  // a vertical swipe still scrolls the page on touch screens
}

// Chart.js plugin: draws the box being dragged by benchmarkAttachReleaseZoom.
const benchmarkReleaseZoomPlugin = {
    id: "benchmarkReleaseZoom",
    afterDraw(chart) {
        const state = chart.$releaseDrag;
        if (!state) return;
        const { ctx } = chart;
        ctx.save();
        ctx.fillStyle = "rgba(42, 120, 214, 0.10)";
        ctx.strokeStyle = "rgba(42, 120, 214, 0.8)";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 3]);
        const x = Math.min(state.x0, state.x1);
        const y = Math.min(state.y0, state.y1);
        ctx.fillRect(x, y, Math.abs(state.x1 - state.x0), Math.abs(state.y1 - state.y0));
        ctx.strokeRect(x, y, Math.abs(state.x1 - state.x0), Math.abs(state.y1 - state.y0));
        ctx.restore();
    },
};

// Set a score-vs-date chart's axes to its stored zoom (or its full range) and show / hide its reset button.
function benchmarkApplyReleaseZoom(chart) {
    const key = chart.options.zoomKey;
    const zoom = benchmarkReleaseZoom[key];
    const full = chart.options.fullRange;
    chart.options.scales.x.min = zoom ? zoom.xMin : full.xMin;
    chart.options.scales.x.max = zoom ? zoom.xMax : full.xMax;
    chart.options.scales.y.min = zoom ? zoom.yMin : 0;
    chart.options.scales.y.max = zoom ? zoom.yMax : 100;
    chart.update("none");
    const button = document.querySelector(`[data-release-zoom-reset="${CSS.escape(key)}"]`);
    if (button) button.hidden = !zoom;
}

// Chart.js plugin: writes each point's model name next to it, trying positions around the point until the
// label overlaps neither an earlier label nor a point; a label moved away from its point gets a leader line.
// VLMs are drawn on the evolving curves as their maker's logo (assets/logos/models/, a white badge so the marks stay
// legible in both themes): DepthVLM by Tencent Hunyuan, DepthLM by Meta, Qwen3-VL by Qwen, SenseNova-Vision by
// SenseTime (its GitHub organisation's avatar), and the three commercial models by Google, Anthropic and OpenAI. Keyed by base method, so every track of a model gets the logo.
const BENCHMARK_MODEL_LOGOS = {
    DEPTHVLM: "hunyuan", DEPTHLM: "meta", QWEN3_VL: "qwen", SENSENOVA_VISION: "sensenova",
    GEMINI_4_ARGON: "google", CLAUDE_OPUS_5_5: "anthropic", GPT_6_ASTRA: "openai",
};
const BENCHMARK_LOGO_SIZE = 22;
const BENCHMARK_TOOLTIP_MARKER_SIZE = 16;

// Results shown with a caveat until they are rerun, keyed by base method (every track of the model). Tables tag the
// name ("rerun pending", the note on hover), charts add "*", and the leaderboard, task tables and evolving curves
// print the notes underneath. Remove an entry once the method's rerun is collected.
// Entries look like `DEPTHLM: "<the caveat>"`. None at present: DepthLM and Qwen3-VL, first scored from one point query
// per image, were rerun with a 16x16 grid (MethodSpec.point_grid_size) on every MonoDepth and VideoDepth case
// (2026-11-11 to 2026-11-19).
const BENCHMARK_METHOD_FLAGS = {};

function benchmarkMethodFlag(methodOrRow) {
    const method = typeof methodOrRow === "string" ? methodOrRow : methodOrRow?.method;
    return BENCHMARK_METHOD_FLAGS[benchmarkBaseMethodName(method)] || null;
}

// A chart label: the method's name, with "*" when its results carry a caveat.
function benchmarkChartMethodLabel(row) {
    return `${benchmarkMethodLabel(row)}${benchmarkMethodFlag(row) ? " *" : ""}`;
}

// A table cell's method name, with a "rerun pending" tag whose title holds the caveat.
function benchmarkMethodNameHtml(row) {
    const flag = benchmarkMethodFlag(row);
    const name = escapeBenchmarkHtml(benchmarkMethodLabel(row));
    if (!flag) return name;
    return `${name} <span class="benchmark-method-flag" title="${escapeBenchmarkAttr(flag)}">`
        + `<i class="fas fa-triangle-exclamation" aria-hidden="true"></i> rerun pending</span>`;
}

// The caveats under the leaderboard, the task tables and the evolving curves.
function renderBenchmarkFlagNotes() {
    const entries = Object.entries(BENCHMARK_METHOD_FLAGS);
    // Methods sharing a caveat are named together: "DepthLM and Qwen3-VL: ...".
    const byNote = new Map();
    for (const [method, note] of entries) byNote.set(note, [...(byNote.get(note) || []), benchmarkMethodLabel({ method })]);
    const names = list => (list.length > 1 ? `${list.slice(0, -1).join(", ")} and ${list[list.length - 1]}` : list[0]);
    const text = [...byNote].map(([note, methods]) => `${names(methods)}: ${note}.`).join(" ");
    document.querySelectorAll("[data-flag-note]").forEach(element => {
        element.hidden = !entries.length;
        element.innerHTML = entries.length
            ? `<i class="fas fa-triangle-exclamation" aria-hidden="true"></i> * ${escapeBenchmarkHtml(text)}` : "";
    });
}  // CSS pixels; the SVG badges stay sharp at any device pixel ratio
const benchmarkLogoImages = {};

// `size` is the drawn size in CSS pixels (Chart.js draws an image point style at the image's own width and height):
// the plotted badge by default, a smaller one for the tooltip's marker.
function benchmarkModelLogo(method, size = BENCHMARK_LOGO_SIZE) {
    const logo = BENCHMARK_MODEL_LOGOS[benchmarkBaseMethodName(method)];
    if (!logo) return null;
    const name = size === BENCHMARK_LOGO_SIZE ? logo : `${logo}@${size}`;
    if (!benchmarkLogoImages[name]) {
        const image = new Image(size, size);
        image.alt = logo;
        // Charts drawn before the badge arrived redraw once it has.
        image.addEventListener("load", () => Object.values(benchmarkScatterCharts).forEach(chart => chart?.draw()));
        image.src = `assets/logos/models/${logo}.svg`;
        benchmarkLogoImages[name] = image;
    }
    return benchmarkLogoImages[name];
}

const benchmarkScatterLabelPlugin = {
    id: "benchmarkScatterLabels",
    afterDatasetsDraw(chart) {
        const { ctx, chartArea } = chart;
        const points = [];
        chart.data.datasets.forEach((dataset, datasetIndex) => {
            if (dataset.trend) return;
            const { chartArea: area } = chart;
            chart.getDatasetMeta(datasetIndex).data.forEach((element, index) => {
                // Points outside a zoomed-in region are not drawn, so get no label.
                if (element.x < area.left - 1 || element.x > area.right + 1
                    || element.y < area.top - 1 || element.y > area.bottom + 1) return;
                points.push({ x: element.x, y: element.y, raw: dataset.data[index] });
            });
        });
        ctx.save();
        ctx.font = "11px Inter, system-ui, sans-serif";
        ctx.textBaseline = "middle";
        const pad = 2;
        // A logo point is about twice a dot's size: keep its label clear of the badge.
        const radiusOf = point => (benchmarkModelLogo(point.raw.method) ? BENCHMARK_LOGO_SIZE / 2 + 1 : 5);
        const placed = points.map(point => {
            const radius = radiusOf(point);
            return { x0: point.x - radius, y0: point.y - radius, x1: point.x + radius, y1: point.y + radius };
        });
        const overlaps = box => placed.some(other => box.x0 < other.x1 && box.x1 > other.x0
            && box.y0 < other.y1 && box.y1 > other.y0);
        const inside = box => box.x0 >= chartArea.left - 40 && box.x1 <= chartArea.right + 80
            && box.y0 >= chartArea.top - 8 && box.y1 <= chartArea.bottom + 4;
        // Lower points first, so the dense top band (high scores) gets the remaining space last.
        const order = points.map((_, index) => index).sort((a, b) => points[a].x - points[b].x);
        for (const index of order) {
            const point = points[index];
            const text = point.raw.label;
            const width = ctx.measureText(text).width;
            const height = 12;
            let chosen = null;
            const offset = radiusOf(point) - 5;
            for (const distance of [8, 16, 26, 38, 52].map(value => value + offset)) {
                const candidates = [
                    [distance, 0], [-distance - width, 0], [-width / 2, -distance - 2], [-width / 2, distance + 2],
                    [distance, -distance], [distance, distance], [-distance - width, -distance], [-distance - width, distance],
                ];
                for (const [dx, dy] of candidates) {
                    const box = { x0: point.x + dx - pad, y0: point.y + dy - height / 2 - pad,
                                  x1: point.x + dx + width + pad, y1: point.y + dy + height / 2 + pad };
                    if (inside(box) && !overlaps(box)) {
                        chosen = { dx, dy, box, distance };
                        break;
                    }
                }
                if (chosen) break;
            }
            if (!chosen) {  // no free spot: right of the point, overlapping
                const dx = 8 + offset;
                chosen = { dx, dy: 0, distance: dx,
                           box: { x0: point.x + dx, y0: point.y - 6, x1: point.x + dx + width, y1: point.y + 6 } };
            }
            placed.push(chosen.box);
            const color = benchmarkModelColor(point.raw.method);
            if (chosen.distance > 8 + offset) {
                ctx.strokeStyle = benchmarkModelColor(point.raw.method, 0.45);
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(point.x, point.y);
                ctx.lineTo(Math.min(Math.max(point.x, chosen.box.x0), chosen.box.x1),
                           Math.min(Math.max(point.y, chosen.box.y0), chosen.box.y1));
                ctx.stroke();
            }
            ctx.fillStyle = color;
            ctx.fillText(text, point.x + chosen.dx, point.y + chosen.dy);
        }
        ctx.restore();
    },
};

// Width of the score-vs-date chart: "full" (the default on every page load) or "half" (a smaller figure when
// exported). The choice is not stored, so the home page always opens at full width.
function benchmarkReleaseWidth() {
    return "full";
}

function initBenchmarkReleaseWidth() {
    const select = document.getElementById("benchmark-release-width");
    if (!select) return;
    select.value = benchmarkReleaseWidth();
    select.addEventListener("change", () => renderBenchmarkReleaseScatterCharts());
}

// Evolving Curves for Geometric Foundation Models (score vs. release date): a slide of the home page, one task at a time (the tabs above the plot). It plots the
// leaderboard's task scores as published (the Overview's filters do not apply).
let benchmarkReleaseTask = null;

function benchmarkReleasePoints(task, rows) {
    const points = [];
    const undated = new Set();
    for (const row of rows) {
        const score = row.task_scores?.[task]?.score;
        if (!isFiniteNumber(score)) continue;
        const month = benchmarkMethodReleaseMonth(row.method);
        if (month === null) {
            undated.add(benchmarkMethodLabel({ method: benchmarkBaseMethodName(row.method) }));
            continue;
        }
        points.push({
            x: month,
            month,
            y: score,
            method: row.method,
            label: benchmarkChartMethodLabel(row),
            released: benchmarkMethodInfo[benchmarkBaseMethodName(row.method)].released,
            auxiliary: benchmarkMethodAuxiliaryKind(row) !== "rgb",
            family: benchmarkReleaseFamily(row.method),
        });
    }
    benchmarkDodgeReleasePoints(points);
    return { points, undated };
}

// Runs released the same month with nearly the same score would cover each other: shift them sideways within the
// month (x only; the paper date, the tooltip and the trend fits keep ``month``). Sizes are in axis units at the full
// chart size: about 28 px per month and 7 px per score point; a logo badge is about twice a dot.
function benchmarkDodgeReleasePoints(points) {
    const halfWidth = point => (benchmarkModelLogo(point.method) ? 0.42 : 0.24);   // months
    const halfHeight = point => (benchmarkModelLogo(point.method) ? 1.7 : 0.95);   // score points
    const byMonth = new Map();
    for (const point of points) {
        if (!byMonth.has(point.month)) byMonth.set(point.month, []);
        byMonth.get(point.month).push(point);
    }
    for (const group of byMonth.values()) {
        if (group.length < 2) continue;
        const placed = [];
        for (const point of [...group].sort((a, b) => b.y - a.y)) {
            const step = 2 * halfWidth(point);
            const clashes = offset => placed.some(other => Math.abs(other.y - point.y) < halfHeight(point) + halfHeight(other)
                && Math.abs(other.offset - offset) < halfWidth(point) + halfWidth(other));
            let offset = 0;
            for (let lane = 1; clashes(offset) && lane < 12; lane += 1) {
                offset = (lane % 2 ? 1 : -1) * Math.ceil(lane / 2) * step;
            }
            placed.push({ y: point.y, offset, method: point.method });
            point.x = point.month + offset;
        }
    }
}

// Image-only (rgb) runs take their task scores from ``ranking``, the pool of the overall leaderboard, so a point and
// the trend fits match the leaderboard's task columns; ``task_ranking`` also normalizes against the auxiliary
// [+C]/[+D] runs and shifts some rgb scores (MVRecon, SLAM). Auxiliary runs exist only in ``task_ranking``.
function benchmarkReleaseRows() {
    const rgbRows = benchmarkData.ranking?.rows || [];
    const taskRows = benchmarkData.task_ranking?.rows || [];
    if (!rgbRows.length) return taskRows;
    const rgbMethods = new Set(rgbRows.map(row => row.method));
    return [...rgbRows, ...taskRows.filter(row => !rgbMethods.has(row.method) && benchmarkMethodAuxiliaryKind(row) !== "rgb")];
}

function renderBenchmarkReleaseScatterCharts() {
    const container = document.getElementById("benchmark-release-scatter-grid");
    if (!container || !benchmarkData || typeof Chart === "undefined") return;
    const width = document.getElementById("benchmark-release-width")?.value || benchmarkReleaseWidth();
    container.classList.toggle("benchmark-release-grid-half", width === "half");
    const rows = benchmarkReleaseRows();
    const allTasks = benchmarkFilterTasks();
    if (!allTasks.includes(benchmarkReleaseTask)) benchmarkReleaseTask = allTasks[0] || null;
    const tabs = document.getElementById("benchmark-release-tabs");
    if (tabs) {
        tabs.innerHTML = allTasks.map(task => {
            const count = benchmarkReleasePoints(task, rows).points.length;
            return `<button type="button" role="tab" data-release-task="${escapeBenchmarkAttr(task)}"
                aria-selected="${task === benchmarkReleaseTask}">${escapeBenchmarkHtml(benchmarkTaskLabel(task))}
                <small>${count}</small></button>`;
        }).join("");
        for (const button of tabs.querySelectorAll("[data-release-task]")) {
            button.addEventListener("click", () => {
                benchmarkReleaseTask = button.dataset.releaseTask;
                renderBenchmarkReleaseScatterCharts();
            });
        }
    }
    const tasks = benchmarkReleaseTask ? [benchmarkReleaseTask] : [];
    for (const chart of Object.values(benchmarkScatterCharts)) chart.destroy();
    for (const key of Object.keys(benchmarkScatterCharts)) delete benchmarkScatterCharts[key];
    container.innerHTML = tasks.map(task => `
        <div class="benchmark-bar-panel">
            <div class="benchmark-radar-title benchmark-release-title">
                <span>${escapeBenchmarkHtml(benchmarkTaskLabel(task))}</span>
                <button type="button" class="benchmark-btn benchmark-release-reset"
                    data-release-zoom-reset="${escapeBenchmarkAttr(task)}" ${benchmarkReleaseZoom[task] ? "" : "hidden"}>
                    Reset zoom
                </button>
            </div>
            <div class="benchmark-release-wrap">
                <canvas id="benchmarkReleaseScatter-${escapeBenchmarkAttr(task)}"></canvas>
            </div>
        </div>
    `).join("");
    for (const button of container.querySelectorAll("[data-release-zoom-reset]")) {
        button.addEventListener("click", () => {
            const chart = benchmarkScatterCharts[button.dataset.releaseZoomReset];
            delete benchmarkReleaseZoom[button.dataset.releaseZoomReset];
            if (chart) benchmarkApplyReleaseZoom(chart);
        });
    }

    const undated = new Set();
    for (const task of tasks) {
        const { points, undated: missing } = benchmarkReleasePoints(task, rows);
        missing.forEach(label => undated.add(label));
        const canvas = document.getElementById(`benchmarkReleaseScatter-${task}`);
        if (!canvas || !points.length) continue;
        const months = points.map(point => point.x);
        const dark = document.documentElement.getAttribute("data-theme") === "dark";
        // One trend line per model family, fitted on its image-only runs ([+C]/[+D] runs get extra inputs).
        const trends = benchmarkReleaseFamilies.map(family => {
            // Fitted on the paper dates, not the dodged positions.
            const members = points.filter(point => point.family === family && !point.auxiliary)
                .map(point => ({ x: point.month, y: point.y }));
            const fit = benchmarkLinearFit(members);
            if (!fit) return null;
            const first = Math.min(...members.map(point => point.x));
            const last = Math.max(...members.map(point => point.x));
            const perYear = fit.slope * 12;
            const color = dark ? family.dark : family.light;
            return {
                type: "line",
                trend: true,
                label: `${family.label}: ${perYear >= 0 ? "+" : "\u2212"}${formatBenchmarkNumber(Math.abs(perYear), 1)}/yr (n=${members.length})`,
                perYear,
                data: [{ x: first, y: fit.at(first) }, { x: last, y: fit.at(last) }],
                borderColor: color,
                backgroundColor: color,
                borderWidth: 2,
                borderDash: family.dash,
                pointRadius: 0,
                pointHoverRadius: 0,
                pointHitRadius: 0,
                fill: false,
                order: 1,
            };
        }).filter(Boolean);
        benchmarkScatterCharts[task] = new Chart(canvas.getContext("2d"), {
            type: "scatter",
            data: {
                datasets: [...trends, {
                    order: 0,
                    clip: 8,  // points just past a zoomed edge stay whole; the rest are cut
                    label: `${benchmarkTaskLabel(task)} score`,
                    data: points,
                    // Image-only runs are filled circles in the model's color, ringed in their family's trend color;
                    // runs given extra GT input are hollow triangles in the model's color (not in the trends).
                    backgroundColor: context => (context.raw?.auxiliary
                        ? "transparent" : benchmarkModelColor(context.raw?.method, 0.85)),
                    borderColor: context => {
                        const family = context.raw?.family;
                        if (context.raw?.auxiliary || !family) return benchmarkModelColor(context.raw?.method);
                        return dark ? family.dark : family.light;
                    },
                    borderWidth: context => (context.raw?.auxiliary ? 1.8 : context.raw?.family ? 2.5 : 1),
                    pointStyle: context => benchmarkModelLogo(context.raw?.method)
                        || (context.raw?.auxiliary ? "triangle" : "circle"),
                    pointRadius: context => (benchmarkModelLogo(context.raw?.method) ? BENCHMARK_LOGO_SIZE / 2 : 6),
                    pointHoverRadius: context => (benchmarkModelLogo(context.raw?.method) ? BENCHMARK_LOGO_SIZE / 2 + 2 : 8),
                }],
            },
            plugins: [benchmarkScatterLabelPlugin, benchmarkReleaseZoomPlugin],
            options: {
                animation: false,  // labels are placed from the final point positions
                zoomKey: task,
                fullRange: { xMin: Math.min(...months) - 1, xMax: Math.max(...months) + 1 },
                responsive: true,
                maintainAspectRatio: false,
                layout: { padding: { right: 90, top: 10 } },
                plugins: {
                    // The legend lists the trend lines only; the points are labelled in the plot.
                    legend: {
                        display: trends.length > 0,
                        position: "top",
                        align: "start",
                        labels: {
                            color: benchmarkChartTextColor(),
                            boxWidth: 28,
                            boxHeight: 0,
                            filter: (item, data) => Boolean(data.datasets[item.datasetIndex]?.trend),
                            generateLabels: chart => chart.data.datasets.map((dataset, index) => ({
                                text: dataset.label,
                                datasetIndex: index,
                                strokeStyle: dataset.borderColor,
                                fillStyle: dataset.borderColor,
                                lineWidth: 2,
                                lineDash: dataset.borderDash || [],
                                hidden: !chart.isDatasetVisible(index),
                                fontColor: benchmarkChartTextColor(),
                            })),
                        },
                    },
                    tooltip: {
                        filter: item => !item.dataset.trend,
                        // The marker beside each entry is the point's own: the model's logo, or its circle / triangle
                        // (not Chart.js's default colour square).
                        usePointStyle: true,
                        boxWidth: BENCHMARK_TOOLTIP_MARKER_SIZE,
                        boxHeight: BENCHMARK_TOOLTIP_MARKER_SIZE,
                        callbacks: {
                            labelPointStyle: context => ({
                                pointStyle: benchmarkModelLogo(context.raw?.method, BENCHMARK_TOOLTIP_MARKER_SIZE)
                                    || (context.raw?.auxiliary ? "triangle" : "circle"),
                                rotation: 0,
                            }),
                            label: context => {
                                const point = context.raw;
                                const family = point.family ? ` \u00b7 ${point.family.label}` : "";
                                return `${point.label} (${point.released}${family}): ${formatBenchmarkNumber(point.y, 2)}`;
                            },
                            afterLabel: context => {
                                const flag = benchmarkMethodFlag(context.raw?.method);
                                return flag ? `* ${flag}` : "";
                            },
                        },
                    },
                },
                scales: {
                    x: {
                        type: "linear",
                        min: benchmarkReleaseZoom[task]?.xMin ?? Math.min(...months) - 1,
                        max: benchmarkReleaseZoom[task]?.xMax ?? Math.max(...months) + 1,
                        title: { display: true, text: "Paper date (yy-mm, arXiv v1 first)", color: benchmarkChartTextColor() },
                        grid: { color: "rgba(148, 163, 184, 0.18)" },
                        // A tick every quarter (Jan / Apr / Jul / Oct): month indices that are multiples of 3; every
                        // month when zoomed into a year or less.
                        afterBuildTicks: axis => {
                            const ticks = [];
                            const step = axis.max - axis.min <= 12 ? 1 : 3;
                            for (let month = Math.ceil(axis.min / step) * step; month <= axis.max; month += step) {
                                ticks.push({ value: month });
                            }
                            axis.ticks = ticks;
                        },
                        ticks: {
                            color: benchmarkChartTextColor(),
                            autoSkip: false,
                            callback: value => benchmarkMonthLabel(value),
                        },
                    },
                    y: {
                        min: benchmarkReleaseZoom[task]?.yMin ?? 0,  // trend lines past the scores are clipped
                        max: benchmarkReleaseZoom[task]?.yMax ?? 100,
                        title: { display: true, text: `${benchmarkTaskLabel(task)} score`, color: benchmarkChartTextColor() },
                        grid: { color: "rgba(148, 163, 184, 0.18)" },
                        ticks: { color: benchmarkChartTextColor() },
                    },
                },
            },
        });
    }
    for (const chart of Object.values(benchmarkScatterCharts)) benchmarkAttachReleaseZoom(chart);
    const note = document.getElementById("benchmark-release-undated");
    if (note) {
        note.textContent = undated.size
            ? `No release date recorded (not plotted): ${[...undated].sort().join(", ")}.`
            : "";
    }
}

function renderBenchmarkHorizontalBarChart(canvasId, labels, values, valueLabel) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    if (benchmarkBarCharts[canvasId]) {
        benchmarkBarCharts[canvasId].destroy();
    }
    benchmarkBarCharts[canvasId] = new Chart(canvas.getContext("2d"), {
        type: "bar",
        data: {
            labels,
            datasets: [{
                label: valueLabel,
                data: values,
                backgroundColor: labels.map((_, index) => benchmarkBarColor(index, 0.72)),
                borderColor: labels.map((_, index) => benchmarkBorderColors[index % benchmarkBorderColors.length]),
                borderWidth: 1,
                borderRadius: 6,
            }],
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: context => `${valueLabel}: ${formatBenchmarkNumber(context.parsed.x, 2)}`,
                    },
                },
            },
            scales: {
                x: {
                    beginAtZero: true,
                    suggestedMax: 100,
                    grid: { color: "rgba(148, 163, 184, 0.18)" },
                    ticks: { color: benchmarkChartTextColor() },
                },
                y: {
                    grid: { display: false },
                    ticks: {
                        color: benchmarkChartTextColor(),
                        autoSkip: false,
                        font: { size: 11 },
                    },
                },
            },
        },
    });
}

function benchmarkBarColor(index, alpha = 0.7) {
    const color = benchmarkBorderColors[index % benchmarkBorderColors.length];
    const match = color.match(/\d+/g);
    if (!match || match.length < 3) return color;
    return `rgba(${match[0]}, ${match[1]}, ${match[2]}, ${alpha})`;
}

function benchmarkChartTextColor() {
    return getComputedStyle(document.documentElement).getPropertyValue("--text-secondary").trim() || "#64748b";
}

function renderBenchmarkTaskTables() {
    const container = document.getElementById("benchmark-task-tables");
    if (!container || !benchmarkData) return;

    benchmarkVisibleTaskSections = collectBenchmarkTaskSections();
    container.innerHTML = benchmarkVisibleTaskSections.length
        ? benchmarkVisibleTaskSections.map(section => renderBenchmarkTaskSection(section)).join("")
        : '<div class="benchmark-empty">No benchmark rows match the current filters.</div>';
}

function collectBenchmarkTaskSections() {
    const task = activeBenchmarkTaskMetricTab();
    const datasetTab = activeBenchmarkDatasetTab();
    const selectedFamily = document.getElementById("benchmark-family-filter")?.value || "all";
    const methodSearch = (document.getElementById("benchmark-method-search")?.value || "").trim().toLowerCase();
    const taskTable = benchmarkData?.task_tables?.[task];
    if (!taskTable) return [];

    if (datasetTab === "all") {
        // Overall: SLAM consolidates each range track over its datasets; the other tasks over all their datasets.
        if (taskTable.range_tables) return collectBenchmarkSlamRangeSections(task, taskTable, selectedFamily, methodSearch);
        const section = buildBenchmarkTableSection(task, "all", taskTable, selectedFamily, methodSearch);
        return section ? [section] : [];
    }
    const tableKeys = benchmarkDatasetTableKeys(taskTable, datasetTab);
    return tableKeys.map(tableKey => {
        const slamRange = tableKey.split(BENCHMARK_SLAM_RANGE_SEPARATOR)[1];
        return buildBenchmarkTableSection(task, tableKey, taskTable.dataset_tables[tableKey], selectedFamily,
            methodSearch, slamRange ? `${slamRange.charAt(0).toUpperCase()}${slamRange.slice(1)} range` : "");
    }).filter(Boolean);
}

function buildBenchmarkTableSection(task, tableKey, sourceTable, selectedFamily, methodSearch, title = "") {
    const metricKeys = (sourceTable.metrics || []).filter(metricKey => {
        const metric = benchmarkMetricInfo(metricKey);
        return selectedFamily === "all" || metric?.family === selectedFamily;
    });
    if (!metricKeys.length) return null;

    const sectionKey = benchmarkTaskSectionKey(task, tableKey);
    const sourceRows = (sourceTable.methods || [])
        // Overall leaves out rows whose results are all on datasets outside the leaderboard (e.g. VideoDepth [+C],
        // SpatialArena only): they are not scored there; their dataset tabs show them.
        .filter(row => !(tableKey === "all" && row.unscored_only))
        .filter(row => !methodSearch || benchmarkMethodSearchText(row).includes(methodSearch))
        .map(row => {
            if (row.missing) return { ...row, task_score: undefined, task_rank: undefined };
            if ("ranked" in row) return { ...row, task_score: row.score, task_rank: row.score_rank };
            if (tableKey !== "all") {
                // A dataset tab scores and ranks each method on that dataset alone.
                const datasetScore = benchmarkDatasetScoreForMethod(row.method, task, tableKey);
                return { ...row, task_score: datasetScore, task_rank: undefined, ranked: isFiniteNumber(datasetScore) };
            }
            const taskScore = benchmarkTaskScoreForMethod(row.method, task);
            return {
                ...row,
                task_score: taskScore?.score,
                task_rank: taskScore?.rank,
            };
        });
    const ownScore = sourceRows.some(row => "ranked" in row);
    const methodRows = sortBenchmarkTaskRows(sourceRows, task, sectionKey, ownScore);

    return {
        task,
        sectionKey,
        selectedDataset: tableKey,
        datasetLabel: tableKey === "all" ? "All datasets" : tableKey,
        title,
        metricKeys,
        methodRows,
        ownScore,
        scoreLabel: task === "SLAM" ? "Cell Score" : ownScore ? "Dataset Score" : "Task Score",
        note: task === "SLAM" && ownScore
            ? "Cell Score = mean over the sequences of the 0-100 normalized ATE RMSE and drift (the protocol's "
                + "ranking metrics); a failed (e.g. OOM) or missing sequence scores 0."
            : ownScore
                ? "Dataset Score = the task score on this dataset alone: the mean over its cases of the 0-100 "
                    + "normalized metrics; a failed (e.g. OOM) or missing case scores 0. Ranks are within this dataset."
                : "",
    };
}

function collectBenchmarkSlamRangeSections(task, taskTable, selectedFamily, methodSearch) {
    const sections = [];
    for (const slamRange of taskTable.ranges || Object.keys(taskTable.range_tables)) {
        const rangeTable = taskTable.range_tables[slamRange];
        if (!rangeTable) continue;
        const metricKeys = (rangeTable.metrics || []).filter(metricKey => (
            selectedFamily === "all" || benchmarkMetricInfo(metricKey).family === selectedFamily
        ));
        if (!metricKeys.length) continue;
        const sectionKey = benchmarkTaskSectionKey(task, `range:${slamRange}`);
        const sourceRows = (rangeTable.methods || [])
            .filter(row => !methodSearch || benchmarkMethodSearchText(row).includes(methodSearch))
            .map(row => ({ ...row, task_score: row.score, task_rank: row.score_rank }));
        const datasets = (rangeTable.datasets || []).map(dataset => benchmarkDatasetLabel(dataset)).join(", ");
        sections.push({
            task,
            sectionKey,
            selectedDataset: "all",
            datasetLabel: `${slamRange} range`,
            title: `${slamRange.charAt(0).toUpperCase()}${slamRange.slice(1)} range`,
            note: `${datasets}. Range Score = mean over every sequence of the range of the 0-100 normalized ATE `
                + "RMSE and drift; a failed (e.g. OOM) or missing sequence scores 0.",
            metricKeys,
            columnGroups: true,
            methodRows: sortBenchmarkTaskRows(sourceRows, task, sectionKey, true),
            ownScore: true,
            scoreLabel: "Range Score",
        });
    }
    return sections;
}

// A row's failure for a metric column it has no value in: the whole cell's (a dataset table) or, in a SLAM range
// table, that column's dataset's -- e.g. "7 GPU OOM".
function benchmarkFailureLabel(row, metricKey) {
    const dataset = benchmarkSlamColumnDataset(metricKey);
    if (dataset) return row.dataset_failures?.[dataset] || "";
    return row.failure_label || "";
}

// A row's method limitations (failures of the method's own code, operators or outputs, not of the run): a badge on
// its method cell whose tooltip names each one and why.
function benchmarkLimitationBadge(row) {
    const limitations = row.limitations || [];
    if (!limitations.length) return "";
    const title = ["Method limitation (not a run failure):"].concat(limitations.map(entry =>
        `${entry.count} ${entry.label}: ${(entry.notes || []).join(" ")}`)).join("\n").replace(/`/g, "");
    return `<span class="benchmark-limitation" title="${escapeBenchmarkAttr(title)}">Method limitation</span>`;
}

// Whether every failure a cell names ("15 Adapter argument error, 3 No camera export") is a method limitation.
function benchmarkFailureIsLimitation(row, failure) {
    const labels = new Set((row.limitations || []).map(entry => entry.label));
    return labels.size > 0 && failure.split(", ").every(part => labels.has(part.replace(/^\d+ /, "")));
}

function benchmarkSectionRankRows(section) {
    // Methods marked unranked (incomplete SLAM coverage) are shown but never ranked.
    const rows = benchmarkResultRows(section.methodRows);
    return section.ownScore ? rows.filter(row => row.ranked) : rows;
}

// Rows whose metric means cover only some of their cases (``incomplete``) are left out of the per-metric ranks
// that rank the raw metric means (payloads without ``metric_case_scores``).
function benchmarkSectionMetricRankRows(section) {
    return benchmarkSectionRankRows(section).filter(row => !row.incomplete);
}

// Each metric column is ranked by the metric's case score (``metric_case_scores``: its normalized 0-100 score
// averaged over every case of the table, a case the method did not evaluate counting 0, as in the task score), so
// a method that skipped cases is ranked, not dropped. Payloads without case scores rank the raw means instead.
function benchmarkSectionMetricRanks(section) {
    const allRows = benchmarkSectionRankRows(section);
    const byCaseScore = allRows.some(row => row.metric_case_scores);
    const rankRows = byCaseScore ? allRows : benchmarkSectionMetricRankRows(section);
    return Object.fromEntries(section.metricKeys.map(metricKey => {
        const metric = benchmarkMetricInfo(metricKey);
        return [
            metricKey,
            byCaseScore
                ? rankBenchmarkVisibleRows(
                    rankRows,
                    row => (isFiniteNumber(row.metrics?.[metricKey]?.value) ? row.metric_case_scores?.[metricKey] : undefined),
                    "higher",
                )
                : rankBenchmarkVisibleRows(
                    rankRows,
                    row => row.metrics?.[metricKey]?.value,
                    metric.direction === "higher" ? "higher" : "lower",
                ),
        ];
    }));
}

function renderBenchmarkSlamColumnGroups(section) {
    const groups = [];
    for (const metricKey of section.metricKeys) {
        const dataset = benchmarkSlamColumnDataset(metricKey);
        const last = groups[groups.length - 1];
        if (last && last.dataset === dataset) last.span += 1;
        else groups.push({ dataset, span: 1 });
    }
    return `
        <tr class="benchmark-column-groups">
            <th></th><th></th>
            ${groups.map(group => `
                <th class="benchmark-column-group" colspan="${group.span}">${escapeBenchmarkHtml(benchmarkDatasetLabel(group.dataset))}</th>
            `).join("")}
        </tr>
    `;
}

function renderBenchmarkTaskSection(section) {
    const { task, sectionKey, metricKeys, methodRows } = section;
    const taskScoreRanks = rankBenchmarkVisibleRows(
        benchmarkSectionRankRows(section),
        row => row.task_score,
        "higher",
    );
    const metricRanks = benchmarkSectionMetricRanks(section);
    const headerCells = metricKeys.map(metricKey => {
        const metric = benchmarkMetricInfo(metricKey);
        const direction = metric.direction === "higher" ? "higher" : "lower";
        const directionIcon = metric.direction === "higher" ? "fa-arrow-up" : "fa-arrow-down";
        return `
            <th class="numeric benchmark-sortable-th" data-section-key="${escapeBenchmarkAttr(sectionKey)}" data-sort-key="${escapeBenchmarkAttr(metricKey)}" data-sort-type="metric">
                <button type="button" class="benchmark-sort-header">
                    <span>${escapeBenchmarkHtml(metric.label || metricKey)}</span>
                    ${renderBenchmarkSortIndicator(benchmarkTaskSortState[sectionKey], metricKey, "metric")}
                </button>
                <span class="benchmark-direction"><i class="fas ${directionIcon}"></i> ${direction}</span>
            </th>
        `;
    }).join("");

    const rows = methodRows.map(row => {
        const metricCells = metricKeys.map(metricKey => {
            const failure = row.metrics?.[metricKey] ? "" : benchmarkFailureLabel(row, metricKey);
            const limitation = failure && benchmarkFailureIsLimitation(row, failure);
            return failure
                ? `<td class="numeric benchmark-failed-cell${limitation ? " benchmark-limitation-cell" : ""}" title="${escapeBenchmarkAttr(limitation ? `Method limitation: ${failure}` : failure)}">${escapeBenchmarkHtml(failure)}</td>`
                : renderBenchmarkMetricCell(row.metrics?.[metricKey], metricRanks[metricKey]?.get(row.method) ?? null, metricKey,
                    row.metric_case_scores?.[metricKey]);
        }).join("");
        const unranked = section.ownScore && !row.ranked;
        const coverageTitle = row.unscored_only ? "Results only on datasets left out of the leaderboard score; see their dataset tables"
            : unranked ? "Not ranked: incomplete coverage"
            : section.ownScore ? "Sequences not evaluated score 0"
                : "Cases not evaluated score 0; the metric means cover the evaluated ones and are not ranked";
        const coverage = row.coverage_label
            ? `<span class="benchmark-unranked" title="${coverageTitle}">${unranked ? "unranked · " : ""}${escapeBenchmarkHtml(row.coverage_label)}</span>`
            : "";
        const scoreCell = row.missing
            ? `<td class="numeric benchmark-muted" title="No result for this method">-</td>`
            : section.ownScore
                ? renderBenchmarkOwnScoreCell(row, taskScoreRanks.get(row.method))
                : renderBenchmarkTaskScoreCell(row, task, taskScoreRanks.get(row.method));
        const rowClass = row.missing ? "benchmark-row-missing" : unranked ? "benchmark-row-unranked" : "";
        return `
            <tr${rowClass ? ` class="${rowClass}"` : ""}${row.missing ? ' title="Required for this track; no result yet"' : ""}>
                <td class="benchmark-method-name">${benchmarkMethodNameHtml(row)}${coverage}${benchmarkLimitationBadge(row)}</td>
                ${scoreCell}
                ${metricCells}
            </tr>
        `;
    }).join("");

    const heading = section.title || section.note
        ? `
            <div class="benchmark-task-section-heading">
                ${section.title ? `<h4>${escapeBenchmarkHtml(section.title)}</h4>` : ""}
                ${section.note ? `<p>${escapeBenchmarkHtml(section.note)}</p>` : ""}
            </div>
        `
        : "";
    return `
        <div class="benchmark-task-section">
            ${heading}
            <div class="table-container">
                <table class="benchmark-table">
                    <thead>
                        ${section.columnGroups ? renderBenchmarkSlamColumnGroups(section) : ""}
                        <tr>
                            <th class="benchmark-sortable-th" data-section-key="${escapeBenchmarkAttr(sectionKey)}" data-sort-key="method" data-sort-type="base">
                                <button type="button" class="benchmark-sort-header">
                                    <span>Method</span>
                                    ${renderBenchmarkSortIndicator(benchmarkTaskSortState[sectionKey], "method", "base")}
                                </button>
                            </th>
                            <th class="numeric benchmark-sortable-th" data-section-key="${escapeBenchmarkAttr(sectionKey)}" data-sort-key="task_score" data-sort-type="base">
                                <button type="button" class="benchmark-sort-header">
                                    <span>${escapeBenchmarkHtml(section.scoreLabel || "Task Score")}</span>
                                    ${renderBenchmarkSortIndicator(benchmarkTaskSortState[sectionKey], "task_score", "base")}
                                </button>
                            </th>
                            ${headerCells}
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        </div>
    `;
}

function rankBenchmarkVisibleRows(rows, valueFn, direction) {
    const entries = (rows || [])
        .map(row => ({ method: row.method, value: valueFn(row) }))
        .filter(entry => isFiniteNumber(entry.value));
    entries.sort((a, b) => (
        compareFiniteDirection(a.value, b.value, direction === "higher" ? "desc" : "asc")
        || benchmarkMethodLabel(a.method).localeCompare(benchmarkMethodLabel(b.method))
    ));
    const ranks = new Map();
    let previousValue = null;
    let previousRank = 0;
    entries.forEach((entry, index) => {
        const rank = previousValue !== null && benchmarkValuesTie(entry.value, previousValue)
            ? previousRank
            : index + 1;
        ranks.set(entry.method, rank);
        previousValue = entry.value;
        previousRank = rank;
    });
    return ranks;
}

function benchmarkValuesTie(a, b) {
    const aValue = Number(a);
    const bValue = Number(b);
    if (!Number.isFinite(aValue) || !Number.isFinite(bValue)) return false;
    const tolerance = 1e-12 * Math.max(1, Math.abs(aValue), Math.abs(bValue));
    return Math.abs(aValue - bValue) <= tolerance;
}

function benchmarkTaskSectionKey(task, dataset) {
    return `${task || ""}::${dataset || "all"}`;
}

function handleBenchmarkTaskMetricHeaderClick(event) {
    const header = event.target.closest("[data-sort-key]");
    if (!header) return;
    const sectionKey = header.dataset.sectionKey || "";
    const key = header.dataset.sortKey || "";
    const type = header.dataset.sortType || "metric";
    if (!sectionKey || !key) return;
    const current = benchmarkTaskSortState[sectionKey];
    const sameColumn = current?.key === key && current?.type === type;
    benchmarkTaskSortState[sectionKey] = {
        key,
        type,
        direction: sameColumn
            ? (current.direction === "asc" ? "desc" : "asc")
            : defaultBenchmarkTaskSortDirection(key, type),
    };
    renderBenchmarkTaskTables();
}

function defaultBenchmarkTaskSortDirection(key, type) {
    if (type === "base") return key === "method" ? "asc" : "desc";
    const metric = benchmarkMetricInfo(key);
    return metric?.direction === "higher" ? "desc" : "asc";
}

function sortBenchmarkTaskRows(rows, task, sectionKey, ownScore = false) {
    // Required methods without a result ("missing" rows, all "-") always follow the ones with results.
    const sorted = sortBenchmarkTaskRowsWithResults(rows.filter(row => !row.missing), task, sectionKey, ownScore);
    const missing = rows.filter(row => row.missing)
        .sort((a, b) => benchmarkMethodLabel(a).localeCompare(benchmarkMethodLabel(b)));
    return sorted.concat(missing);
}

function sortBenchmarkTaskRowsWithResults(rows, task, sectionKey, ownScore = false) {
    const state = benchmarkTaskSortState[sectionKey];
    const sorted = rows.slice();
    if (!state && ownScore) {
        // Ranked methods first by their own score, then the unranked ones by name.
        return sorted.sort((a, b) => (
            Number(Boolean(b.ranked)) - Number(Boolean(a.ranked))
            || compareFiniteDesc(a.task_score, b.task_score)
            || benchmarkMethodLabel(a).localeCompare(benchmarkMethodLabel(b))
        ));
    }
    if (!state) {
        return sorted.sort((a, b) => compareBenchmarkMethodsForTask(a.method, b.method, task));
    }
    sorted.sort((a, b) => {
        let compare = 0;
        if (state.type === "metric") {
            compare = compareFiniteDirection(
                a.metrics?.[state.key]?.value,
                b.metrics?.[state.key]?.value,
                state.direction,
            );
        } else if (state.key === "method") {
            compare = compareBenchmarkStrings(benchmarkMethodLabel(a), benchmarkMethodLabel(b), state.direction);
        } else {
            compare = compareFiniteDirection(a[state.key], b[state.key], state.direction);
        }
        return compare
            || compareBenchmarkMethodsForTask(a.method, b.method, task)
            || benchmarkMethodLabel(a).localeCompare(benchmarkMethodLabel(b));
    });
    return sorted;
}

function benchmarkMethodAuxiliaryKind(rowOrMethod) {
    const method = typeof rowOrMethod === "string" ? rowOrMethod : String(rowOrMethod?.method || "");
    const track = typeof rowOrMethod === "object" && rowOrMethod?.track ? String(rowOrMethod.track) : "";
    const key = track || method;
    // Masked-depth tracks (spatialarena/depth_prior_masking.py) share the depth / depth+cameras filter groups.
    if (key.endsWith("_rgb_masked_depth_cameras") || key === "rgb_masked_depth_cameras") return "depth_cameras";
    if (key.endsWith("_rgb_sparse_depth_cameras") || key === "rgb_sparse_depth_cameras") return "depth_cameras";
    if (key.endsWith("_rgb_sparse_depth") || key === "rgb_sparse_depth") return "depth";
    if (key.endsWith("_rgb_masked_depth") || key === "rgb_masked_depth") return "depth";
    if (key.endsWith("_rgb_depth_cameras") || key.endsWith("_rgb_cameras_depth") || key === "rgb_depth_cameras" || key === "rgb_cameras_depth") {
        return "depth_cameras";
    }
    if (key.endsWith("_rgb_cameras") || key === "rgb_cameras") return "cameras";
    if (key.endsWith("_rgb_depth") || key === "rgb_depth") return "depth";
    return "rgb";
}

// ---- Overview filters: tracks, datasets, model types ------------------------------------------------------------
// null = everything. Datasets are "<task>|<dataset>" keys: a dataset name can appear under several tracks.
const benchmarkOverviewFilter = { tasks: null, datasets: null, types: null };

function benchmarkMethodType(method) {
    const types = benchmarkData?.method_types || {};
    return types[method] || types[benchmarkBaseMethodName(method)] || "other";
}

function benchmarkMethodTypeLabel(type) {
    return benchmarkData?.method_type_labels?.[type] || type;
}

function benchmarkFilterTasks() {
    return benchmarkData?.task_ranking?.tasks || benchmarkData?.ranking?.tasks || benchmarkData?.tasks || [];
}

// The datasets of the filter, per track: the leaderboard datasets, then those kept out of the leaderboard
// (SpatialArena, Spring, ScanNet++; excluded: true). Selecting the latter shows them in the dataset radar only: their
// scores (excluded_dataset_scores) never enter a track or overall score.
function benchmarkFilterDatasetPairs() {
    const tasks = benchmarkFilterTasks();
    const pairs = new Map();
    const rows = [...(benchmarkData?.task_ranking?.rows || []), ...(benchmarkData?.ranking?.rows || [])];
    for (const [field, excluded] of [["task_dataset_scores", false], ["excluded_dataset_scores", true]]) {
        for (const row of rows) {
            for (const [task, byDataset] of Object.entries(row[field] || {})) {
                for (const dataset of Object.keys(byDataset)) {
                    const key = `${task}|${dataset}`;
                    if (!pairs.has(key)) pairs.set(key, { task, dataset, key, excluded });
                }
            }
        }
    }
    return [...pairs.values()].sort((a, b) => (tasks.indexOf(a.task) - tasks.indexOf(b.task))
        || (a.excluded - b.excluded)
        || benchmarkDatasetLabel(a.dataset).localeCompare(benchmarkDatasetLabel(b.dataset)));
}

function benchmarkFilterDatasetLabel(pair) {
    return pair.excluded ? `${benchmarkDatasetLabel(pair.dataset)} \u2020` : benchmarkDatasetLabel(pair.dataset);
}

function benchmarkFilterTypes() {
    const present = new Set([...(benchmarkData?.task_ranking?.rows || []), ...(benchmarkData?.ranking?.rows || [])]
        .map(row => benchmarkMethodType(row.method)));
    const order = Object.keys(benchmarkData?.method_type_labels || {});
    return [...present].sort((a, b) => order.indexOf(a) - order.indexOf(b));
}

function benchmarkSelectedTasks() {
    const tasks = benchmarkFilterTasks();
    return benchmarkOverviewFilter.tasks ? tasks.filter(task => benchmarkOverviewFilter.tasks.has(task)) : tasks;
}

function benchmarkDatasetSelected(task, dataset) {
    return !benchmarkOverviewFilter.datasets || benchmarkOverviewFilter.datasets.has(`${task}|${dataset}`);
}

function benchmarkTypeSelected(method) {
    return !benchmarkOverviewFilter.types || benchmarkOverviewFilter.types.has(benchmarkMethodType(method));
}

function benchmarkDatasetLabel(dataset) {
    return benchmarkData?.dataset_display_names?.[dataset] || dataset;
}

// A task score over the selected datasets: the per-dataset means weighted by their counts -- the task's cases (each
// scene, or SLAM sequence, weighs the same; a missing or failed case scores 0). Rows from exports without
// per-dataset scores keep their published score.
function benchmarkFilteredTaskScore(row, task) {
    const byDataset = row.task_dataset_scores?.[task];
    if (!byDataset) return benchmarkOverviewFilter.datasets ? null : (row.task_scores?.[task]?.score ?? null);
    const entries = Object.entries(byDataset)
        .filter(([dataset, entry]) => benchmarkDatasetSelected(task, dataset) && isFiniteNumber(entry.score) && entry.coverage > 0);
    if (!entries.length) return null;
    let sum = 0;
    let count = 0;
    for (const [, entry] of entries) {
        sum += entry.score * entry.coverage;
        count += entry.coverage;
    }
    return sum / count;
}

function benchmarkAssignRanks(rows, valueOf, assign) {
    const ranked = rows.filter(row => isFiniteNumber(valueOf(row))).sort((a, b) => valueOf(b) - valueOf(a));
    ranked.forEach((row, index) => assign(row, index + 1));
}

// Ranking rows with task and overall scores recomputed for the selected tracks and datasets, then limited to the
// selected model types (the overall score's track set is taken from all models, so hiding some changes no score).
function benchmarkFilteredRankingRows(rows) {
    const tasks = benchmarkSelectedTasks();
    const scored = rows.map(row => {
        const taskScores = {};
        for (const task of tasks) {
            const score = benchmarkFilteredTaskScore(row, task);
            if (!isFiniteNumber(score)) continue;
            const byDataset = row.task_dataset_scores?.[task];
            const coverage = byDataset
                ? Object.entries(byDataset).reduce((sum, [dataset, entry]) => sum
                    + (benchmarkDatasetSelected(task, dataset) ? entry.coverage || 0 : 0), 0)
                : row.task_scores?.[task]?.coverage;
            taskScores[task] = { ...(row.task_scores?.[task] || {}), score, coverage, rank: null };
        }
        const coverage = Object.values(taskScores).reduce((sum, entry) => sum + (entry.coverage || 0), 0);
        return { ...row, task_scores: taskScores, coverage };
    });
    const scoredTasks = tasks.filter(task => scored.some(row => row.task_scores[task]));
    const visible = scored
        .map(row => {
            // A conditioned track ([+C], [+D]) averages over the selected tasks it can run (ranking_tasks), not
            // over tasks the task/track matrix never gives it; an rgb row over every selected task.
            const rowTasks = row.ranking_tasks ? scoredTasks.filter(task => row.ranking_tasks.includes(task)) : scoredTasks;
            const score = rowTasks.length
                ? rowTasks.reduce((sum, task) => sum + (row.task_scores[task]?.score ?? 0), 0) / rowTasks.length
                : null;
            return { ...row, score, score_display: isFiniteNumber(score) ? score.toFixed(2) : "-", rank: null };
        })
        .filter(row => benchmarkTypeSelected(row.method));
    for (const task of scoredTasks) {
        benchmarkAssignRanks(visible, row => row.task_scores[task]?.score, (row, rank) => { row.task_scores[task].rank = rank; });
    }
    // Conditioned tracks ([+C], [+D]) get no overall rank and follow the rgb rows: their score covers only the tasks
    // they run, so it is not comparable with an rgb method's (their task ranks above stay).
    const conditioned = row => (row.track || "rgb") !== "rgb";
    benchmarkAssignRanks(visible.filter(row => !conditioned(row)), row => row.score, (row, rank) => { row.rank = rank; });
    return visible.sort((a, b) => (Number(conditioned(a)) - Number(conditioned(b))) || compareFiniteDesc(a.score, b.score));
}

// The overall leaderboard rows (no auxiliary-input tracks) and the per-task rows (input filter applied), filtered.
function benchmarkOverviewRankingRows() {
    return benchmarkFilteredRankingRows(benchmarkData?.ranking?.rows || benchmarkData?.task_ranking?.rows || []);
}

function benchmarkOverviewTaskRows() {
    return benchmarkFilteredRankingRows(benchmarkData?.task_ranking?.rows || benchmarkData?.ranking?.rows || []);
}

function benchmarkFilterIsActive() {
    return Boolean(benchmarkOverviewFilter.tasks || benchmarkOverviewFilter.datasets || benchmarkOverviewFilter.types);
}

function renderBenchmarkOverviewFilters() {
    const root = document.getElementById("benchmark-overview-filters");
    if (!root || !benchmarkData) return;
    const tasks = benchmarkFilterTasks();
    const selectedTasks = new Set(benchmarkSelectedTasks());
    const pairs = benchmarkFilterDatasetPairs().filter(pair => selectedTasks.has(pair.task));
    const types = benchmarkFilterTypes();
    const typeCounts = {};
    for (const row of benchmarkData.task_ranking?.rows || []) {
        const type = benchmarkMethodType(row.method);
        typeCounts[type] = (typeCounts[type] || 0) + 1;
    }
    const option = (filter, value, label, checked, note = "") => `
        <label class="benchmark-filter-option">
            <input type="checkbox" data-filter="${filter}" value="${escapeBenchmarkAttr(value)}" ${checked ? "checked" : ""}>
            ${escapeBenchmarkHtml(label)}${note ? ` <small>${escapeBenchmarkHtml(note)}</small>` : ""}
        </label>`;
    const actions = filter => `<div class="benchmark-filter-actions">
        <button type="button" data-filter-all="${filter}">All</button><button type="button" data-filter-none="${filter}">None</button></div>`;
    const panels = {
        tasks: actions("tasks") + tasks.map(task => option("tasks", task, benchmarkTaskLabel(task), selectedTasks.has(task))).join(""),
        datasets: actions("datasets") + tasks.filter(task => selectedTasks.has(task)).map(task => {
            const taskPairs = pairs.filter(pair => pair.task === task);
            if (!taskPairs.length) return "";
            return `<div class="benchmark-filter-group">${escapeBenchmarkHtml(benchmarkTaskLabel(task))}</div>`
                + taskPairs.map(pair => option("datasets", pair.key, benchmarkFilterDatasetLabel(pair),
                    benchmarkDatasetSelected(pair.task, pair.dataset))).join("");
        }).join("") + (pairs.some(pair => pair.excluded)
            ? '<div class="benchmark-filter-footnote">\u2020 Not in the leaderboard: shown in the dataset radar, never in a score.</div>'
            : ""),
        types: actions("types") + types.map(type => option("types", type, benchmarkMethodTypeLabel(type),
            !benchmarkOverviewFilter.types || benchmarkOverviewFilter.types.has(type), `${typeCounts[type] || 0}`)).join(""),
    };
    const counts = {
        tasks: benchmarkOverviewFilter.tasks ? `${selectedTasks.size}/${tasks.length}` : "all",
        datasets: benchmarkOverviewFilter.datasets
            ? `${pairs.filter(pair => benchmarkDatasetSelected(pair.task, pair.dataset)).length}/${pairs.length}` : "all",
        types: benchmarkOverviewFilter.types ? `${benchmarkOverviewFilter.types.size}/${types.length}` : "all",
    };
    for (const menu of root.querySelectorAll(".benchmark-filter-menu")) {
        const filter = menu.dataset.filter;
        menu.querySelector(".benchmark-filter-panel").innerHTML = panels[filter];
        menu.querySelector(".benchmark-filter-count").textContent = `(${counts[filter]})`;
    }
    const visibleModels = benchmarkOverviewTaskRows().filter(row => Object.keys(row.task_scores).length).length;
    setBenchmarkText("benchmark-filter-summary", benchmarkFilterIsActive()
        ? `${visibleModels} models shown` : "Showing everything");
}

function benchmarkFilterAllValues(filter) {
    if (filter === "tasks") return benchmarkFilterTasks();
    if (filter === "types") return benchmarkFilterTypes();
    return benchmarkFilterDatasetPairs().map(pair => pair.key);
}

function setBenchmarkOverviewFilter(filter, values) {
    const all = benchmarkFilterAllValues(filter);
    const set = new Set(values);
    benchmarkOverviewFilter[filter] = all.every(value => set.has(value)) ? null : set;
    renderBenchmarkFilteredOverview();
}

function initBenchmarkOverviewFilters() {
    const root = document.getElementById("benchmark-overview-filters");
    if (!root) return;
    root.addEventListener("change", event => {
        const input = event.target.closest("input[data-filter]");
        if (!input) return;
        const filter = input.dataset.filter;
        const checked = [...root.querySelectorAll(`input[data-filter="${filter}"]:checked`)].map(box => box.value);
        if (filter === "datasets") {
            // Keep the selection of datasets whose tracks are hidden right now.
            const shown = new Set([...root.querySelectorAll('input[data-filter="datasets"]')].map(box => box.value));
            const hidden = benchmarkFilterAllValues("datasets").filter(key => !shown.has(key)
                && (!benchmarkOverviewFilter.datasets || benchmarkOverviewFilter.datasets.has(key)));
            setBenchmarkOverviewFilter(filter, [...checked, ...hidden]);
        } else {
            setBenchmarkOverviewFilter(filter, checked);
        }
    });
    root.addEventListener("click", event => {
        const all = event.target.closest("[data-filter-all]");
        const none = event.target.closest("[data-filter-none]");
        if (all) setBenchmarkOverviewFilter(all.dataset.filterAll, benchmarkFilterAllValues(all.dataset.filterAll));
        if (none) setBenchmarkOverviewFilter(none.dataset.filterNone, []);
    });
    document.getElementById("benchmark-filter-reset")?.addEventListener("click", () => {
        benchmarkOverviewFilter.tasks = benchmarkOverviewFilter.datasets = benchmarkOverviewFilter.types = null;
        renderBenchmarkFilteredOverview();
    });
    // Close an open menu when clicking elsewhere.
    document.addEventListener("click", event => {
        for (const menu of root.querySelectorAll(".benchmark-filter-menu[open]")) {
            if (!menu.contains(event.target)) menu.removeAttribute("open");
        }
    });
}

function renderBenchmarkFilteredOverview() {
    if (!benchmarkData) return;
    renderBenchmarkOverviewFilters();
    renderBenchmarkSummary();
    renderBenchmarkRadar();
    renderBenchmarkOverallBarChart();
    renderBenchmarkTrackBarCharts();
}

function exportBenchmarkRankingCsv() {
    if (!benchmarkData) return;
    const tasks = benchmarkData.ranking?.tasks || [];
    const headers = ["rank", "method", "method_id", "score", "average_rank", "coverage", "wins"]
        .concat(tasks.map(task => `${task}_score`))
        .concat(tasks.map(task => `${task}_rank`));
    const visibleRows = benchmarkVisibleRankingRows.length ? benchmarkVisibleRankingRows : getBenchmarkRankingRows();
    const rows = visibleRows.map((row, index) => {
        const base = [index + 1, benchmarkMethodLabel(row), row.method, row.score, row.average_rank, row.coverage, row.first_place_wins];
        const taskScores = tasks.map(task => row.task_scores?.[task]?.score ?? "");
        const taskRanks = tasks.map(task => row.task_scores?.[task]?.rank ?? "");
        return base.concat(taskScores, taskRanks);
    });
    downloadBenchmarkCsv("benchmark_leaderboard.csv", [headers].concat(rows));
}

function exportBenchmarkTaskCsv() {
    if (!benchmarkData) return;
    const sections = benchmarkVisibleTaskSections.length ? benchmarkVisibleTaskSections : collectBenchmarkTaskSections();
    const rows = [["task", "dataset_scope", "method", "method_id", "datasets_count", "task_score", "task_rank", "metric", "value", "display", "rank"]];
    for (const section of sections) {
        const taskScoreRanks = rankBenchmarkVisibleRows(
            benchmarkSectionRankRows(section),
            row => row.task_score,
            "higher",
        );
        const metricRanks = benchmarkSectionMetricRanks(section);
        for (const methodRow of section.methodRows) {
            const taskScore = section.ownScore
                ? { score: methodRow.task_score }
                : benchmarkTaskScoreForMethod(methodRow.method, section.task);
            for (const metricKey of section.metricKeys) {
                const cell = methodRow.metrics?.[metricKey];
                rows.push([
                    section.task,
                    section.datasetLabel,
                    benchmarkMethodLabel(methodRow),
                    methodRow.method,
                    methodRow.datasets_count ?? methodRow.datasets?.length ?? 0,
                    taskScore?.score ?? "",
                    taskScoreRanks.get(methodRow.method) ?? "",
                    metricKey,
                    cell?.value ?? "",
                    cell?.display ?? "",
                    metricRanks[metricKey]?.get(methodRow.method) ?? "",
                ]);
            }
        }
    }
    downloadBenchmarkCsv("benchmark_task_metrics.csv", rows);
}

function downloadBenchmarkCsv(filename, rows) {
    const csv = rows
        .map(row => row.map(value => csvEscapeBenchmark(value)).join(","))
        .join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
}

// ``caseScore`` (the metric's case score, which ranks the column) is added to the hover text.
function renderBenchmarkMetricCell(cell, rankOverride = undefined, metricKey = "", caseScore = undefined) {
    const caseLine = isFiniteNumber(caseScore)
        ? `\nCase score ${caseScore.toFixed(1)} / 100 (ranks this column): normalized over every case, a missing case counting 0`
        : "";
    const title = metricKey ? ` title="${escapeBenchmarkAttr(benchmarkMetricCellTitle(metricKey) + caseLine)}"` : "";
    if (!cell || cell.display === "-") {
        return `<td class="numeric benchmark-muted"${title}>-</td>`;
    }
    const rank = rankOverride === undefined ? cell.rank : rankOverride;
    // With a metric title on the cell, the badge carries none, so hovering it shows the metric too.
    const rankBadge = rank ? renderBenchmarkRankBadge(rank, metricKey ? null : undefined) : "";
    return `
        <td class="numeric"${title}>
            <span class="benchmark-metric-cell">
                <span>${escapeBenchmarkHtml(cell.display)}</span>
                ${rankBadge}
            </span>
        </td>
    `;
}

function renderBenchmarkTaskScoreCell(row, task, rankOverride = undefined) {
    const taskScore = benchmarkTaskScoreForMethod(row.method, task);
    if (!taskScore || !isFiniteNumber(taskScore.score)) {
        return '<td class="numeric benchmark-muted">-</td>';
    }
    const rank = rankOverride === undefined ? taskScore.rank : rankOverride;
    // SLAM: the task score is the mean of the per-range scores (a range not fully covered counts as 0).
    const rangeTitle = taskScore.ranges
        ? Object.entries(taskScore.ranges)
            .map(([slamRange, entry]) => `${slamRange}: ${formatBenchmarkNumber(entry.score, 2)} (#${entry.rank ?? "-"})`)
            .join(" · ")
        : "";
    return `
        <td class="numeric"${rangeTitle ? ` title="${escapeBenchmarkAttr(rangeTitle)}"` : ""}>
            <span class="benchmark-metric-cell">
                <span>${formatBenchmarkNumber(taskScore.score, 2)}</span>
                ${renderBenchmarkRankBadge(rank)}
            </span>
        </td>
    `;
}

function renderBenchmarkOwnScoreCell(row, rank) {
    if (!isFiniteNumber(row.task_score)) {
        return '<td class="numeric benchmark-muted">-</td>';
    }
    return `
        <td class="numeric">
            <span class="benchmark-metric-cell">
                <span>${formatBenchmarkNumber(row.task_score, 2)}</span>
                ${renderBenchmarkRankBadge(rank)}
            </span>
        </td>
    `;
}

// `title` overrides the badge's hover text; null leaves it without one (the enclosing cell's title shows instead).
function renderBenchmarkRankBadge(rank, title = undefined) {
    const numericRank = Number(rank);
    if (!Number.isFinite(numericRank) || numericRank <= 0) {
        return '<span class="benchmark-rank benchmark-rank-empty">-</span>';
    }
    const integerRank = Math.trunc(numericRank);
    const topClass = integerRank >= 1 && integerRank <= 3 ? ` benchmark-rank-top-${integerRank}` : "";
    const medalLabels = {
        1: "🏅",
        2: "🥈",
        3: "🥉",
    };
    const label = medalLabels[integerRank] || `#${integerRank}`;
    const rankTitle = title !== undefined ? title
        : integerRank === 1 ? "Gold rank" : integerRank === 2 ? "Silver rank" : integerRank === 3 ? "Bronze rank" : `Rank ${integerRank}`;
    const titleAttribute = rankTitle === null ? "" : ` title="${escapeBenchmarkAttr(rankTitle)}"`;
    return `<span class="benchmark-rank${topClass}"${titleAttribute}>${label}</span>`;
}

// A method's score on one dataset of a task (the task score computed on that dataset alone), including the datasets
// kept out of the leaderboard (excluded_dataset_scores); null without a result there.
function benchmarkDatasetScoreForMethod(method, task, dataset) {
    const row = (benchmarkData?.task_ranking?.rows || benchmarkData?.ranking?.rows || [])
        .find(item => item.method === method);
    const entry = row?.task_dataset_scores?.[task]?.[dataset] ?? row?.excluded_dataset_scores?.[task]?.[dataset];
    return isFiniteNumber(entry?.score) ? entry.score : null;
}

function benchmarkTaskScoreForMethod(method, task) {
    const row = (benchmarkData?.task_ranking?.rows || benchmarkData?.ranking?.rows || [])
        .find(item => item.method === method);
    return row?.task_scores?.[task] || null;
}

function compareBenchmarkMethodsForTask(methodA, methodB, task) {
    const taskRows = benchmarkData.task_ranking?.rows || benchmarkData.ranking?.rows || [];
    const rowA = taskRows.find(row => row.method === methodA);
    const rowB = taskRows.find(row => row.method === methodB);
    // A row without a rank on this task (e.g. VideoDepth [+C], scored only on other tasks) follows the ranked ones;
    // its overall rank says nothing about this task. Without a task, the overall rank orders them.
    const taskRankA = (task ? rowA?.task_scores?.[task]?.rank : rowA?.rank) ?? 1e9;
    const taskRankB = (task ? rowB?.task_scores?.[task]?.rank : rowB?.rank) ?? 1e9;
    if (taskRankA !== taskRankB) return taskRankA - taskRankB;
    return benchmarkMethodLabel(rowA || methodA).localeCompare(benchmarkMethodLabel(rowB || methodB));
}

function benchmarkMethodLabel(methodOrRow) {
    if (methodOrRow && typeof methodOrRow === "object") {
        return methodOrRow.method_display || benchmarkMethodLabel(methodOrRow.method);
    }
    const method = String(methodOrRow || "");
    if (!method || method === "-") return method;
    const generatedLabel = benchmarkData?.method_display_names?.[method];
    if (generatedLabel) return generatedLabel;
    return fallbackBenchmarkMethodLabel(method);
}

function fallbackBenchmarkMethodLabel(method) {
    const baseNames = {
        COME: "COME",
        CUT3R: "CUT3R",
        DA3: "DA3",
        "DA3-G": "DA3-G",
        "DA3-L": "DA3-L",
        "DA3-NGL": "DA3-NGL",
        DEJAVIEW: "Déjà View",
        DEPTHLM: "DepthLM",
        DEPTHVLM: "DepthVLM",
        DUST3R: "DUSt3R",
        FAST3R: "Fast3R",
        FASTVGGT: "FastVGGT",
        IGGT: "IGGT",
        IGGT4D: "IGGT4D",
        MAPANYTHING: "MapAnything",
        MAST3R: "MASt3R",
        OMNIVGGT: "OmniVGGT",
        PI3: "Pi3",
        PI3_X: "Pi3X",
        POW3R: "Pow3R",
        QWEN3_VL: "Qwen3-VL",
        ROBUSTVGGT: "RobustVGGT",
        SENSENOVA_VISION: "SenseNova-Vision",
        VGGT: "VGGT",
        VGGT_OMEGA: "VGGT-Ω",
        VLM3: "VLM3",
        MOGE: "MoGe",
        MOGE2: "MoGe-2",
        MOGE3: "MoGe-3",
        MARIGOLD: "Marigold",
        MARIGOLD_V2: "Marigold V2",
        MARIGOLD_V1_1: "Marigold v1.1",
        LOTUS_D: "Lotus-D",
        DEPTH_PRO: "Depth Pro",
        UNIDEPTH_V2: "UniDepthV2",
        INFINIDEPTH: "InfiniDepth",
        DVD: "DVD",
        DEPTHCRAFTER: "DepthCrafter",
        CHRONODEPTH: "ChronoDepth",
        GEONEXT_OFFICIAL_SVD: "GeoNeXt",
        ABOT_RECON: "ABot-Recon",
        LINGBOT_MAP: "LingBot-Map",
        LOGER: "LoGeR",
        PI_LONG: "Pi-Long",
        SCAL3R: "Scal3R",
        VGGT_LONG: "VGGT-Long",
        VGGT_SLAM: "VGGT-SLAM",
        ZIPMAP: "ZipMap",
        COME: "Co-Me",
    };
    const tracks = [
        ["_rgb_sparse_depth_cameras", " [+sDC]"],
        ["_rgb_sparse_depth", " [+sD]"],
        ["_rgb_masked_depth_cameras", " [+mDC]"],
        ["_rgb_masked_depth", " [+mD]"],
        ["_rgb_depth_cameras", " [+DC]"],
        ["_rgb_cameras_depth", " [+DC]"],
        ["_rgb_cameras", " [+C]"],
        ["_rgb_depth", " [+D]"],
        ["_rgb", ""],
    ];
    for (const [suffix, labelSuffix] of tracks) {
        if (method.endsWith(suffix)) {
            const base = method.slice(0, -suffix.length);
            return `${baseNames[base] || base}${labelSuffix}`;
        }
    }
    return baseNames[method] || method;
}

function benchmarkTrackLabel(track) {
    const labels = {
        rgb: "RGB",
        rgb_depth: "RGB + depth",
        rgb_cameras: "RGB + cameras",
        rgb_masked_depth: "RGB + masked depth",
        rgb_masked_depth_cameras: "RGB + masked depth + cameras",
        rgb_sparse_depth: "RGB + sparse depth",
        rgb_sparse_depth_cameras: "RGB + sparse depth + cameras",
    };
    return labels[String(track || "")] || String(track || "-").replace(/_/g, " ");
}

function benchmarkMethodSearchText(row) {
    return `${row?.method || ""} ${benchmarkMethodLabel(row)}`.toLowerCase();
}

function benchmarkTaskLabel(task) {
    const key = String(task || "");
    const generatedLabel = benchmarkData?.task_display_names?.[key];
    if (generatedLabel) return generatedLabel;
    const fallback = {
        MonoDepth: "MonoDepth",
        VideoDepth: "Video Depth",
        MVRecon: "MVRecon",
        SfM: "SfM",
        SLAM: "SLAM",
    };
    return fallback[key] || key;
}

function benchmarkDatasetLabel(dataset) {
    const key = String(dataset || "");
    const generatedLabel = benchmarkData?.dataset_display_names?.[key];
    if (generatedLabel) return generatedLabel;
    const fallback = {
        "7scenes": "7Scenes",
        bonn: "Bonn",
        diode: "DIODE",
        dtu: "DTU",
        eth3d: "ETH3D",
        imc: "IMC",
        kitti: "KITTI",
        marigold: "Marigold",
        nyuv2: "NYUv2",
        sintel: "Sintel",
        tanks_and_temples: "Tanks and Temples",
        spatialarena: "SpatialArena",
        dycheck: "DyCheck",
    };
    return fallback[key] || key;
}

function benchmarkDatasetScopeLabel(datasetLabel) {
    return datasetLabel === "All datasets" ? datasetLabel : benchmarkDatasetLabel(datasetLabel);
}

function setBenchmarkText(id, value) {
    const element = document.getElementById(id);
    if (element) element.textContent = value;
}

function formatBenchmarkNumber(value, precision = 2) {
    return isFiniteNumber(value) ? Number(value).toFixed(precision) : "-";
}

function isFiniteNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
}

function compareFiniteAsc(a, b) {
    const aValue = isFiniteNumber(a) ? a : Infinity;
    const bValue = isFiniteNumber(b) ? b : Infinity;
    return aValue === bValue ? 0 : aValue - bValue;
}

function compareFiniteDesc(a, b) {
    const aValue = isFiniteNumber(a) ? a : -Infinity;
    const bValue = isFiniteNumber(b) ? b : -Infinity;
    return bValue === aValue ? 0 : bValue - aValue;
}

function compareFiniteDirection(a, b, direction) {
    const aFinite = isFiniteNumber(a);
    const bFinite = isFiniteNumber(b);
    if (!aFinite && !bFinite) return 0;
    if (!aFinite) return 1;
    if (!bFinite) return -1;
    return direction === "asc" ? a - b : b - a;
}

function compareBenchmarkStrings(a, b, direction) {
    const compare = String(a || "").localeCompare(String(b || ""));
    return direction === "desc" ? -compare : compare;
}

function formatBenchmarkGeneratedAt(value) {
    if (!value) return "not generated";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "generated";
    return `updated ${date.toLocaleDateString()} ${date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
}

function csvEscapeBenchmark(value) {
    const stringValue = String(value ?? "");
    if (/[",\n\r]/.test(stringValue)) {
        return `"${stringValue.replace(/"/g, '""')}"`;
    }
    return stringValue;
}

function escapeBenchmarkHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function escapeBenchmarkAttr(value) {
    return escapeBenchmarkHtml(value);
}
