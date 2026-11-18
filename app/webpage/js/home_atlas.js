// Copyright (c) 2026 Yu Chen. All rights reserved.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

// Benchmark statistics on the SpatialArena home tab, ported from the atlas page (docs/webpage/index.html): the
// hierarchical composition, the scene-coverage matrix and the model families. Data and drawing follow the atlas; colours come from
// the .home-atlas scope (css/home.css) and fonts are set per chart, so the dashboard's Chart.js defaults stay as they are.

(function () {
    "use strict";

    // SpatialArena benchmark atlas. Numbers follow Table 1 and Sec. 4-5 of the paper.
    const TASKS = [
      { key: 'MonoDepth', label: 'Monocular depth', color: '--primary' },
      { key: 'VideoDepth', label: 'Video depth', color: '--sky' },
      { key: 'MVRecon', label: 'Multi-view reconstruction', color: '--secondary' },
      { key: 'SfM', label: 'Structure from motion', color: '--amber' },
      { key: 'SLAM', label: 'SLAM', color: '--accent' },
    ];
    // [task, name, thumb, scenes, images, evaluated, attrs]
    // attrs: I indoor, O outdoor, A aerial, Obj object-centric, D dynamic, Syn synthetic, Web internet photos, Long recordings of 10k+ frames
    // Spatial Arena (ours) is last: images = unique images over its 30 scenes; VideoDepth scenes = clips, evaluated = frames with GT;
    // MVRecon and SfM evaluate up to 300 images per scene; SLAM evaluates 918 short (stride 3) + 2,149 long + 10,919 large frames.
    // Its attrs are per task (the statues and the monument are in MonoDepth, MVRecon and SfM only; crowded scenes are not in VideoDepth;
    // the SLAM scenes CoalDropsYard2 and CambridgeWesthub3 are 12,435- and 11,124-image recordings over the 3 cameras).
    const DATASETS = [
      ['MonoDepth', 'Marigold suite', 'marigold', 4, 2531, 2531, 'I O'],
      ['MonoDepth', 'HAMMER', 'hammer', 1, 775, 775, 'I Obj'],
      ['MonoDepth', 'iBims-1', 'ibims1', 1, 100, 100, 'I'],
      ['MonoDepth', 'Spring', 'spring', 1, 1000, 1000, 'O D Syn'],
      ['MonoDepth', 'Synth4K', 'synth4k', 5, 1960, 1960, 'O D Syn'],
      ['MonoDepth', 'AerialMetric', 'aerialmetric', 3, 3596, 3596, 'O A D'],
      ['VideoDepth', 'Sintel', 'sintel', 23, 1064, 1064, 'I O D Syn'],
      ['VideoDepth', 'Bonn', 'bonn', 26, 2860, 2600, 'I D'],
      ['VideoDepth', 'KITTI', 'kitti', 13, 1269, 1189, 'O D'],
      ['VideoDepth', 'DyCheck', 'dycheck', 12, 3952, 1200, 'I O D'],
      ['MVRecon', '7-Scenes', '7scenes', 7, 43000, 700, 'I'],
      ['MVRecon', 'ETH3D', 'eth3d', 11, 908, 856, 'I O'],
      ['MVRecon', 'DTU', 'dtu', 15, 855, 855, 'Obj'],
      ['MVRecon', 'Tanks & Temples', 'tanks_and_temples', 6, 2784, 600, 'I O Obj'],
      ['MVRecon', 'ScanNet++', 'scannetpp', 20, 178345, 2000, 'I'],
      ['MVRecon', 'HiRoom', 'hiroom', 30, 400, 400, 'I Syn'],
      ['MVRecon', 'NRGBD', 'nrgbd', 9, 10970, 900, 'I Syn'],
      ['SfM', 'ETH3D', 'eth3d', 13, 908, 856, 'I O'],
      ['SfM', 'Tanks & Temples', 'tanks_and_temples', 6, 2784, 600, 'I O Obj'],
      ['SfM', 'IMC Phototourism', 'imc', 8, 3145, 800, 'O Web'],
      ['SfM', 'DTU', 'dtu', 15, 735, 735, 'Obj'],
      ['SfM', 'ScanNet++', 'scannetpp', 20, 178345, 2000, 'I'],
      ['SfM', 'HiRoom', 'hiroom', 30, 400, 400, 'I Syn'],
      ['SfM', 'TexturePoorSfM', 'texturepoorsfm', 30, 350, 350, 'Obj'],
      ['SLAM', '7-Scenes', '7scenes', 18, 17000, 5678, 'I'],
      ['SLAM', 'TUM RGB-D (dynamic)', 'tum_dynamics', 9, 10806, 3606, 'I D'],
      ['SLAM', 'Oxford Spires', 'oxford_spires', 10, 38380, 38380, 'O Long'],
      ['SLAM', 'VBR', 'vbr', 7, 86678, 86678, 'O D Long'],
      ['SLAM', 'KITTI Odometry', 'kitti_odometry', 11, 23201, 23201, 'O D Long'],
      ['MonoDepth', 'Spatial Arena (ours)', 'spatialarena', 19, 32655, 1688, 'I O Obj D'],
      ['VideoDepth', 'Spatial Arena (ours)', 'spatialarena', 15, 32655, 4484, 'I O'],
      ['MVRecon', 'Spatial Arena (ours)', 'spatialarena', 6, 32655, 1800, 'I O Obj D'],
      ['SfM', 'Spatial Arena (ours)', 'spatialarena', 7, 32655, 2100, 'I O Obj D'],
      ['SLAM', 'Spatial Arena (ours)', 'spatialarena', 8, 32655, 13986, 'I O D Long'],
    ];
    const OURS = 'spatialarena';
    const OURS_SCENES = 30;
    const ATTRS = [['I', 'Indoor'], ['O', 'Outdoor'], ['A', 'Aerial'], ['Obj', 'Object-centric'], ['D', 'Dynamic'], ['Syn', 'Synthetic'], ['Web', 'Internet photos'], ['Long', 'Long (10k+)', '10k+ frame recordings']];  // [key, label, prose]
    const FAMILIES = [
      { name: 'Feed-forward 3D/4D', color: '--primary', desc: 'Cameras, depth, and point maps in one pass; all five tasks.',
        models: ['DUST3R', 'MAST3R', 'POW3R', 'CUT3R', 'FAST3R', 'VGGT', 'FASTVGGT', 'VGGT_OMEGA', 'ROBUSTVGGT', 'OMNIVGGT', 'PI3_X', 'DA3-L', 'DA3-G', 'DA3-NGL', 'AMB3R', 'MAPANYTHING', 'DEJAVIEW', 'COME', 'IGGT', 'IGGT4D'],
        prior: ['DA3-NGL', 'PI3_X', 'MAPANYTHING', 'OMNIVGGT'] },
      { name: 'Monocular depth', color: '--sky', desc: 'Affine-invariant and metric depth specialists.',
        models: ['MARIGOLD', 'MARIGOLD_V1_1', 'MARIGOLD_V2', 'LOTUS_D', 'MOGE', 'MOGE2', 'MOGE3', 'DEPTH_PRO', 'UNIDEPTH_V2', 'INFINIDEPTH'], prior: [] },
      { name: 'Video generative', color: '--green', desc: 'Depth and camera models built on video diffusion or generation backbones: DVD, DepthCrafter and ChronoDepth for video depth; GenCeption for depth and pose; GeoNeXt for monocular depth.',
        models: ['DVD', 'DEPTHCRAFTER', 'CHRONODEPTH', 'GENCEPTION_DEPTH', 'GENCEPTION_POSE', 'GEONEXT_OFFICIAL_SVD'], prior: [] },
      { name: 'Long-sequence & SLAM', color: '--accent', desc: 'Chunked, streaming, and memory-based systems for thousands of frames.',
        models: ['VGGT_LONG', 'PI_LONG', 'VGGT_SLAM', 'LOGER', 'ZIPMAP', 'LINGBOT_MAP', 'R3', 'SCAL3R', 'ABOT_RECON'], prior: [] },
      { name: 'Vision–language', color: '--amber', desc: 'Open vision–language models, and commercial ones queried through their APIs (MonoDepth only), asked for depth or pose.',
        models: ['QWEN3_VL', 'SENSENOVA_VISION', 'DEPTHVLM', 'DEPTHLM', 'CLAUDE_OPUS_5_5', 'GPT_6_ASTRA', 'GEMINI_4_ARGON'], prior: [] },
    ];



    const host = () => document.getElementById('home-atlas');
    const css = (v) => getComputedStyle(host() || document.documentElement).getPropertyValue(v).trim();
    const mark = (v) => v === 'yes' ? '<i class="atlas-mark yes">&#10003;</i>' : v === 'part' ? '<i class="atlas-mark part">&#9680;</i>' : v === 'unk' ? '<i class="atlas-mark no">?</i>' : '<i class="atlas-mark no">&ndash;</i>';
    const fmt = (n) => n == null ? '?' : n.toLocaleString('en-US');
    const uniqueDatasets = () => { const seen = new Set(); return DATASETS.filter(d => { if (seen.has(d[2])) return false; seen.add(d[2]); return true; }); };

    function renderStatic() {
      // attribute matrix, grouped by task
      const uniq = uniqueDatasets();
      const tasksOf = (thumb) => TASKS.map(t => DATASETS.some(d => d[2] === thumb && d[0] === t.key));
      const attrsOf = (thumb) => [...new Set(DATASETS.filter(d => d[2] === thumb).flatMap(d => d[6].split(' ')))];
      const reaches = (k, t, ds = uniq) => ds.some(d => DATASETS.some(x => x[2] === d[2] && x[0] === t.key && x[6].split(' ').includes(k)));
      let m = '<thead><tr><th></th>' + ATTRS.map(a => `<th class="rot"><div>${a[1]}</div></th>`).join('') + '<th></th>' +
        TASKS.map(t => `<th class="rot"><div>${t.key}</div></th>`).join('') + '<th class="rot"><div>Images</div></th></tr></thead><tbody>';
      for (const d of uniq) {
        const at = attrsOf(d[2]);
        m += `<tr class="${d[2] === OURS ? 'mine' : ''}"><td class="name">${d[1]}</td>` + ATTRS.map(a => `<td class="cell ${at.includes(a[0]) ? 'on' : ''}" title="${a[1]}"></td>`).join('') + '<td style="width:10px"></td>' +
          tasksOf(d[2]).map((on, i) => `<td class="cell task ${on ? 'on' : ''}" style="${on ? `background:var(${TASKS[i].color})` : ''}" title="${TASKS[i].key}"></td>`).join('') +
          `<td class="cnt">${fmt(d[4])}</td></tr>`;
      }
      document.getElementById('atlas-matrix').innerHTML = m + '</tbody>';

      // side summary: how many datasets test each condition, and which tasks reach it
      const nTasks = uniq.map(d => tasksOf(d[2]).filter(Boolean).length);
      const rows = ATTRS.map(([k, label]) => {
        const ds = uniq.filter(d => attrsOf(d[2]).includes(k));
        const reach = TASKS.map(t => reaches(k, t, ds));
        return `<div class="atlas-cov-row"><span>${label}</span><span class="bar"><i style="width:${(100 * ds.length / uniq.length).toFixed(1)}%"></i></span><b>${ds.length}</b>
          <span class="dots">${reach.map((on, i) => `<i title="${TASKS[i].key}" style="${on ? `background:var(${TASKS[i].color})` : ''}"></i>`).join('')}</span></div>`;
      }).join('');
      const multi = nTasks.filter(n => n > 1).length;
      const reachN = ATTRS.map(([k]) => TASKS.filter(t => reaches(k, t)).length);
      const all5 = ATTRS.filter((_, i) => reachN[i] === TASKS.length).map(a => a[2] || a[1].toLowerCase());
      const single = ATTRS.filter((_, i) => reachN[i] === 1).map(a => a[2] || a[1].toLowerCase());
      const list = (xs) => xs.length < 3 ? xs.join(' and ') : xs.slice(0, -1).join(', ') + ', and ' + xs[xs.length - 1];
      const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
      const pub = uniq.filter(d => d[2] !== OURS);
      const real = pub.filter(d => !attrsOf(d[2]).includes('Syn')).length;
      document.getElementById('atlas-coverage-side').innerHTML = `<div class="atlas-kicker">Datasets per condition (of ${uniq.length}) and tasks that test it</div>${rows}
        <ul class="atlas-cov-facts">
          <li><b>${real}</b> of ${pub.length} public datasets are real captures; the synthetic ones add exact ground truth for rendered rooms, film, and games.</li>
          <li><b>${multi}</b> datasets feed more than one task from one copy on disk, so tasks can be compared on the same scenes.</li>
          <li>${cap(list(all5))} scenes are tested by all five tasks; ${list(single)} by only one task so far.</li>
          <li>Spatial Arena, our own captures, adds <b>${OURS_SCENES}</b> indoor and outdoor scenes, from single objects and rooms to buildings and city blocks, captured with a handheld 3-camera LiDAR rig, to all five tasks: LiDAR-inertial poses refined by bundle adjustment, a dense LiDAR map, and GGGS-rendered dense depth, every depth frame checked by hand.</li>
        </ul>`;

      // families
      document.getElementById('atlas-families').innerHTML = FAMILIES.map(f => `<div class="atlas-family"><h4><i style="background:var(${f.color})"></i>${f.name}<small>${f.models.length}</small></h4>
        <p>${f.desc}</p><div class="atlas-chips">${f.models.map(n => `<span class="${f.prior.includes(n) ? 'prior' : ''}">${n}</span>`).join('')}</div></div>`).join('');
    }

    let charts = [];
    function renderCharts() {
      charts.forEach(c => c.destroy()); charts = [];
      if (typeof Chart === 'undefined') return;
      const fg = css('--text-primary'), fg2 = css('--text-secondary'), family = css('--font-body');
      const font = (size = 13, weight = '500') => ({ family, size, weight });
      const legend = { position: 'bottom', labels: { boxWidth: 16, color: fg, padding: 14, font: font(13) } };
      const base = { responsive: true, maintainAspectRatio: false, animation: false };

      // model families
      charts.push(new Chart(document.getElementById('atlas-family-donut'), {
        type: 'doughnut',
        data: { labels: FAMILIES.map(f => f.name), datasets: [{ data: FAMILIES.map(f => f.models.length), backgroundColor: FAMILIES.map(f => css(f.color)), borderColor: css('--atlas-card-bg'), borderWidth: 3 }] },
        options: { ...base, cutout: '62%', plugins: { legend } },
        plugins: [{ id: 'center', afterDraw(c) { const { left, right, top, bottom } = c.chartArea; const g = c.ctx; g.save(); g.textAlign = 'center'; g.fillStyle = fg;
          g.font = `800 32px ${family}`; g.fillText(String(FAMILIES.reduce((n, f) => n + f.models.length, 0)), (left + right) / 2, (top + bottom) / 2 + 6); g.font = `500 13px ${family}`; g.fillStyle = fg2;
          g.fillText('models', (left + right) / 2, (top + bottom) / 2 + 26); g.restore(); } }],
      }));
    }

    // Hierarchical composition: environment -> motion -> scene type -> dataset, outer bars = evaluation cases stacked by task.
    // Each scene of the full-standard case lists is placed once (ETH3D and Tanks & Temples split per scene, Marigold per subset).
    // [environment, motion, scene type, dataset, task, cases]
    const COMPOSITION = [
      ['Indoor', 'Static', 'Room-scale', 'Marigold', 'MonoDepth', 1],
      ['Indoor', 'Static', 'Room-scale', 'iBims-1', 'MonoDepth', 1],
      ['Indoor', 'Static', 'Room-scale', '7-Scenes', 'MVRecon', 7], ['Indoor', 'Static', 'Room-scale', '7-Scenes', 'SLAM', 36],
      ['Indoor', 'Static', 'Room-scale', 'ETH3D', 'MVRecon', 7], ['Indoor', 'Static', 'Room-scale', 'ETH3D', 'SfM', 7],
      ['Indoor', 'Static', 'Room-scale', 'T&T', 'MVRecon', 1], ['Indoor', 'Static', 'Room-scale', 'T&T', 'SfM', 1],
      ['Indoor', 'Static', 'Room-scale', 'ScanNet++', 'MVRecon', 20], ['Indoor', 'Static', 'Room-scale', 'ScanNet++', 'SfM', 20],
      ['Indoor', 'Static', 'Room-scale', 'HiRoom', 'MVRecon', 30], ['Indoor', 'Static', 'Room-scale', 'HiRoom', 'SfM', 30],
      ['Indoor', 'Static', 'Room-scale', 'NRGBD', 'MVRecon', 9],
      ['Indoor', 'Static', 'Object-centric', 'HAMMER', 'MonoDepth', 1],
      ['Indoor', 'Static', 'Object-centric', 'DTU', 'MVRecon', 15], ['Indoor', 'Static', 'Object-centric', 'DTU', 'SfM', 15],
      ['Indoor', 'Static', 'Object-centric', 'TexturePoorSfM', 'SfM', 30],
      ['Indoor', 'Dynamic', 'Room-scale', 'Bonn', 'VideoDepth', 26],
      ['Indoor', 'Dynamic', 'Room-scale', 'TUM RGB-D', 'SLAM', 18],
      ['Outdoor', 'Static', 'Buildings', 'ETH3D', 'MVRecon', 4], ['Outdoor', 'Static', 'Buildings', 'ETH3D', 'SfM', 6],
      ['Outdoor', 'Static', 'Buildings', 'T&T', 'MVRecon', 5], ['Outdoor', 'Static', 'Buildings', 'T&T', 'SfM', 5],
      ['Outdoor', 'Static', 'Buildings', 'IMC', 'SfM', 8],
      ['Outdoor', 'Static', 'Campus', 'Oxford Spires', 'SLAM', 20],
      ['Outdoor', 'Dynamic', 'Driving', 'Marigold', 'MonoDepth', 1],
      ['Outdoor', 'Dynamic', 'Driving', 'KITTI', 'VideoDepth', 13],
      ['Outdoor', 'Dynamic', 'Driving', 'KITTI Odo.', 'SLAM', 11],
      ['Outdoor', 'Dynamic', 'City-scale', 'VBR', 'SLAM', 7],
      ['Outdoor', 'Dynamic', 'Film & game', 'Spring', 'MonoDepth', 1],
      ['Outdoor', 'Dynamic', 'Film & game', 'Synth4K', 'MonoDepth', 5],
      ['Aerial', 'Dynamic', 'UAV', 'AerialMetric', 'MonoDepth', 3],
      ['Mixed', 'Static', 'Scans', 'Marigold', 'MonoDepth', 2],
      ['Mixed', 'Dynamic', 'Film & game', 'Sintel', 'VideoDepth', 23],
      ['Mixed', 'Dynamic', 'Handheld', 'DyCheck', 'VideoDepth', 12],
      // Spatial Arena (ours), one case per MonoDepth scene, VideoDepth clip, MVRecon/SfM scene and SLAM sequence. Scene type from
      // each scene's setting and extent; dynamic = people or vehicles in over 3% of the pixels (or 60% of the frames) the case uses.
      ['Indoor', 'Static', 'Room-scale', 'SpatialArena', 'MonoDepth', 7], ['Indoor', 'Static', 'Room-scale', 'SpatialArena', 'VideoDepth', 7],
      ['Indoor', 'Static', 'Room-scale', 'SpatialArena', 'MVRecon', 1], ['Indoor', 'Static', 'Room-scale', 'SpatialArena', 'SfM', 2],
      ['Indoor', 'Static', 'Room-scale', 'SpatialArena', 'SLAM', 1],
      ['Indoor', 'Static', 'Object-centric', 'SpatialArena', 'MonoDepth', 1], ['Indoor', 'Static', 'Object-centric', 'SpatialArena', 'MVRecon', 1],
      ['Indoor', 'Dynamic', 'Room-scale', 'SpatialArena', 'MonoDepth', 2], ['Indoor', 'Dynamic', 'Room-scale', 'SpatialArena', 'MVRecon', 1],
      ['Indoor', 'Dynamic', 'Room-scale', 'SpatialArena', 'SfM', 1],
      ['Indoor', 'Dynamic', 'Buildings', 'SpatialArena', 'SLAM', 1],
      ['Outdoor', 'Static', 'Buildings', 'SpatialArena', 'MonoDepth', 1], ['Outdoor', 'Static', 'Buildings', 'SpatialArena', 'MVRecon', 1],
      ['Outdoor', 'Static', 'Campus', 'SpatialArena', 'SLAM', 1],
      ['Outdoor', 'Static', 'City-scale', 'SpatialArena', 'MonoDepth', 3], ['Outdoor', 'Static', 'City-scale', 'SpatialArena', 'VideoDepth', 5],
      ['Outdoor', 'Static', 'City-scale', 'SpatialArena', 'SLAM', 1],
      ['Outdoor', 'Static', 'Small-scale', 'SpatialArena', 'MonoDepth', 2], ['Outdoor', 'Static', 'Small-scale', 'SpatialArena', 'VideoDepth', 1],
      ['Outdoor', 'Static', 'Small-scale', 'SpatialArena', 'MVRecon', 1], ['Outdoor', 'Static', 'Small-scale', 'SpatialArena', 'SfM', 1],
      ['Outdoor', 'Static', 'Parks', 'SpatialArena', 'MonoDepth', 3], ['Outdoor', 'Static', 'Parks', 'SpatialArena', 'VideoDepth', 2],
      ['Outdoor', 'Static', 'Parks', 'SpatialArena', 'MVRecon', 1], ['Outdoor', 'Static', 'Parks', 'SpatialArena', 'SLAM', 2],
      ['Outdoor', 'Dynamic', 'Buildings', 'SpatialArena', 'SfM', 1], ['Outdoor', 'Dynamic', 'Buildings', 'SpatialArena', 'SLAM', 1],
      ['Outdoor', 'Dynamic', 'City-scale', 'SpatialArena', 'SLAM', 1],
      ['Outdoor', 'Dynamic', 'Object-centric', 'SpatialArena', 'SfM', 2],
    ];
    const OURS_LABEL = 'SpatialArena';
    const alphaHex = (c, a) => { const x = c.replace('#', ''); const n = parseInt(x.length === 3 ? x.split('').map(s => s + s).join('') : x, 16); return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`; };
    const ENV_COLORS = { Indoor: '#2f6fd6', Outdoor: '#e0702a', Aerial: '#11966b', Mixed: '#8b5cf6' };

    function buildCompositionTree() {
      const root = { name: 'all', children: [] };
      const child = (n, name) => { let c = n.children.find(x => x.name === name); if (!c) { c = { name, children: [] }; n.children.push(c); } return c; };
      for (const [env, mot, cat, ds, task, n] of COMPOSITION) {
        const leaf = child(child(child(child(root, env), mot), cat), ds);
        leaf.tasks = leaf.tasks || {}; leaf.tasks[task] = (leaf.tasks[task] || 0) + n;
      }
      const sum = (n) => (n.value = n.tasks ? Object.values(n.tasks).reduce((a, b) => a + b, 0) : n.children.reduce((a, c) => a + sum(c), 0));
      const leaves = (n) => (n.leaves = n.tasks ? 1 : n.children.reduce((a, c) => a + leaves(c), 0));
      sum(root); leaves(root);
      return root;
    }

    function renderComposition() {
      const host = document.getElementById('atlas-composition'); if (!host) return;
      const NS = 'http://www.w3.org/2000/svg';
      const W = 1010, C = W / 2, font = css('--font-body'), fg = css('--text-primary'), fg2 = css('--text-secondary'), bg = css('--card-bg') || '#ffffff';
      const R = { env: [64, 104], mot: [106, 146], cat: [148, 250], bar0: 256, barMax: 332 };
      const root = buildCompositionTree();
      const mix = (a, b, t) => { const p = (h) => { h = h.replace('#', ''); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)); };
        const x = p(a), y = p(b.startsWith('#') ? b : '#ffffff'); return '#' + x.map((v, i) => Math.round(v + (y[i] - v) * t).toString(16).padStart(2, '0')).join(''); };
      const pt = (r, a) => [C + r * Math.sin(a), C - r * Math.cos(a)];
      const sector = (r0, r1, a0, a1) => {
        const big = a1 - a0 > Math.PI ? 1 : 0, [x0, y0] = pt(r1, a0), [x1, y1] = pt(r1, a1), [x2, y2] = pt(r0, a1), [x3, y3] = pt(r0, a0);
        return `M${x0},${y0}A${r1},${r1} 0 ${big} 1 ${x1},${y1}L${x2},${y2}A${r0},${r0} 0 ${big} 0 ${x3},${y3}Z`;
      };
      const svg = document.createElementNS(NS, 'svg');
      svg.setAttribute('viewBox', `0 0 ${W} ${W}`); svg.setAttribute('width', W); svg.setAttribute('height', W);
      svg.setAttribute('class', 'vector-figure'); svg.setAttribute('role', 'img');
      svg.setAttribute('aria-label', 'Hierarchical composition of SpatialArena: environment, motion, scene type, and evaluation cases per dataset');
      // rgba() fills and strokes become black in many SVG editors and converters: write rgb plus an opacity attribute instead
      const setColor = (e, k, v) => { const m = /^rgba\(([^,]+),([^,]+),([^,]+),([^)]+)\)$/.exec(String(v).replace(/\s/g, ''));
        if (!m || !['fill', 'stroke'].includes(k)) return e.setAttribute(k, v);
        e.setAttribute(k, `rgb(${m[1]},${m[2]},${m[3]})`); e.setAttribute(`${k}-opacity`, m[4]); };
      const add = (tag, attrs, parent = svg, text) => { const e = document.createElementNS(NS, tag); for (const k in attrs) setColor(e, k, attrs[k]); if (text != null) e.textContent = text; parent.appendChild(e); return e; };
      const label = (text, r, a, { size = 14, weight = 600, color = fg, radial = false } = {}) => {
        const [x, y] = pt(r, a); let deg = a * 180 / Math.PI, anchor = 'middle';
        if (radial) { deg -= 90; anchor = 'start'; if (a > Math.PI) { deg += 180; anchor = 'end'; } }
        else if (a > Math.PI / 2 && a < 3 * Math.PI / 2) deg += 180;
        return add('text', { x, y, transform: `rotate(${deg.toFixed(2)} ${x} ${y})`, 'text-anchor': anchor, 'dominant-baseline': 'central',
          'font-family': font, 'font-size': size, 'font-weight': weight, fill: color }, svg, text);
      };
      const gap = 0.004, total = root.leaves;
      let maxV = 0; (function f(n) { if (n.tasks) maxV = Math.max(maxV, n.value); else n.children.forEach(f); })(root);
      const barLen = (v) => (R.barMax - R.bar0) * (0.12 + 0.88 * Math.sqrt(v / maxV));
      // light radial guides for the outer bars
      for (const v of [10, 30, 60]) {
        const r = R.bar0 + barLen(v);
        add('circle', { cx: C, cy: C, r, fill: 'none', stroke: css('--grid-line'), 'stroke-dasharray': '3 4' });
        add('text', { x: C + 3, y: C - r - 3, 'font-family': font, 'font-size': 12, fill: fg2 }, svg, v);
      }
      let a = 0;
      for (const env of root.children) {
        const base = ENV_COLORS[env.name], e0 = a;
        for (const mot of env.children) {
          const m0 = a;
          for (const cat of mot.children) {
            const c0 = a;
            for (const ds of cat.children) {
              const a0 = a, a1 = a + 2 * Math.PI * ds.leaves / total; let r = R.bar0;
              const mine = ds.name === OURS_LABEL;
              if (mine) add('path', { d: sector(R.cat[1] + 2, R.barMax + 6, a0 + gap, a1 - gap), fill: css('--primary'), 'fill-opacity': 0.16,
                stroke: css('--primary'), 'stroke-width': 1.6 }, svg);
              for (const t of TASKS) {
                const n = ds.tasks[t.key]; if (!n) continue;
                const r1 = r + barLen(ds.value) * n / ds.value;
                add('path', { d: sector(r, r1, a0 + gap * 2, a1 - gap * 2), fill: css(t.color) }, svg);
                add('title', {}, svg.lastChild, `${ds.name} · ${t.key}: ${n} cases`);
                r = r1;
              }
              label(`${ds.name}  ${ds.value}`, (mine ? Math.max(r, R.barMax + 6) : r) + 6, (a0 + a1) / 2,
                { size: 16, weight: mine ? 800 : 600, color: mine ? css('--primary') : fg, radial: true });
              a = a1;
            }
            add('path', { d: sector(R.cat[0], R.cat[1], c0 + gap, a - gap), fill: mix(base, bg, 0.62) });
            add('title', {}, svg.lastChild, `${env.name} · ${mot.name} · ${cat.name}: ${cat.value} cases`);
            const span = (a - c0) * (R.cat[0] + R.cat[1]) / 2;
            if (span > 70) label(cat.name, (R.cat[0] + R.cat[1]) / 2, (c0 + a) / 2, { size: 14 });
            else { const short = cat.name === 'Object-centric' ? 'Objects' : cat.name;  // radial labels in narrow wedges
              label(short, R.cat[0] + 6, (c0 + a) / 2, { size: Math.min(13.5, (R.cat[1] - R.cat[0] - 12) / (0.58 * short.length)), radial: true }); }
          }
          add('path', { d: sector(R.mot[0], R.mot[1], m0 + gap, a - gap), fill: mix(base, bg, 0.38) });
          add('title', {}, svg.lastChild, `${env.name} · ${mot.name}: ${mot.value} cases`);
          if ((a - m0) * R.mot[1] > 52) label(mot.name, (R.mot[0] + R.mot[1]) / 2, (m0 + a) / 2, { size: 14, color: '#1f2937' });
        }
        add('path', { d: sector(R.env[0], R.env[1], e0 + gap, a - gap), fill: base });
        add('title', {}, svg.lastChild, `${env.name}: ${env.value} cases`);
        if ((a - e0) * R.env[1] > 40) label(env.name, (R.env[0] + R.env[1]) / 2, (e0 + a) / 2, { size: 15, weight: 700, color: '#ffffff' });
      }
      add('text', { x: C, y: C - 24, 'text-anchor': 'middle', 'font-family': font, 'font-size': 15, 'font-weight': 700, fill: css('--primary') }, svg, 'SpatialArena');
      add('text', { x: C, y: C + 10, 'text-anchor': 'middle', 'font-family': font, 'font-size': 36, 'font-weight': 800, fill: fg }, svg, String(root.value));
      add('text', { x: C, y: C + 32, 'text-anchor': 'middle', 'font-family': font, 'font-size': 13, 'font-weight': 500, fill: fg2 }, svg, 'evaluation cases');
      host.replaceChildren(svg);

      const legend = document.getElementById('atlas-composition-legend');
      const envTotals = root.children.map(e => `<span><i style="background:${ENV_COLORS[e.name]}"></i>${e.name} <b>${e.value}</b></span>`).join('');
      const taskTotals = TASKS.map(t => `<span><i style="background:var(${t.color})"></i>${t.key} <b>${COMPOSITION.filter(r => r[4] === t.key).reduce((s, r) => s + r[5], 0)}</b></span>`).join('');
      legend.innerHTML = `<div class="atlas-kicker">Environment (inner ring)</div><div class="atlas-comp-keys">${envTotals}</div>
        <div class="atlas-kicker">Task (outer bars)</div><div class="atlas-comp-keys">${taskTotals}</div>
        <div class="atlas-comp-keys"><span><i style="background:${alphaHex(css('--primary'), 0.25)};box-shadow:inset 0 0 0 1.5px var(--primary)"></i><b style="color:var(--primary)">${OURS_LABEL}</b>&nbsp;${COMPOSITION.filter(r => r[3] === OURS_LABEL).reduce((s, r) => s + r[5], 0)} cases</span></div>`;
    }


    let staticDone = false;
    function visible() {
        const panel = document.getElementById('benchmark-view-home');
        return panel && !panel.hidden && host();
    }
    function renderAll() {
        if (!visible()) return;
        if (!staticDone) { renderStatic(); staticDone = true; }
        renderCharts();
        renderComposition();
    }
    window.renderSpatialArenaAtlas = renderAll;

    // "SpatialArena at a Glance" shows composition, scene coverage and models as tabs of one slide. A chart drawn while
    // its tab was hidden has no size yet, so the visible tab's charts are resized when it opens.
    function selectGlanceTab(key, focus = false) {
      document.querySelectorAll('[data-glance-tab]').forEach(tab => {
        const on = tab.dataset.glanceTab === key;
        tab.setAttribute('aria-selected', on ? 'true' : 'false');
        tab.tabIndex = on ? 0 : -1;
        if (on && focus) tab.focus();
      });
      document.querySelectorAll('[data-glance-panel]').forEach(panel => { panel.hidden = panel.dataset.glancePanel !== key; });
      charts.forEach(chart => chart.resize());
    }
    function initGlanceTabs() {
      const list = document.getElementById('glance-tabs'); if (!list) return;
      const keys = [...list.querySelectorAll('[data-glance-tab]')].map(tab => tab.dataset.glanceTab);
      list.addEventListener('click', event => {
        const tab = event.target.closest('[data-glance-tab]');
        if (tab) selectGlanceTab(tab.dataset.glanceTab);
      });
      list.addEventListener('keydown', event => {
        if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return;
        const current = keys.indexOf(list.querySelector('[aria-selected="true"]')?.dataset.glanceTab);
        const next = keys[(current + (event.key === 'ArrowRight' ? 1 : keys.length - 1)) % keys.length];
        event.preventDefault();
        selectGlanceTab(next, true);
      });
      selectGlanceTab(keys[0]);
    }
    document.addEventListener('DOMContentLoaded', initGlanceTabs);
    new MutationObserver(renderAll).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    document.addEventListener('DOMContentLoaded', () => window.setTimeout(renderAll, 0));
})();
