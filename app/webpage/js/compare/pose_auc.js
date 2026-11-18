// Copyright (c) 2026 Yu Chen. All rights reserved.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

// Relative pose accuracy of a trajectory payload (spatialarena/eval/aligned_trajectory.py), computed as
// spatialarena/eval/sparse_reconstruction_evaluator.py does: for frame pairs, the error max(relative rotation
// angle, relative translation direction angle), and the AUC of its recall curve. No three.js, so Node can test it.

// Pose AUC: the evaluator's relative pose error max(rotation, translation direction) over frame pairs (evenly
// spaced over all pairs beyond AUC_MAX_PAIRS), its AUC thresholds and min_baseline.
const AUC_MAX_PAIRS = 300000;
export const AUC_THRESHOLDS = [5, 10, 30];
export const AUC_CURVE_MAX = 30;
const MIN_BASELINE = 1e-8;

function quaternionToMatrix(quaternion) {
    // The payload rounds quaternions to 4 decimals: renormalize so the matrix is a rotation.
    const norm = Math.hypot(...quaternion) || 1;
    const [w, x, y, z] = quaternion.map(value => value / norm);
    return [
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ];
}

// World-to-camera rotations (row-major 3x3) and translations of camera-to-world positions and quaternions.
function worldToCamera(positions, quaternions, indices) {
    const rotations = new Float64Array(9 * indices.length);
    const translations = new Float64Array(3 * indices.length);
    indices.forEach((frame, k) => {
        const c = quaternionToMatrix(quaternions[frame]);
        const p = positions[frame];
        for (let r = 0; r < 3; r += 1) {
            for (let col = 0; col < 3; col += 1) rotations[9 * k + 3 * r + col] = c[3 * col + r];  // transpose
            translations[3 * k + r] = -(c[r] * p[0] + c[3 + r] * p[1] + c[6 + r] * p[2]);
        }
    });
    return { rotations, translations };
}

// Relative pose (second relative to first) of world-to-camera poses a and b: R = Rb Ra^T, t = tb - R ta.
function relativePose(rotations, translations, a, b, outR, outT) {
    for (let r = 0; r < 3; r += 1) {
        for (let col = 0; col < 3; col += 1) {
            let sum = 0;
            for (let k = 0; k < 3; k += 1) sum += rotations[9 * b + 3 * r + k] * rotations[9 * a + 3 * col + k];
            outR[3 * r + col] = sum;
        }
    }
    for (let r = 0; r < 3; r += 1) {
        outT[r] = translations[3 * b + r] - (outR[3 * r] * translations[3 * a] + outR[3 * r + 1] * translations[3 * a + 1]
            + outR[3 * r + 2] * translations[3 * a + 2]);
    }
}

function angleDeg(cosine) {
    return (Math.acos(Math.min(1, Math.max(-1, cosine))) * 180) / Math.PI;
}

// Sorted relative pose errors (degrees) of a trajectory payload: frames with a predicted pose, all pairs (or
// AUC_MAX_PAIRS evenly spaced ones). Relative poses do not depend on the Sim(3) alignment.
export function relativePoseErrors(data) {
    const frames = [];
    data.pred.positions.forEach((position, index) => {
        if (position && data.pred.quaternions[index] && data.gt.positions[index]) frames.push(index);
    });
    const count = frames.length;
    const total = (count * (count - 1)) / 2;
    if (total < 1) return { errors: new Float64Array(0), pairs: 0, totalPairs: 0 };
    const gt = worldToCamera(data.gt.positions, data.gt.quaternions, frames);
    const pred = worldToCamera(data.pred.positions, data.pred.quaternions, frames);
    const kept = Math.min(total, AUC_MAX_PAIRS);
    const errors = new Float64Array(kept);
    const gtR = new Float64Array(9); const gtT = new Float64Array(3);
    const predR = new Float64Array(9); const predT = new Float64Array(3);
    let pair = 0;
    let filled = 0;
    for (let a = 0; a < count && filled < kept; a += 1) {
        for (let b = a + 1; b < count && filled < kept; b += 1, pair += 1) {
            // np.linspace(0, total - 1, kept) as in selected_pairs, all pairs when they fit.
            const target = kept === total ? pair : Math.floor((filled * (total - 1)) / Math.max(1, kept - 1));
            if (pair < target) continue;
            relativePose(gt.rotations, gt.translations, a, b, gtR, gtT);
            relativePose(pred.rotations, pred.translations, a, b, predR, predT);
            let trace = 0;
            for (let k = 0; k < 9; k += 1) trace += predR[k] * gtR[k];
            const rotation = angleDeg((trace - 1) / 2);
            const gtNorm = Math.hypot(gtT[0], gtT[1], gtT[2]);
            const predNorm = Math.hypot(predT[0], predT[1], predT[2]);
            const translation = gtNorm < MIN_BASELINE || predNorm <= 0 ? 0
                : angleDeg((gtT[0] * predT[0] + gtT[1] * predT[1] + gtT[2] * predT[2]) / (gtNorm * predNorm));
            errors[filled] = Math.max(rotation, translation);
            filled += 1;
        }
    }
    const sorted = errors.subarray(0, filled).sort();
    return { errors: sorted, pairs: filled, totalPairs: total };
}

// AUC at `threshold` of sorted errors: the trapezoid area under the recall curve up to it, divided by it
// (sparse_reconstruction_evaluator.compute_auc).
export function poseAuc(errors, threshold) {
    const n = errors.length;
    let below = 0;
    while (below < n && errors[below] <= threshold) below += 1;
    if (!below) return 0;
    let area = 0;
    let previousX = 0;
    let previousY = 0;
    for (let i = 0; i < below; i += 1) {
        const y = (i + 1) / n;
        area += ((errors[i] - previousX) * (previousY + y)) / 2;
        previousX = errors[i];
        previousY = y;
    }
    area += (threshold - previousX) * previousY;
    return area / threshold;
}
