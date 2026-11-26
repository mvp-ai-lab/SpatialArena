// Submit page: the model submission form.
//
// The form, its validation, the draft and the submission file all run in the browser. Sending a submission and
// checking its status need the evaluation backend, which GitHub Pages cannot host. Until the site runs on our own
// servers, SUBMIT_API is empty and public models are submitted by email instead: Submit saves the submission file
// and opens the visitor's email app addressed to SUBMIT_EMAIL, with the submission in the message, and we start the
// evaluation by hand. Email submissions must be public (code and weights reachable without a login), which the
// visitor confirms with the agree_public checkbox. With neither address set, a filled form can only be downloaded.
//
// Backend contract (for the server side, once it exists):
//   POST {SUBMIT_API}/submissions          body: the submission file (JSON, schema SUBMISSION_SCHEMA)
//     -> 201 {"id": "<submission id>", "status": "queued"}
//   GET  {SUBMIT_API}/submissions/{id}
//     -> 200 {"id", "status": "queued" | "running" | "scored" | "failed", "message"?, "runs"?: [{task, track,
//        dataset, status}]}
// SUBMIT_API comes from <meta name="spatialarena-submit-api" content="https://...">, so the backend can be switched
// on without changing this file.
(function () {
    const SUBMISSION_SCHEMA = "spatialarena-submission/1";
    const SUBMIT_API = (document.querySelector('meta[name="spatialarena-submit-api"]')?.content || "").replace(/\/+$/, "");
    // <meta name="spatialarena-submit-email" content="...">: where email submissions go while SUBMIT_API is empty.
    const SUBMIT_EMAIL = (document.querySelector('meta[name="spatialarena-submit-email"]')?.content || "").trim();
    // <meta name="spatialarena-submit-cc" content="a@x,b@y">: copied on email submissions.
    const SUBMIT_CC = (document.querySelector('meta[name="spatialarena-submit-cc"]')?.content || "")
        .split(",").map(address => address.trim()).filter(Boolean);
    const EMAIL_MODE = !SUBMIT_API && Boolean(SUBMIT_EMAIL);
    // Longer mailto: links are cut by some browsers and email apps; the message then asks for the file instead.
    const MAILTO_LIMIT = 1800;
    const DRAFT_KEY = "spatialarena-submit-draft";

    const TASKS = ["MonoDepth", "VideoDepth", "MVRecon", "SfM", "SLAM"];
    // spatialarena/task_track_matrix.py TASK_TRACK_MATRIX: the tracks each task runs.
    const TASK_TRACKS = {
        MonoDepth: ["rgb", "rgb_masked_depth"],
        VideoDepth: ["rgb", "rgb_cameras", "rgb_masked_depth", "rgb_sparse_depth"],
        MVRecon: ["rgb", "rgb_cameras", "rgb_depth", "rgb_masked_depth", "rgb_masked_depth_cameras", "rgb_sparse_depth",
            "rgb_sparse_depth_cameras"],
        SfM: ["rgb", "rgb_depth", "rgb_masked_depth", "rgb_masked_depth_cameras", "rgb_sparse_depth",
            "rgb_sparse_depth_cameras"],
        SLAM: ["rgb", "rgb_depth", "rgb_masked_depth"],
    };
    const TRACKS = [
        ["rgb", "RGB"],
        ["rgb_cameras", "RGB + cameras"],
        ["rgb_depth", "RGB + depth"],
        ["rgb_masked_depth", "RGB + masked depth"],
        ["rgb_masked_depth_cameras", "RGB + masked depth + cameras"],
        ["rgb_sparse_depth", "RGB + sparse depth"],
        ["rgb_sparse_depth_cameras", "RGB + sparse depth + cameras"],
    ];
    // The release's datasets per task (as on the Download page).
    const DATASETS = {
        MonoDepth: ["aerialmetric", "hammer", "ibims1", "marigold", "spatialarena", "spring", "synth4k"],
        VideoDepth: ["bonn", "dycheck", "kitti", "sintel", "spatialarena"],
        MVRecon: ["7scenes", "dtu", "eth3d", "hiroom", "nrgbd", "scannetpp", "spatialarena", "tanks_and_temples"],
        SfM: ["dtu", "eth3d", "hiroom", "imc", "scannetpp", "spatialarena", "tanks_and_temples", "texturepoorsfm"],
        SLAM: ["7scenes", "kitti_odometry", "oxford_spires", "spatialarena", "tum_dynamics", "vbr"],
    };
    // The command template's placeholders (scripts/inference/external_command_adapter.py).
    const PLACEHOLDERS = [
        ["{images_dir}", "The folder of the scene's images"],
        ["{output_path}", "An empty folder where your script writes its results"],
        ["{code_path}", "Your repository, cloned at the commit you give"],
        ["{ckpt_path}", "Your weights, downloaded from the checkpoint link"],
        ["{data_path}", "The whole scene folder (images/ and, on conditioned tracks, their inputs)"],
        ["{camera_dir}", "Ground-truth cameras (COLMAP), on the camera-conditioned tracks"],
        ["{depth_dir}", "Ground-truth depth maps, on the depth-conditioned tracks"],
        ["{track}", "The track being run, e.g. rgb"],
        ["{resolution}", "Our input-resolution hint, in pixels"],
    ];
    // The command of scripts/inference/submission_template.py, copied into the repository as spatialarena_infer.py.
    const TEMPLATE_COMMAND = "python {code_path}/spatialarena_infer.py --images {images_dir} --output {output_path} "
        + "--checkpoint {ckpt_path}";
    // The command must write to {output_path} and read the scene from {data_path} or its images from {images_dir}.
    const INPUT_PLACEHOLDERS = ["{data_path}", "{images_dir}"];

    const state = { tasks: new Set(TASKS), tracks: new Set(["rgb"]) };
    let form;
    let saveTimer = 0;

    const byId = id => document.getElementById(id);
    const field = name => form.elements.namedItem(name);
    const value = name => String(field(name)?.value || "").trim();
    const isHttpsUrl = text => {
        try {
            return new URL(text).protocol === "https:";
        } catch (_error) {
            return false;
        }
    };

    // ----- task x track picker -----

    function renderPicker() {
        byId("submit-tasks").innerHTML = TASKS.map(task => `
            <button type="button" class="download-chip${state.tasks.has(task) ? " active" : ""}" data-submit-task="${task}"
                aria-pressed="${state.tasks.has(task)}">${task}</button>`).join("");
        byId("submit-tracks").innerHTML = TRACKS.map(([track, label]) => `
            <button type="button" class="download-chip${state.tracks.has(track) ? " active" : ""}" data-submit-track="${track}"
                aria-pressed="${state.tracks.has(track)}">${label}</button>`).join("");
        const tracks = TRACKS.filter(([track]) => state.tracks.has(track));
        const head = `<tr><th scope="col">Track</th>${TASKS.map(task => `<th scope="col"${state.tasks.has(task) ? "" : ' class="is-off"'}>${task}</th>`).join("")}</tr>`;
        const body = tracks.map(([track, label]) => `<tr><th scope="row">${label}</th>${TASKS.map(task => {
            const runs = TASK_TRACKS[task].includes(track);
            const on = runs && state.tasks.has(task);
            const text = on ? '<i class="fas fa-check" aria-hidden="true"></i><span class="sr-only">evaluated</span>'
                : runs ? '<span class="sr-only">not selected</span>&ndash;' : "n/a";
            const title = runs ? (on ? "Evaluated" : "Task not selected") : `${task} does not run this track`;
            return `<td class="${on ? "is-on" : runs ? "" : "is-na"}" title="${title}">${text}</td>`;
        }).join("")}</tr>`).join("");
        byId("submit-matrix").innerHTML = tracks.length ? `<thead>${head}</thead><tbody>${body}</tbody>` : "";
        const cells = evaluationCells();
        byId("submit-matrix-summary").textContent = cells.length
            ? `${cells.length} task–track pair${cells.length === 1 ? "" : "s"} to evaluate. The planner keeps the datasets each track runs on.`
            : "Pick at least one task and a track it runs.";
    }

    function evaluationCells() {
        const cells = [];
        for (const task of TASKS) {
            if (!state.tasks.has(task)) continue;
            for (const [track] of TRACKS) {
                if (state.tracks.has(track) && TASK_TRACKS[task].includes(track)) cells.push({ task, track });
            }
        }
        return cells;
    }

    // ----- the submission file and its previews -----

    function submission() {
        const environment = value("env_kind") === "docker"
            ? { docker_image: value("docker_image") }
            : { python: value("python_version"), install: value("install") };
        return {
            schema: SUBMISSION_SCHEMA,
            model: {
                name: value("model_name"),
                display_name: value("display_name") || value("model_name"),
                description: value("description"),
                paper_url: value("paper_url") || null,
                code_url: value("code_url"),
                code_revision: value("code_revision") || null,
                license: value("license"),
                weights_license: value("weights_license") === "same" ? value("license") : value("weights_license"),
            },
            evaluation: {
                scope: value("scope"),
                tasks: TASKS.filter(task => state.tasks.has(task)),
                tracks: TRACKS.map(([track]) => track).filter(track => state.tracks.has(track)),
                pairs: evaluationCells(),
            },
            inference: {
                command: value("command"),
                checkpoint_url: value("checkpoint_url") || null,
                environment,
                gpu_memory_gb: Number(value("gpu_memory")) || null,
                notes: value("notes") || null,
            },
            contact: {
                name: value("contact_name"),
                email: value("contact_email"),
                affiliation: value("affiliation") || null,
            },
            visibility: value("visibility"),
        };
    }

    // A YAML scalar: JSON strings are valid YAML, so anything that is not a plain word is written as one.
    const yamlString = text => (/^[A-Za-z0-9_./-]+$/.test(text) ? text : JSON.stringify(text));

    function configYaml(data) {
        const datasets = data.evaluation.tasks.flatMap(task => (data.evaluation.scope === "spatialarena"
            ? ["spatialarena"] : DATASETS[task]).map(dataset => `  - {task: ${task}, dataset: ${dataset}}`));
        const name = data.model.name || "MyModel";
        // We check out the code at the submitted commit and download the checkpoint under the submission's folder.
        const folder = `\${SPATIALARENA_ROOT}/submissions/${name}`;
        const models = data.evaluation.tracks.flatMap(track => [
            `  - name: ${yamlString(name)}`,
            `    track: ${track}`,
            `    command: ${yamlString(data.inference.command || "python {code_path}/infer.py --images {images_dir} --output {output_path}")}`,
            `    code_path: ${folder}/code`,
            ...(data.inference.checkpoint_url ? [`    ckpt_path: ${folder}/checkpoint`] : []),
        ]);
        return [
            `# How we will run ${name}: python -m spatialarena.run_config <this file> plan`,
            "job_system: slurm",
            "benchmark_root: ${SPATIALARENA_ROOT}/benchmarks",
            `experiment_name: ${yamlString(`submission_${name}`)}`,
            "",
            "datasets:",
            ...(datasets.length ? datasets : ["  []"]),
            "",
            "models:",
            ...(models.length ? models : ["  []"]),
        ].join("\n");
    }

    function renderPreview() {
        const data = submission();
        byId("submit-preview-json").textContent = JSON.stringify(data, null, 2);
        byId("submit-preview-yaml").textContent = configYaml(data);
    }

    // ----- validation -----

    function check() {
        const errors = [];
        const add = (name, message) => errors.push({ name, message });
        if (!/^[A-Za-z][A-Za-z0-9_.-]{1,39}$/.test(value("model_name"))) {
            add("model_name", "2–40 characters: letters, digits, \"_\", \".\" or \"-\", starting with a letter.");
        }
        if (value("description").length < 20) add("description", "A sentence or two (at least 20 characters).");
        if (value("paper_url") && !isHttpsUrl(value("paper_url"))) add("paper_url", "An https:// link, or leave it empty.");
        if (!isHttpsUrl(value("code_url"))) add("code_url", "An https:// link to the repository with your inference code.");
        if (!value("license")) add("license", "Choose the license of your code and weights.");
        if (!evaluationCells().length) add("tasks", "Pick at least one task and a track it runs.");
        const command = value("command");
        const missing = [];
        if (!command.includes("{output_path}")) missing.push("{output_path}");
        if (!INPUT_PLACEHOLDERS.some(placeholder => command.includes(placeholder))) missing.push("{data_path} or {images_dir}");
        // A script given by a relative path would be looked up in our checkout, where the command runs.
        const relativeScript = command.split(/\s+/).find(word => /\.(py|sh)$/.test(word) && !word.startsWith("{")
            && !word.startsWith("/"));
        if (!command) add("command", "The command that runs your model on one scene.");
        else if (missing.length) add("command", `The command must use ${missing.join(" and ")}.`);
        else if (relativeScript) {
            add("command", `The command runs from our checkout: write ${relativeScript} as {code_path}/${relativeScript}.`);
        }
        if (value("checkpoint_url") && !isHttpsUrl(value("checkpoint_url"))) {
            add("checkpoint_url", "An https:// link, or leave it empty if the code downloads the weights.");
        }
        if (!["32", "128"].includes(value("gpu_memory"))) add("gpu_memory", "Choose the GPU memory your model needs.");
        if (value("env_kind") === "docker") {
            if (!/^[\w.-]+(?::\d+)?(\/[\w.-]+)+(:[\w.-]+|@sha256:[a-f0-9]{64})?$|^[\w.-]+(:[\w.-]+)?$/.test(value("docker_image"))) {
                add("docker_image", "A public image, e.g. ghcr.io/you/model:1.0.");
            }
        } else if (!value("install")) {
            add("install", "The command that installs your code's dependencies, e.g. pip install -e .");
        }
        if (!value("contact_name")) add("contact_name", "Who we contact about this submission.");
        if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value("contact_email"))) add("contact_email", "A valid email address.");
        for (const name of EMAIL_MODE ? ["agree_public", "agree_code", "agree_terms"] : ["agree_code", "agree_terms"]) {
            if (!field(name).checked) add(name, "Required to submit.");
        }
        return errors;
    }

    function showErrors(errors) {
        form.querySelectorAll(".submit-error").forEach(node => node.remove());
        form.querySelectorAll("[aria-invalid]").forEach(node => node.removeAttribute("aria-invalid"));
        for (const { name, message } of errors) {
            const anchor = name === "tasks" ? byId("submit-matrix-summary") : field(name);
            if (!anchor) continue;
            const note = document.createElement("p");
            note.className = "submit-error";
            note.id = `submit-error-${name}`;
            note.textContent = message;
            const holder = anchor.closest(".submit-field, .submit-check") || anchor;
            holder.appendChild(note);
            if (name !== "tasks") {
                anchor.setAttribute("aria-invalid", "true");
                anchor.setAttribute("aria-describedby", note.id);
            }
        }
    }

    function setStatus(kind, html) {
        const status = byId("submit-status");
        status.className = `submit-result submit-result-${kind}`;
        status.innerHTML = html;
        status.hidden = false;
    }

    // ----- draft (this browser only) -----

    function saveDraft() {
        window.clearTimeout(saveTimer);
        saveTimer = window.setTimeout(() => {
            const fields = {};
            for (const element of form.elements) {
                if (!element.name) continue;
                fields[element.name] = element.type === "checkbox" ? element.checked
                    : element.type === "radio" ? (element.checked ? element.value : fields[element.name]) : element.value;
            }
            try {
                localStorage.setItem(DRAFT_KEY, JSON.stringify({ fields, tasks: [...state.tasks], tracks: [...state.tracks] }));
                byId("submit-draft-note").textContent = "Draft saved in this browser.";
            } catch (_error) {
                byId("submit-draft-note").textContent = "";
            }
        }, 400);
    }

    function restoreDraft() {
        let draft = null;
        try {
            draft = JSON.parse(localStorage.getItem(DRAFT_KEY) || "null");
        } catch (_error) {
            draft = null;
        }
        if (!draft) return;
        for (const [name, saved] of Object.entries(draft.fields || {})) {
            const element = field(name);
            if (!element || saved === undefined) continue;
            if (element instanceof RadioNodeList) element.value = saved;
            else if (element.type === "checkbox") element.checked = Boolean(saved);
            else element.value = saved;
        }
        if (Array.isArray(draft.tasks)) state.tasks = new Set(draft.tasks.filter(task => TASKS.includes(task)));
        if (Array.isArray(draft.tracks)) {
            state.tracks = new Set(draft.tracks.filter(track => TRACKS.some(([known]) => known === track)));
        }
        byId("submit-draft-note").textContent = "Restored your draft from this browser.";
    }

    function clearDraft() {
        try {
            localStorage.removeItem(DRAFT_KEY);
        } catch (_error) {
            // Nothing stored.
        }
    }

    // ----- actions -----

    function downloadFile() {
        const data = submission();
        const blob = new Blob([`${JSON.stringify(data, null, 2)}\n`], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `spatialarena_submission_${data.model.name || "model"}.json`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
    }

    async function send(event) {
        event.preventDefault();
        const errors = check();
        showErrors(errors);
        if (errors.length) {
            setStatus("error", `Fix the ${errors.length} highlighted field${errors.length === 1 ? "" : "s"} first.`);
            const first = errors[0].name === "tasks" ? byId("submit-tasks").querySelector("button") : field(errors[0].name);
            first?.focus?.();
            return;
        }
        if (EMAIL_MODE) {
            sendByEmail();
            return;
        }
        if (!SUBMIT_API) {
            setStatus("info", "The form is complete, but submissions cannot be sent yet: the evaluation backend is not "
                + "live. Download the submission file to keep it; you will be able to submit it once the site moves to "
                + "our own servers.");
            return;
        }
        const button = byId("submit-send");
        button.disabled = true;
        setStatus("info", "Sending…");
        try {
            const response = await fetch(`${SUBMIT_API}/submissions`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(submission()),
            });
            const reply = await response.json().catch(() => ({}));
            if (!response.ok || !reply.id) throw new Error(reply.message || `the server answered ${response.status}`);
            clearDraft();
            byId("submit-lookup-id").value = reply.id;
            setStatus("success", `Submitted. Your submission ID is <code>${escapeHtml(reply.id)}</code>; keep it to check the `
                + "status below. We also email the contact address.");
        } catch (error) {
            setStatus("error", `The submission was not sent: ${escapeHtml(error.message)}. Your draft is kept; try again later.`);
        } finally {
            button.disabled = false;
        }
    }

    // Saves the submission file and opens an email to SUBMIT_EMAIL carrying the submission (or, when it is too long
    // for a mailto: link, asking for the saved file as an attachment).
    // The email body when the whole file does not fit in a mailto: link: every field needed to start the evaluation,
    // so a forgotten attachment loses only the description and the notes.
    function emailSummary(data) {
        const env = data.inference.environment;
        const lines = [
            "Please evaluate this model on SpatialArena. The submission file "
                + `spatialarena_submission_${data.model.name}.json is attached; it also has the description and notes.`,
            "",
            `Model: ${data.model.name} (${data.model.display_name})`,
            `Code: ${data.model.code_url}${data.model.code_revision ? ` @ ${data.model.code_revision}` : ""}`,
            `Paper: ${data.model.paper_url || "-"}`,
            `Licenses: code ${data.model.license}, weights ${data.model.weights_license}`,
            `Evaluate: ${data.evaluation.pairs.map(pair => `${pair.task}/${pair.track}`).join(", ")}`
                + ` (${data.evaluation.scope === "spatialarena" ? "SpatialArena only" : "every dataset"})`,
            `Command: ${data.inference.command}`,
            `Checkpoint: ${data.inference.checkpoint_url || "fetched by the code"}`,
            `Environment: ${env.docker_image ? `docker ${env.docker_image}` : `Python ${env.python}; ${env.install}`}`,
            `GPU memory: up to ${data.inference.gpu_memory_gb} GB`,
            `Contact: ${data.contact.name} <${data.contact.email}>${data.contact.affiliation ? `, ${data.contact.affiliation}` : ""}`,
            `Results: ${data.visibility}`,
        ];
        return `${lines.join("\n")}\n`;
    }

    function sendByEmail() {
        const data = submission();
        downloadFile();
        const subject = `SpatialArena submission: ${data.model.name}`;
        const intro = "Please evaluate this model on SpatialArena. The submission file (spatialarena-submission/1) "
            + "follows.";
        let body = `${intro}\n\n${JSON.stringify(data)}\n`;
        let attach = false;
        const cc = SUBMIT_CC.length ? `&cc=${SUBMIT_CC.map(encodeURIComponent).join(",")}` : "";
        const link = text => `mailto:${SUBMIT_EMAIL}?subject=${encodeURIComponent(subject)}${cc}&body=${encodeURIComponent(text)}`;
        if (link(body).length > MAILTO_LIMIT) {
            attach = true;
            body = emailSummary(data);
        }
        window.location.href = link(body);
        const address = `<a href="mailto:${escapeHtml(SUBMIT_EMAIL)}">${escapeHtml(SUBMIT_EMAIL)}</a>`;
        const copied = SUBMIT_CC.length ? ` (copying ${SUBMIT_CC.map(escapeHtml).join(", ")})` : "";
        setStatus("success", `Your email app should open with the submission addressed to ${address}${copied}; send it, and we `
            + "reply to the contact address when the evaluation starts. The submission file was also saved"
            + (attach ? ": attach it to the email." : ".")
            + ` If no email app opened, email the saved file to ${address}${copied}.`);
    }

    async function lookup(event) {
        event.preventDefault();
        const id = byId("submit-lookup-id").value.trim();
        const out = byId("submit-lookup-result");
        out.hidden = false;
        if (!SUBMIT_API) {
            out.textContent = "Status lookup needs the evaluation backend, which is not live yet.";
            return;
        }
        if (!id) {
            out.textContent = "Enter the submission ID you received.";
            return;
        }
        out.textContent = "Checking…";
        try {
            const response = await fetch(`${SUBMIT_API}/submissions/${encodeURIComponent(id)}`);
            const reply = await response.json();
            if (!response.ok) throw new Error(reply.message || `the server answered ${response.status}`);
            const runs = Array.isArray(reply.runs) ? reply.runs : [];
            out.innerHTML = `<p><strong>${escapeHtml(reply.status || "unknown")}</strong>${reply.message ? ` · ${escapeHtml(reply.message)}` : ""}</p>`
                + (runs.length ? `<ul>${runs.map(run => `<li>${escapeHtml(run.task)} · ${escapeHtml(run.dataset)} · ${escapeHtml(run.track)}: ${escapeHtml(run.status)}</li>`).join("")}</ul>` : "");
        } catch (error) {
            out.textContent = `Could not get the status: ${error.message}.`;
        }
    }

    const escapeHtml = text => String(text ?? "").replace(/[&<>"']/g, char => (
        { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));

    function syncEnvironment() {
        const docker = value("env_kind") === "docker";
        byId("submit-env-python").hidden = docker;
        byId("submit-env-docker").hidden = !docker;
    }

    function refresh() {
        syncEnvironment();
        renderPicker();
        renderPreview();
    }

    function init() {
        form = byId("submit-form");
        if (!form) return;
        byId("submit-placeholders").innerHTML = PLACEHOLDERS.map(([placeholder, meaning]) => `
            <button type="button" class="submit-placeholder" data-submit-placeholder="${placeholder}"
                title="${meaning}" aria-label="Insert ${placeholder}: ${meaning}">${placeholder}</button>`).join("");
        byId("submit-use-template").addEventListener("click", () => {
            const command = field("command");
            command.value = TEMPLATE_COMMAND;
            command.dispatchEvent(new Event("input", { bubbles: true }));
            command.focus();
        });
        byId("submit-backend-notice").hidden = Boolean(SUBMIT_API);
        byId("submit-email-notice").hidden = !EMAIL_MODE;
        byId("submit-plain-notice").hidden = EMAIL_MODE;
        byId("submit-public-check").hidden = !EMAIL_MODE;
        if (EMAIL_MODE) {
            byId("submit-email-address").textContent = SUBMIT_EMAIL;
            byId("submit-email-address").href = `mailto:${SUBMIT_EMAIL}`;
            byId("submit-send").innerHTML = '<i class="fas fa-envelope" aria-hidden="true"></i> Submit by email';
        }
        byId("submit-lookup-button").disabled = !SUBMIT_API;
        restoreDraft();
        refresh();

        form.addEventListener("input", () => {
            renderPreview();
            saveDraft();
        });
        form.addEventListener("change", () => {
            refresh();
            saveDraft();
        });
        form.addEventListener("click", event => {
            const task = event.target.closest("[data-submit-task]")?.dataset.submitTask;
            const track = event.target.closest("[data-submit-track]")?.dataset.submitTrack;
            const placeholder = event.target.closest("[data-submit-placeholder]")?.dataset.submitPlaceholder;
            if (task || track) {
                const set = task ? state.tasks : state.tracks;
                const key = task || track;
                if (set.has(key)) set.delete(key); else set.add(key);
                refresh();
                saveDraft();
            } else if (placeholder) {
                const command = field("command");
                const start = command.selectionStart ?? command.value.length;
                const end = command.selectionEnd ?? command.value.length;
                const before = command.value.slice(0, start);
                const text = before && !/\s$/.test(before) ? ` ${placeholder}` : placeholder;
                command.setRangeText(text, start, end, "end");
                command.focus();
                renderPreview();
                saveDraft();
            }
        });
        form.addEventListener("submit", send);
        byId("submit-download").addEventListener("click", downloadFile);
        byId("submit-reset").addEventListener("click", () => {
            form.reset();
            state.tasks = new Set(TASKS);
            state.tracks = new Set(["rgb"]);
            clearDraft();
            showErrors([]);
            byId("submit-status").hidden = true;
            byId("submit-draft-note").textContent = "Form cleared.";
            refresh();
        });
        byId("submit-lookup").addEventListener("submit", lookup);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
