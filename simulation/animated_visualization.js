// Animated watermark visualization logic extracted to a separate module.
// Shows bias application: satisfied rules add +delta to the running score; unsatisfied rules add nothing.

const CONTEXT_WINDOW = [-2, -1, 1, 2];

const fallbackData = {
    text: "sample watermark demo",
    tokens: [
        { text: "sample", id: 0 },
        { text: "watermark", id: 1 },
        { text: "demo", id: 2 },
        { text: "runs", id: 3 },
        { text: "smoothly", id: 4 }
    ],
    steps: [
        {
            step: 1,
            token_id: 0,
            token_text: "sample",
            evaluations: [
                { pair: ["sample", "watermark"], satisfied: 1, context_token_id: 1 },
                { pair: ["sample", "demo"], satisfied: 0, context_token_id: 2 }
            ],
            total_satisfied: 1,
            confidence_boost: 2.0
        },
        {
            step: 2,
            token_id: 1,
            token_text: "watermark",
            evaluations: [
                { pair: ["sample", "watermark"], satisfied: 1, context_token_id: 0 },
                { pair: ["watermark", "demo"], satisfied: 1, context_token_id: 2 }
            ],
            total_satisfied: 2,
            confidence_boost: 4.0
        },
        {
            step: 3,
            token_id: 2,
            token_text: "demo",
            evaluations: [
                { pair: ["watermark", "demo"], satisfied: 0, context_token_id: 1 },
                { pair: ["demo", "runs"], satisfied: 1, context_token_id: 3 }
            ],
            total_satisfied: 1,
            confidence_boost: 2.0
        },
        {
            step: 4,
            token_id: 3,
            token_text: "runs",
            evaluations: [
                { pair: ["demo", "runs"], satisfied: 1, context_token_id: 2 },
                { pair: ["runs", "smoothly"], satisfied: 1, context_token_id: 4 }
            ],
            total_satisfied: 2,
            confidence_boost: 4.0
        },
        {
            step: 5,
            token_id: 4,
            token_text: "smoothly",
            evaluations: [
                { pair: ["runs", "smoothly"], satisfied: 1, context_token_id: 3 }
            ],
            total_satisfied: 1,
            confidence_boost: 2.0
        }
    ]
};

const BASE_MS = 1300;
let timer = null;
let isPlaying = false;
let stepIndex = -1;
let confidences = [];
let steps = [];
let tokens = [];
let tokenIndexById = new Map();
let visitedIndices = new Set();

const tokenGrid = d3.select("#token-grid");
const evalList = d3.select("#eval-list");
const bars = d3.select("#confidence-bars");
const status = d3.select("#status");
const progressFill = d3.select("#progress");
const stepTitle = d3.select("#step-title");
const stepNote = d3.select("#step-note");

const playBtn = document.getElementById("play-pause");
const resetBtn = document.getElementById("reset");
const nextBtn = document.getElementById("step-forward");
const prevBtn = document.getElementById("step-back");
const speedInput = document.getElementById("speed");
const speedLabel = document.getElementById("speed-label");

function softmax(values) {
    const expValues = values.map(v => Math.exp(v));
    const sum = expValues.reduce((a, b) => a + b, 0);
    return expValues.map(v => v / sum);
}

function formatStepLabel(idx) {
    if (idx < 0) return "Ready";
    return `Step ${idx + 1} of ${steps.length}`;
}

function pairDelta(step) {
    if (!step || !step.total_satisfied) return 0;
    return step.confidence_boost / step.total_satisfied;
}

function recomputeState(targetStep) {
    confidences = new Array(tokens.length).fill(0);
    visitedIndices = new Set();
    const capped = Math.min(targetStep, steps.length - 1);
    for (let i = 0; i <= capped; i++) {
        const step = steps[i];
        if (!step) continue;
        const idx = tokenIndexById.get(step.token_id);
        if (idx !== undefined) {
            confidences[idx] += step.confidence_boost;
            visitedIndices.add(idx);
        }
    }
}

function renderTokens() {
    const cards = tokenGrid.selectAll(".token-card").data(tokens, d => d.id);

    const enter = cards.enter().append("div").attr("class", "token-card");
    enter.append("div").attr("class", "label").text(d => d.text);
    enter.append("div").attr("class", "meta").text(d => `id ${d.id} ${d.is_masked ? '[M]' : '[U]'}`);
    enter.append("div").attr("class", "conf").text("conf: 0.00 | p: 0.000");
    
    // Add background styling for masked tokens
    enter.style("opacity", d => d.is_masked ? 1.0 : 0.7);

    cards.exit().remove();
}

function renderBars() {
    const rows = bars.selectAll(".bar-row").data(tokens, d => d.id);
    const enter = rows.enter().append("div").attr("class", "bar-row");
    enter.append("span").text(d => d.text);
    enter.append("div").attr("class", "bar-track").append("div").attr("class", "bar-fill");
    enter.append("span").attr("class", "bar-value").text("0.00");
    rows.exit().remove();
}

function updateBars() {
    const maxVal = Math.max(1, d3.max(confidences));
    bars.selectAll(".bar-row").each(function(d, i) {
        const val = confidences[i] || 0;
        const pct = (val / maxVal) * 100;
        d3.select(this).select(".bar-fill").style("width", `${pct}%`);
        d3.select(this).select(".bar-value").text(val.toFixed(2));
    });
}

function updateTokens() {
    const contextIds = new Set();
    const satisfiedIds = new Set();
    const notSatisfiedIds = new Set();
    const current = steps[stepIndex];
    const currentIdx = current ? tokenIndexById.get(current.token_id) : -1;

    if (current && Array.isArray(current.evaluations)) {
        current.evaluations.forEach(ev => {
            const idx = tokenIndexById.get(ev.context_token_id);
            if (idx === undefined) return;
            contextIds.add(idx);
            if (ev.satisfied) {
                satisfiedIds.add(idx);
            } else {
                notSatisfiedIds.add(idx);
            }
        });
    }

    const softmaxProbs = softmax(confidences);
    tokenGrid.selectAll(".token-card").each(function(d, i) {
        const card = d3.select(this);
        const raw = confidences[i] || 0;
        const prob = softmaxProbs[i] || 0;
        card.classed("current", i === currentIdx);
        card.classed("visited", visitedIndices.has(i));
        card.classed("context", contextIds.has(i));
        card.classed("satisfied", satisfiedIds.has(i));
        card.classed("not-satisfied", notSatisfiedIds.has(i));
        card.select(".conf").text(`conf: ${raw.toFixed(2)} | p: ${prob.toFixed(3)}`);
    });
}


function renderEvaluations() {
    evalList.selectAll(".eval-item").remove();
    const current = steps[stepIndex];
    if (!current) {
        stepTitle.text("Ready");
        stepNote.html("Press Play to begin. <strong>Note:</strong> First 2 and last 2 tokens skipped (context window bounds). Masked tokens shown with darker background.");
        return;
    }

    const idx = tokenIndexById.get(current.token_id);
    const prevConf = (idx !== undefined && stepIndex > 0) ? 
        (confidences[idx] - current.confidence_boost) : 0;
    const currConf = (idx !== undefined) ? confidences[idx] : 0;
    
    // Compute softmax probabilities
    const softmaxProbs = softmax(confidences);
    const prevSoftmaxProbs = softmax(confidences.map((c, i) => i === idx ? prevConf : c));
    const prevProb = idx !== undefined ? prevSoftmaxProbs[idx] : 0;
    const currProb = idx !== undefined ? softmaxProbs[idx] : 0;
    
    const DELTA = 2.0;
    const totalCount = current.total_count || 0;
    const numSatisfied = current.total_satisfied || 0;
    
    stepTitle.text(`Step ${stepIndex + 1}: Token "${current.token_text}"`);
    stepNote.html(`<div style="font-size: 15px; margin-bottom: 8px;"><strong>Context Window:</strong> {-2, -1, +1, +2}</div>
        <div style="font-size: 15px; margin-bottom: 8px;"><strong>Rule:</strong> If masked neighbor + satisfied: add neighbor's confidence. If unmasked neighbor + satisfied: add 1.0</div>
        <div style="font-size: 16px; background: #fff3cd; padding: 8px; border-left: 4px solid #ff9800;">
            <strong>Confidence Update:</strong><br/>
            ${prevConf.toFixed(2)} + (${totalCount.toFixed(2)} × δ) = ${prevConf.toFixed(2)} + ${current.confidence_boost.toFixed(2)} = <strong style="font-size: 18px;">${currConf.toFixed(2)}</strong>
        </div>
        <div style="font-size: 13px; margin-top: 6px; color: #666;">δ (delta) = ${DELTA.toFixed(1)} | Total contribution from satisfied pairs: ${totalCount.toFixed(2)}</div>
        <div style="font-size: 13px; margin-top: 4px;"><strong>Softmax Prob:</strong> ${prevProb.toFixed(4)} → <strong>${currProb.toFixed(4)}</strong></div>`);

    const rows = evalList.selectAll(".eval-item").data(current.evaluations);
    const enter = rows.enter().append("div").attr("class", "eval-item");
    enter.append("div").attr("class", d => d.satisfied ? "pill ok" : "pill no").text(d => d.satisfied ? "✓" : "×");
    enter.append("div").html(d => {
        const maskedLabel = d.is_masked ? "masked" : "unmasked";
        const contribution = d.contribution || 0;
        const formula = d.satisfied ? `+${contribution.toFixed(2)}` : `+0.0`;
        return `<strong style="font-size: 14px;">${d.pair[0]}</strong> ↔ <strong style="font-size: 14px;">${d.pair[1]}</strong> <span style="color: #666; font-size: 11px;">[${maskedLabel}]</span><br/>
                <span style="color: ${d.satisfied ? 'green' : 'red'}; font-weight: bold; font-size: 13px;">${d.satisfied ? '✓ SATISFIED' : '✗ NOT SATISFIED'}</span> <strong style="font-size: 14px;">${formula}</strong>`;
    });
    rows.exit().remove();
}

function updateProgress() {
    const pct = steps.length ? ((stepIndex + 1) / steps.length) * 100 : 0;
    progressFill.style("width", `${Math.max(0, Math.min(100, pct))}%`);
    status.text(formatStepLabel(stepIndex));
}

function goToStep(target) {
    const clamped = Math.max(-1, Math.min(target, steps.length - 1));
    stepIndex = clamped;
    recomputeState(stepIndex);
    updateTokens();
    updateBars();
    renderEvaluations();
    updateProgress();

    if (stepIndex >= steps.length - 1 && isPlaying) {
        setPlaying(false);
    }
}

function tick() {
    if (!isPlaying) return;
    goToStep(stepIndex + 1);
    const speed = parseFloat(speedInput.value || "1");
    timer = setTimeout(tick, BASE_MS / speed);
}

function setPlaying(state) {
    isPlaying = state;
    playBtn.textContent = state ? "Pause" : "Play";
    if (!state && timer) {
        clearTimeout(timer);
        timer = null;
    }
    if (state) {
        if (stepIndex >= steps.length - 1) {
            goToStep(-1);
        }
        if (timer) clearTimeout(timer);
        tick();
    }
}

function attachEvents() {
    playBtn.addEventListener("click", () => setPlaying(!isPlaying));
    resetBtn.addEventListener("click", () => { setPlaying(false); goToStep(-1); });
    nextBtn.addEventListener("click", () => { setPlaying(false); goToStep(stepIndex + 1); });
    prevBtn.addEventListener("click", () => { setPlaying(false); goToStep(stepIndex - 1); });
    speedInput.addEventListener("input", () => {
        speedLabel.textContent = `${parseFloat(speedInput.value).toFixed(2)}x`;
        if (isPlaying) {
            clearTimeout(timer);
            tick();
        }
    });
}

async function loadData() {
    try {
        const data = await d3.json("visualization_data.json");
        if (data && Array.isArray(data.tokens) && Array.isArray(data.steps)) {
            // Filter out first 2 and last 2 tokens - context window would go out of bounds
            const validSteps = data.steps.filter(s => {
                const idx = s.step - 1;
                return idx >= 2 && idx < data.tokens.length - 2;
            });
            return { ...data, steps: validSteps };
        }
    } catch (err) {
        console.warn("Falling back to built-in sample data", err);
    }
    status.text("Fallback sample loaded");
    return fallbackData;
}

async function main() {
    attachEvents();
    const data = await loadData();
    tokens = data.tokens || [];
    steps = data.steps || [];
    confidences = new Array(tokens.length).fill(0);
    tokenIndexById = new Map(tokens.map((t, idx) => [t.id, idx]));

    renderTokens();
    renderBars();
    updateTokens();
    updateBars();
    updateProgress();
    status.text("Ready");
}

main();
