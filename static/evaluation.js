const THREAD_KEY = "blackwell-thread";
const REPORT_KEY = "blackwell-latest-report";
const EVALUATION_KEY = "blackwell-final-evaluation";
const TRIGGER_KEY = "blackwell-trigger-evaluation";

const evaluationOutput = document.getElementById("evaluation-output");

const threadId = window.sessionStorage.getItem(THREAD_KEY);
const latestReport = window.sessionStorage.getItem(REPORT_KEY);
const storedEvaluation = window.sessionStorage.getItem(EVALUATION_KEY);

function renderMarkdown(container, markdown) {
    const html = window.marked ? window.marked.parse(markdown) : `<pre>${markdown}</pre>`;
    container.innerHTML = `<div class="report-content">${html}</div>`;
}

function showPlaceholder(container, text) {
    container.innerHTML = `<p class="placeholder">${text}</p>`;
}

async function requestEvaluation() {
    const report = window.sessionStorage.getItem(REPORT_KEY);
    const sessionThread = window.sessionStorage.getItem(THREAD_KEY);

    if (!report) {
        showPlaceholder(evaluationOutput, "No anamnesis report found. Complete the interview first.");
        return;
    }

    if (!sessionThread) {
        showPlaceholder(evaluationOutput, "Missing session information. Return to the anamnesis page and restart the session.");
        return;
    }

    showPlaceholder(evaluationOutput, "Generating evaluation...");

    try {
        console.log("Sending evaluation request...");
        const response = await fetch("/api/evaluate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ report, thread_id: sessionThread }),
        });

        console.log("Response status:", response.status);

        if (!response.ok) {
            const errorText = await response.text();
            console.error("Evaluation error:", errorText);
            throw new Error(`Evaluation failed: ${response.status} - ${errorText}`);
        }

        const data = await response.json();
        renderMarkdown(evaluationOutput, data.evaluation);
        window.sessionStorage.setItem(EVALUATION_KEY, data.evaluation);
    } catch (error) {
        console.error("Evaluation error:", error);
        showPlaceholder(evaluationOutput, error.message || "Unexpected error while generating the evaluation.");
    }
}

const shouldTrigger = window.sessionStorage.getItem(TRIGGER_KEY) === "true";

if (!latestReport) {
    showPlaceholder(evaluationOutput, "No anamnesis report found. Return to anamnesis to complete the interview.");
} else if (!threadId) {
    showPlaceholder(evaluationOutput, "Session reference missing. Return to the anamnesis page and restart the session.");
} else if (shouldTrigger) {
    // Clear the trigger flag and generate new evaluation
    window.sessionStorage.removeItem(TRIGGER_KEY);
    requestEvaluation();
} else if (storedEvaluation) {
    // Show cached evaluation when navigating via nav-link
    renderMarkdown(evaluationOutput, storedEvaluation);
} else {
    // No cached evaluation and no trigger - show placeholder
    showPlaceholder(evaluationOutput, "No evaluation generated yet. Return to anamnesis and click 'Generate Evaluation' to start.");
}
