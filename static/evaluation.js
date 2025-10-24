const THREAD_KEY = "blackwell-thread";
const REPORT_KEY = "blackwell-latest-report";
const EVALUATION_KEY = "blackwell-final-evaluation";
const TRIGGER_KEY = "blackwell-trigger-evaluation";

const evaluationOutput = document.getElementById("evaluation-output");

const threadId = window.sessionStorage.getItem(THREAD_KEY);
const latestReport = window.sessionStorage.getItem(REPORT_KEY);
const storedEvaluation = window.sessionStorage.getItem(EVALUATION_KEY);

function renderMarkdown(container, markdown) {
    // Parse the evaluation report and add visual enhancements
    const enhancedHtml = parseEvaluationReport(markdown);
    container.innerHTML = `<div class="report-content evaluation-report">${enhancedHtml}</div>`;
}

function parseEvaluationReport(markdown) {
    let html = '';

    // Add report title
    html += `
        <h2 class="eval-report-title">
            <i class="fa-solid fa-file-medical"></i>
            Clinical Evaluation Report
        </h2>
    `;

    // Parse disclaimer
    const disclaimerMatch = markdown.match(/\*\*\*DISCLAIMER:([^*]+)\*\*\*/i);
    if (disclaimerMatch) {
        html += `
            <div class="eval-disclaimer">
                <i class="fa-solid fa-triangle-exclamation eval-disclaimer-icon"></i>
                <div class="eval-disclaimer-text">
                    <strong>DISCLAIMER:</strong> ${disclaimerMatch[1].trim()}
                </div>
            </div>
        `;
    }

    // Parse Probable Cause
    const probableCauseMatch = markdown.match(/###\s*\*?\*?1\.\s*Probable Cause\*?\*?\s*\n([^\n#]+)/i);
    if (probableCauseMatch) {
        const cause = probableCauseMatch[1].trim();
        html += `
            <div class="eval-section eval-section-probable">
                <div class="eval-section-header">
                    <div class="eval-section-icon">
                        <i class="fa-solid fa-bullseye"></i>
                    </div>
                    <h3 class="eval-section-title">Probable Cause</h3>
                </div>
                <div class="probable-cause-text">${cause}</div>
            </div>
        `;
    }

    // Parse Differential Diagnosis
    const differentialMatch = markdown.match(/###\s*\*?\*?2\.\s*Differential Diagnosis\*?\*?\s*\n([\s\S]*?)(?=###\s*\*?\*?3\.|$)/i);
    if (differentialMatch) {
        const differentialSection = differentialMatch[1];
        html += `
            <div class="eval-section eval-section-differential">
                <div class="eval-section-header">
                    <div class="eval-section-icon">
                        <i class="fa-solid fa-list-ol"></i>
                    </div>
                    <h3 class="eval-section-title">Differential Diagnosis</h3>
                </div>
                ${parseDifferentialDiagnoses(differentialSection)}
            </div>
        `;
    }

    // Parse Treatment Plan
    const treatmentMatch = markdown.match(/###\s*\*?\*?3\.\s*Suggested Treatment Plan\*?\*?\s*\n([\s\S]*?)$/i);
    if (treatmentMatch) {
        const treatmentSection = treatmentMatch[1];
        html += `
            <div class="eval-section eval-section-treatment">
                <div class="eval-section-header">
                    <div class="eval-section-icon">
                        <i class="fa-solid fa-notes-medical"></i>
                    </div>
                    <h3 class="eval-section-title">Suggested Treatment Plan</h3>
                </div>
                ${parseTreatmentPlan(treatmentSection)}
            </div>
        `;
    }

    return html;
}

function parseDifferentialDiagnoses(section) {
    const diagnosisPattern = /\*\s*\*?\*?(\d+)\.\s*([^(]+)\(Likelihood:\s*(High|Medium\/Low|Medium|Low)\)\*?\*?\s*\n([\s\S]*?)(?=\*\s*\*?\*?\d+\.|$)/gi;
    let html = '';
    let match;

    while ((match = diagnosisPattern.exec(section)) !== null) {
        const rank = match[1];
        const diagnosis = match[2].trim();
        const likelihood = match[3].trim();
        const content = match[4];

        // Parse justification
        const justificationMatch = content.match(/\*\s*\*?\*?Justification:\*?\*?\s*([^\n]*(?:\n(?!\*\s*\*?\*?)[^\n]*)*)/i);
        const justification = justificationMatch ? justificationMatch[1].trim() : '';

        html += `
            <div class="diagnosis-item">
                <div class="diagnosis-header">
                    <div style="display: flex; align-items: center;">
                        <span class="diagnosis-rank">${rank}</span>
                        <span class="diagnosis-title">${diagnosis}</span>
                    </div>
                    ${getProbabilityBadge(likelihood)}
                </div>
                ${justification ? `
                    <div class="justification-section">
                        <div class="justification-label">
                            <i class="fa-solid fa-clipboard-check"></i>
                            <span>Clinical Justification</span>
                        </div>
                        <div class="justification-text">${justification}</div>
                    </div>
                ` : ''}
            </div>
        `;
    }

    return html;
}

function getProbabilityBadge(likelihood) {
    const likelihoodLower = likelihood.toLowerCase();
    let badgeClass = 'probability-medium';
    let icon = 'fa-signal';
    let text = likelihood;

    if (likelihoodLower === 'high') {
        badgeClass = 'probability-high';
        icon = 'fa-arrow-up';
    } else if (likelihoodLower === 'medium/low') {
        badgeClass = 'probability-medium-low';
        icon = 'fa-equals';
    } else if (likelihoodLower === 'low') {
        badgeClass = 'probability-low';
        icon = 'fa-arrow-down';
    }

    return `
        <span class="probability-badge ${badgeClass}">
            <i class="fa-solid ${icon}"></i>
            ${text} Probability
        </span>
    `;
}

function parseTreatmentPlan(section) {
    let html = '';

    // Extract the diagnosis name from the treatment plan
    const diagnosisMatch = section.match(/for\s+\*?\*?([^*,]+)\*?\*?/i);
    if (diagnosisMatch) {
        html += `<p style="margin-bottom: 1.5rem; color: #4c5a7d; font-style: italic;">Treatment plan for <strong>${diagnosisMatch[1].trim()}</strong></p>`;
    }

    // Parse treatment categories
    const categories = [
        { 
            pattern: /\*\s*\*?\*?Pharmacological:\*?\*?\s*\n([\s\S]*?)(?=\*\s*\*?\*?Non-Pharmacological|\*\s*\*?\*?Follow-up|$)/i,
            title: 'Pharmacological Treatment',
            icon: 'fa-pills'
        },
        { 
            pattern: /\*\s*\*?\*?Non-Pharmacological\s*\/?\s*Lifestyle:\*?\*?\s*\n([\s\S]*?)(?=\*\s*\*?\*?Follow-up|$)/i,
            title: 'Non-Pharmacological & Lifestyle',
            icon: 'fa-heart-pulse'
        },
        { 
            pattern: /\*\s*\*?\*?Follow-up:\*?\*?\s*\n([\s\S]*?)$/i,
            title: 'Follow-up Care',
            icon: 'fa-calendar-check'
        }
    ];

    categories.forEach(category => {
        const match = category.pattern.exec(section);
        if (match && match[1]) {
            const content = match[1].trim()
                .replace(/\*\s*/g, '<li>')
                .replace(/<li>/g, '<li>')
                .replace(/\n\s*\n/g, '</li>');

            html += `
                <div class="treatment-category">
                    <div class="treatment-header">
                        <i class="fa-solid ${category.icon} treatment-icon"></i>
                        <h4 class="treatment-title">${category.title}</h4>
                    </div>
                    <div class="treatment-content">
                        ${content.includes('<li>') ? '<ul>' + content + '</li></ul>' : content}
                    </div>
                </div>
            `;
        }
    });

    return html;
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
