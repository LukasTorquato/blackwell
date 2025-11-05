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
            Comprehensive Clinical Analysis Report
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

    // Parse sections in new format
    const sections = [
        {
            pattern: /###\s*\*?\*?1\.\s*Introduction and Patient Presentation \(Subjective\)\*?\*?\s*\n([\s\S]*?)(?=###\s*\*?\*?2\.|$)/i,
            title: 'Introduction and Patient Presentation',
            icon: 'fa-user-doctor',
            className: 'eval-section-introduction'
        },
        {
            pattern: /###\s*\*?\*?2\.\s*Clinical Findings and History Review\*?\*?\s*\n([\s\S]*?)(?=###\s*\*?\*?3\.|$)/i,
            title: 'Clinical Findings and History Review',
            icon: 'fa-clipboard-list',
            className: 'eval-section-findings'
        },
        {
            pattern: /###\s*\*?\*?3\.\s*Diagnostic Analysis and Differential\*?\*?\s*\n([\s\S]*?)(?=###\s*\*?\*?4\.|$)/i,
            title: 'Diagnostic Analysis and Differential',
            icon: 'fa-microscope',
            className: 'eval-section-differential',
            parser: parseDiagnosticAnalysis
        },
        {
            pattern: /###\s*\*?\*?4\.\s*Recommended Management Plan \(Plan\)\*?\*?\s*\n([\s\S]*?)(?=###\s*\*?\*?5\.|$)/i,
            title: 'Recommended Management Plan',
            icon: 'fa-notes-medical',
            className: 'eval-section-treatment',
            parser: parseManagementPlan
        },
        {
            pattern: /###\s*\*?\*?5\.\s*Concluding Summary\*?\*?\s*\n([\s\S]*?)$/i,
            title: 'Concluding Summary',
            icon: 'fa-circle-check',
            className: 'eval-section-summary'
        }
    ];

    sections.forEach(section => {
        const match = section.pattern.exec(markdown);
        if (match && match[1]) {
            const content = match[1].trim();
            html += `
                <div class="eval-section ${section.className}">
                    <div class="eval-section-header">
                        <div class="eval-section-icon">
                            <i class="fa-solid ${section.icon}"></i>
                        </div>
                        <h3 class="eval-section-title">${section.title}</h3>
                    </div>
                    ${section.parser ? section.parser(content) : formatTextContent(content)}
                </div>
            `;
        }
    });

    return html;
}

function formatTextContent(text) {
    // Convert markdown-style formatting to HTML
    let html = text
        // Bold text
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        // Italic text
        .replace(/\*([^*]+)\*/g, '<em>$1</em>')
        // Bullet points
        .replace(/^\*\s+(.+)$/gm, '<li>$1</li>')
        // Paragraphs
        .replace(/\n\n+/g, '</p><p>');

    // Wrap bullet points in ul tags
    html = html.replace(/(<li>.*<\/li>)/gs, (match) => {
        return '<ul>' + match + '</ul>';
    });

    // Wrap in paragraph tags if not already wrapped
    if (!html.startsWith('<ul>') && !html.startsWith('<p>')) {
        html = '<p>' + html + '</p>';
    }

    return `<div class="section-content">${html}</div>`;
}

function parseDiagnosticAnalysis(section) {
    let html = '';

    // Parse Probable Cause (without likelihood in parentheses)
    const probableCausePattern = /\*\*Probable Cause:\s*([^\n*]+)\*\*\s*\n([\s\S]*?)(?=\*\*Considered Differential|\*\*Diagnostic Uncertainty|$)/i;
    const probableCauseMatch = probableCausePattern.exec(section);
    
    if (probableCauseMatch) {
        const diagnosis = probableCauseMatch[1].trim();
        const content = probableCauseMatch[2].trim();

        // Extract rationale - look for bullet points or detailed rationale
        const rationaleMatch = content.match(/\*\s*\*?\*?Detailed Rationale:\*?\*?\s*([\s\S]*?)$/i);
        const rationale = rationaleMatch ? rationaleMatch[1].trim() : content;

        html += `
            <div class="diagnosis-item probable-cause-item">
                <div class="diagnosis-header">
                    <div style="display: flex; align-items: center;">
                        <i class="fa-solid fa-bullseye" style="margin-right: 0.5rem; color: #2c5aa0;"></i>
                        <span class="diagnosis-title">${diagnosis}</span>
                    </div>
                </div>
                <div class="justification-section">
                    <div class="justification-label">
                        <i class="fa-solid fa-clipboard-check"></i>
                        <span>Clinical Rationale</span>
                    </div>
                    ${formatTextContent(rationale)}
                </div>
            </div>
        `;
    }

    // Parse Considered Differential Diagnoses
    const differentialPattern = /\*\*Considered Differential Diagnoses:\*\*\s*\n([\s\S]*?)(?=\*\*Diagnostic Uncertainty|$)/i;
    const differentialMatch = differentialPattern.exec(section);
    
    if (differentialMatch) {
        const differentialSection = differentialMatch[1];
        // Match bullet points with diagnosis names
        const diagnosisPattern = /\*\s*\*?\*?([^:*]+?):\*?\*?\s*([\s\S]*?)(?=\n\s*\*\s*\*?\*?[A-Z][^:]+:|$)/gi;
        let match;

        html += '<div class="differential-diagnoses-section">';
        html += '<h4 style="margin: 1.5rem 0 1rem 0; color: #4c5a7d;"><i class="fa-solid fa-list-ol"></i> Differential Diagnoses</h4>';

        while ((match = diagnosisPattern.exec(differentialSection)) !== null) {
            const diagnosis = match[1].trim();
            const content = match[2].trim();

            html += `
                <div class="diagnosis-item">
                    <div class="diagnosis-header">
                        <span class="diagnosis-title">${diagnosis}</span>
                    </div>
                    <div class="justification-section">
                        ${formatTextContent(content)}
                    </div>
                </div>
            `;
        }

        html += '</div>';
    }

    // Parse Diagnostic Uncertainty (appears after differentials in your format)
    const uncertaintyPattern = /\*\*Diagnostic Uncertainty and (?:Further Steps|Clarification):\*\*\s*\n([\s\S]*?)$/i;
    const uncertaintyMatch = uncertaintyPattern.exec(section);
    
    if (uncertaintyMatch) {
        html += `
            <div class="uncertainty-section">
                <div class="uncertainty-header">
                    <i class="fa-solid fa-circle-exclamation"></i>
                    <span>Diagnostic Clarification</span>
                </div>
                ${formatTextContent(uncertaintyMatch[1].trim())}
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

function parseManagementPlan(section) {
    let html = '';

    // Parse Patient-Specific Considerations
    const considerationsPattern = /\*\*Patient-Specific Considerations:\*\*\s*\n([\s\S]*?)(?=\*\*\d+\.|$)/i;
    const considerationsMatch = considerationsPattern.exec(section);
    
    if (considerationsMatch) {
        html += `
            <div class="patient-considerations">
                <div class="considerations-header">
                    <i class="fa-solid fa-user-check"></i>
                    <span>Patient-Specific Considerations</span>
                </div>
                ${formatTextContent(considerationsMatch[1].trim())}
            </div>
        `;
    }

    // Parse numbered management steps
    const stepPattern = /\*\*(\d+)\.\s*([^:]+)(?::\*\*|\*\*:)\s*\n([\s\S]*?)(?=\*\*\d+\.|###\s*\*?\*?5\.|$)/gi;
    let match;

    while ((match = stepPattern.exec(section)) !== null) {
        const stepNumber = match[1];
        const stepTitle = match[2].trim();
        const stepContent = match[3].trim();

        // Determine icon based on title
        let icon = 'fa-notes-medical';
        if (stepTitle.toLowerCase().includes('diagnostic')) icon = 'fa-microscope';
        if (stepTitle.toLowerCase().includes('abortive') || stepTitle.toLowerCase().includes('acute')) icon = 'fa-syringe';
        if (stepTitle.toLowerCase().includes('preventative') || stepTitle.toLowerCase().includes('maintenance')) icon = 'fa-shield-heart';
        if (stepTitle.toLowerCase().includes('non-pharmacological') || stepTitle.toLowerCase().includes('lifestyle')) icon = 'fa-heart-pulse';

        html += `
            <div class="treatment-category">
                <div class="treatment-header">
                    <i class="fa-solid ${icon} treatment-icon"></i>
                    <h4 class="treatment-title">${stepNumber}. ${stepTitle}</h4>
                </div>
                <div class="treatment-content">
                    ${parseManagementStepContent(stepContent)}
                </div>
            </div>
        `;
    }

    return html;
}

function parseManagementStepContent(content) {
    let html = '';
    
    // Parse sub-items (treatments with rationales)
    const itemPattern = /\*\s*\*?\*?([^:]+):\*?\*?\s*([\s\S]*?)(?=\n\s*\*\s*\*?\*?Rationale:|\n\s*\*\s*\*?\*?[A-Z]|$)/gi;
    let match;
    let hasItems = false;

    let tempContent = content;
    
    while ((match = itemPattern.exec(content)) !== null) {
        hasItems = true;
        const itemTitle = match[1].trim();
        const itemContent = match[2].trim();

        // Check for rationale
        const rationalePattern = new RegExp(`\\*\\s*\\*?\\*?Rationale:\\*?\\*?\\s*([^\\n]*(?:\\n(?!\\*\\s*\\*?\\*?)[^\\n]*)*)`, 'i');
        const rationaleMatch = rationalePattern.exec(content.substring(match.index));
        const rationale = rationaleMatch ? rationaleMatch[1].trim() : '';

        html += `
            <div class="treatment-item">
                <div class="treatment-item-title">
                    <i class="fa-solid fa-circle-dot" style="font-size: 0.5rem; margin-right: 0.5rem;"></i>
                    ${itemTitle}
                </div>
                ${itemContent ? `<div class="treatment-item-content">${itemContent}</div>` : ''}
                ${rationale ? `
                    <div class="treatment-rationale">
                        <strong><i class="fa-solid fa-lightbulb"></i> Rationale:</strong> ${rationale}
                    </div>
                ` : ''}
            </div>
        `;
    }

    // If no structured items found, just format the text
    if (!hasItems) {
        html = formatTextContent(content);
    }

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
