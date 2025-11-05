const THREAD_KEY = "blackwell-thread";
const REPORT_KEY = "blackwell-latest-report";
const EVALUATION_KEY = "blackwell-final-evaluation";

const chatPanel = document.getElementById("chat-panel");
const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const resetBtn = document.getElementById("reset-btn");
const reportOutput = document.getElementById("report-output");
const evaluationBtn = document.getElementById("evaluation-btn");

let threadId = window.sessionStorage.getItem(THREAD_KEY) || null;
let latestReport = window.sessionStorage.getItem(REPORT_KEY) || "";
let conversationLocked = Boolean(latestReport);

const placeholder = document.createElement("p");
placeholder.className = "placeholder";
placeholder.textContent = "Complete the anamnesis conversation to unlock the structured report.";

function disableEvaluationBtn() {
    evaluationBtn.disabled = true;
}

function enableEvaluationBtn() {
    evaluationBtn.disabled = false;
}

function setFormDisabled(isDisabled) {
    chatInput.disabled = isDisabled;
    sendBtn.disabled = isDisabled;
    if (isDisabled) {
        chatPanel.classList.add("completed");
    } else {
        chatPanel.classList.remove("completed");
    }
}

function appendMessage(role, content) {
    const messageEl = document.createElement("div");
    messageEl.className = `message ${role}`;
    const contentEl = document.createElement("div");
    contentEl.className = "message-text";
    contentEl.textContent = content;
    messageEl.appendChild(contentEl);
    chatLog.appendChild(messageEl);
    chatLog.scrollTop = chatLog.scrollHeight;
}

function renderMessages(messages) {
    chatLog.innerHTML = "";
    messages.forEach(({ role, content }) => {
        // Don't render the final anamnesis report in the chat
        if (role === "assistant" && content.includes("[ANAMNESIS REPORT]")) {
            return;
        }
        appendMessage(role, content);
    });
}

function renderReport(markdown) {
    // Remove [ANAMNESIS REPORT]: prefix if present
    const cleanedMarkdown = markdown.replace(/^\[ANAMNESIS REPORT\]:?\s*/i, '');
    
    // Parse the anamnesis report and add visual enhancements
    const enhancedHtml = parseAnamnesisReport(cleanedMarkdown);
    
    reportOutput.innerHTML = `<div class="report-content anamnesis-report">${enhancedHtml}</div>`;
}

function parseAnamnesisReport(markdown) {
    // Define section patterns with their icons and CSS classes
    const sections = [
        {
            pattern: /\*\*Chief Complaint \(CC\):\*\*\s*([^*]+?)(?=\*\*|$)/gi,
            title: 'Chief Complaint',
            icon: 'fa-solid fa-stethoscope',
            class: 'section-chief-complaint'
        },
        {
            pattern: /\*\*History of Present Illness \(HPI\):\*\*\s*([\s\S]*?)(?=\*\*Past Medical History|$)/gi,
            title: 'History of Present Illness',
            icon: 'fa-solid fa-notes-medical',
            class: 'section-hpi',
            hasSubsections: true
        },
        {
            pattern: /\*\*Past Medical History \(PMH\):\*\*\s*([\s\S]*?)(?=\*\*Medications and Allergies|$)/gi,
            title: 'Past Medical History',
            icon: 'fa-solid fa-clock-rotate-left',
            class: 'section-pmh'
        },
        {
            pattern: /\*\*Medications and Allergies:\*\*\s*([\s\S]*?)(?=\*\*Review of Systems|$)/gi,
            title: 'Medications and Allergies',
            icon: 'fa-solid fa-pills',
            class: 'section-medications',
            hasSubsections: true
        },
        {
            pattern: /\*\*Review of Systems \(ROS[^)]*\):\*\*\s*([\s\S]*?)$/gi,
            title: 'Review of Systems',
            icon: 'fa-solid fa-list-check',
            class: 'section-ros'
        }
    ];

    let html = '';

    sections.forEach(section => {
        const match = section.pattern.exec(markdown);
        if (match && match[1]) {
            const content = match[1].trim();
            html += `
                <div class="anamnesis-section ${section.class}">
                    <div class="section-header">
                        <i class="${section.icon} section-icon"></i>
                        <h3 class="section-title">${section.title}</h3>
                    </div>
                    <div class="section-content">
                        ${section.hasSubsections ? parseSubsections(content) : formatContent(content)}
                    </div>
                </div>
            `;
        }
        // Reset regex lastIndex for next iteration
        section.pattern.lastIndex = 0;
    });

    return html || `<div class="section-content">${formatContent(markdown)}</div>`;
}

function parseSubsections(content) {
    // Parse subsections like **Onset:**, **Location/Radiation:**, etc.
    const subsectionPattern = /\*\*([^*:]+):\*\*\s*([^*]+?)(?=\*\*|$)/gi;
    let html = '';
    let match;

    const iconMap = {
        'onset': 'fa-regular fa-calendar',
        'location': 'fa-solid fa-location-dot',
        'radiation': 'fa-solid fa-location-dot',
        'character': 'fa-solid fa-wand-magic-sparkles',
        'timing': 'fa-regular fa-clock',
        'triggers': 'fa-solid fa-bolt',
        'severity': 'fa-solid fa-temperature-high',
        'alleviating': 'fa-solid fa-hand-holding-medical',
        'associated': 'fa-solid fa-link',
        'medications': 'fa-solid fa-pills',
        'allergies': 'fa-solid fa-triangle-exclamation'
    };

    while ((match = subsectionPattern.exec(content)) !== null) {
        const title = match[1].trim();
        const text = match[2].trim();
        const key = title.toLowerCase().split('/')[0].split(' ')[0];
        const icon = iconMap[key] || 'fa-solid fa-circle-info';

        // Special handling for severity
        if (key === 'severity' && text.includes('/10')) {
            const severityMatch = text.match(/(\d+)\/10/);
            if (severityMatch) {
                html += `
                    <div class="subsection">
                        <div class="subsection-title">
                            <i class="${icon} subsection-icon"></i>
                            <span>${title}</span>
                        </div>
                        <div class="subsection-content">
                            <span class="severity-badge">
                                <i class="fa-solid fa-gauge-high"></i>
                                ${severityMatch[1]}/10 Severity
                            </span>
                            ${text.replace(/\d+\/10/, '').trim() ? ' - ' + text.replace(/Rated as \d+\/10[^.]*\.?/, '').trim() : ''}
                        </div>
                    </div>
                `;
                continue;
            }
        }

        html += `
            <div class="subsection">
                <div class="subsection-title">
                    <i class="${icon} subsection-icon"></i>
                    <span>${title}</span>
                </div>
                <div class="subsection-content">${text}</div>
            </div>
        `;
    }

    return html || formatContent(content);
}

function formatContent(text) {
    // Remove bold markdown and format basic text
    return text
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/\n\s*_\s*/g, '<br>')
        .replace(/\n/g, '<br>')
        .trim();
}

function resetReportView() {
    reportOutput.innerHTML = "";
    reportOutput.appendChild(placeholder.cloneNode(true));
}

function finalizeAnamnesis(report) {
    // Remove any text before the "[ANAMNESIS REPORT]" marker
    const reportMarkerIndex = report.indexOf("[ANAMNESIS REPORT]");
    if (reportMarkerIndex > 0) {
        report = report.substring(reportMarkerIndex);
    }
    conversationLocked = true;
    latestReport = report;
    window.sessionStorage.setItem(REPORT_KEY, report);
    setFormDisabled(true);
    renderReport(report);
    enableEvaluationBtn();
    chatInput.placeholder = "Anamnesis completed. Start a new session to continue.";
}

async function sendMessage(message) {
    const payload = { message, thread_id: threadId };
    const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });

    if (!response.ok) {
        throw new Error("Failed to reach the anamnesis agent.");
    }

    const data = await response.json();
    threadId = data.thread_id;
    window.sessionStorage.setItem(THREAD_KEY, threadId);

    renderMessages(data.messages);

    const lastMessage = data.messages.slice(-1)[0];
    if (data.finished && lastMessage) {
        finalizeAnamnesis(lastMessage.content);
    } else {
        conversationLocked = false;
        disableEvaluationBtn();
        window.sessionStorage.removeItem(REPORT_KEY);
        window.sessionStorage.removeItem(EVALUATION_KEY);
        resetReportView();
    }
}

async function loadConversationHistory() {
    if (!threadId) {
        return null;
    }

    try {
        const response = await fetch("/api/chat/history", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ thread_id: threadId }),
        });

        if (!response.ok) {
            return null;
        }

        const data = await response.json();
        return data.messages || [];
    } catch (error) {
        console.error("Failed to load conversation history:", error);
        return null;
    }
}

async function triggerEvaluation() {
    const report = window.sessionStorage.getItem(REPORT_KEY);
    const sessionThread = window.sessionStorage.getItem(THREAD_KEY);

    if (!report || !sessionThread) {
        alert("Cannot start evaluation. Please complete the anamnesis first.");
        return;
    }

    // Store a flag to trigger evaluation on the evaluation page
    window.sessionStorage.setItem("blackwell-trigger-evaluation", "true");
    
    // Navigate to evaluation page
    window.location.href = "/evaluation";
}

function resetSession() {
    conversationLocked = false;
    latestReport = "";
    threadId = null;
    window.sessionStorage.removeItem(THREAD_KEY);
    window.sessionStorage.removeItem(REPORT_KEY);
    window.sessionStorage.removeItem(EVALUATION_KEY);
    chatLog.innerHTML = "";
    resetReportView();
    disableEvaluationBtn();
    chatInput.value = "";
    chatInput.placeholder = "Describe the patient's condition...";
    setFormDisabled(false);
    chatInput.focus();
}

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (conversationLocked) {
        return;
    }

    const message = chatInput.value.trim();
    if (!message) {
        return;
    }

    setFormDisabled(true);
    appendMessage("user", message);
    chatInput.value = "";

    try {
        await sendMessage(message);
    } catch (error) {
        const errorEl = document.createElement("div");
        errorEl.className = "message system";
        errorEl.textContent = error.message || "Unexpected error.";
        chatLog.appendChild(errorEl);
        chatLog.scrollTop = chatLog.scrollHeight;
        setFormDisabled(false);
    } finally {
        if (!conversationLocked) {
            setFormDisabled(false);
            chatInput.focus();
        }
    }
});

resetBtn.addEventListener("click", resetSession);
evaluationBtn.addEventListener("click", triggerEvaluation);

async function initializePage() {
    resetReportView();
    latestReport = "Thank you for providing all these details. This gives the doctor a very clear picture of your chief complaint. [ANAMNESIS REPORT]: **Chief Complaint (CC):** Redness and irritation on the penis and groin, aggravated by friction (sexual activity/masturbation). **History of Present Illness (HPI):** * **Onset:** New condition, first time experiencing this. * **Location:** Penis and sometimes the groin, always in the same spots. Patient is circumcised. * **Character:** Primarily redness. When severe, line-shaped fissures (like cuts) appear, but they do not bleed. * **Severity:** Pain/burning rated up to 4/10 at its worst. Occasional itching. * **Aggravating/Alleviating Factors:** Symptoms worsen after masturbation or sexual intercourse (attributed to friction). Symptoms improve and skin heals when friction is avoided. * **Duration/Timing:** Condition comes and goes, healing when friction is avoided. **Relevant Past Medical History (PMH):** No prior history of this specific problem. No diagnosed medical conditions (e.g., diabetes, eczema, autoimmune disorders). **Medications and Allergies:** Not currently taking any medications (prescription, OTC, or supplements). No known allergies (medications, latex, soaps/detergents). **Social History (SH):** Patient reports not using condoms with current partner. **Review of Systems (ROS - Brief):** Denies fever, chills, unexplained weight changes, or changes in urination habits."
    if (threadId && !latestReport) {
        // Thread exists but no report stored - try to restore conversation
        const history = await loadConversationHistory();
        if (history && history.length > 0) {
            renderMessages(history);
            const lastMessage = history[history.length - 1];
            if (lastMessage.role === "assistant" && lastMessage.content.includes("[ANAMNESIS REPORT]")) {
                finalizeAnamnesis(lastMessage.content);
            } else {
                setFormDisabled(false);
            }
        } else {
            setFormDisabled(false);
        }
    } else if (latestReport) {
        // Report exists - restore finalized state
        const history = await loadConversationHistory();
        if (history && history.length > 0) {
            renderMessages(history);
        }
        finalizeAnamnesis(latestReport);
    } else {
        disableEvaluationBtn();
        setFormDisabled(false);
    }
}

initializePage();
