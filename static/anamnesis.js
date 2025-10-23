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
        if (role === "assistant" && content.startsWith("[ANAMNESIS REPORT]")) {
            return;
        }
        appendMessage(role, content);
    });
}

function renderReport(markdown) {
    // Remove [ANAMNESIS REPORT]: prefix if present
    const cleanedMarkdown = markdown.replace(/^\[ANAMNESIS REPORT\]:?\s*/i, '');
    const html = window.marked ? window.marked.parse(cleanedMarkdown) : `<pre>${cleanedMarkdown}</pre>`;
    reportOutput.innerHTML = `<div class="report-content">${html}</div>`;
}

function resetReportView() {
    reportOutput.innerHTML = "";
    reportOutput.appendChild(placeholder.cloneNode(true));
}

function finalizeAnamnesis(report) {
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
    
    if (threadId && !latestReport) {
        // Thread exists but no report stored - try to restore conversation
        const history = await loadConversationHistory();
        if (history && history.length > 0) {
            renderMessages(history);
            const lastMessage = history[history.length - 1];
            if (lastMessage.role === "assistant" && lastMessage.content.startsWith("[ANAMNESIS REPORT]")) {
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
