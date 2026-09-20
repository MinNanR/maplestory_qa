const askBtn = document.getElementById("askBtn");
const searchBtn = document.getElementById("searchBtn");
const reloadBtn = document.getElementById("reloadBtn");
const newChatBtn = document.getElementById("newChatBtn");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");
const questionEl = document.getElementById("question");
const sourcesEl = document.getElementById("sources");
const entriesEl = document.getElementById("entries");
const historyListEl = document.getElementById("historyList");
const chatMessagesEl = document.getElementById("chatMessages");

const STORAGE_KEY = "maplestory_qa_chat_sessions";
const MAX_SESSIONS = 20;
const MAX_MESSAGES_PER_SESSION = 80;
const MAX_CONTEXT_MESSAGES = 8;

let chatSessions = [];
let activeSessionId = null;
let currentController = null;

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderMarkdown(text) {
  if (!window.marked) {
    return escapeHtml(text).replace(/\n/g, "<br>");
  }
  return marked.parse(escapeHtml(text), { breaks: true });
}

function createId() {
  if (window.crypto && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return Date.now().toString() + "-" + Math.random().toString(16).slice(2);
}

function formatTime(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function sessionTitle(session) {
  if (!session.messages.length) {
    return "New chat";
  }
  const firstUserMessage = session.messages.find(function (item) {
    return item.role === "user" && item.content.trim();
  });
  if (!firstUserMessage) {
    return "Conversation";
  }
  return firstUserMessage.content.replace(/\s+/g, " ").trim().slice(0, 24);
}

function normalizeSession(session) {
  if (!session || typeof session !== "object") return null;
  if (!Array.isArray(session.messages)) return null;

  const messages = session.messages
    .filter(function (item) {
      return item && typeof item === "object";
    })
    .map(function (item) {
      return {
        role: item.role,
        content: String(item.content || ""),
      };
    })
    .filter(function (item) {
      return (item.role === "user" || item.role === "assistant") && item.content.trim();
    });

  return {
    id: typeof session.id === "string" && session.id ? session.id : createId(),
    title: typeof session.title === "string" && session.title ? session.title : "New chat",
    createdAt: session.createdAt || new Date().toISOString(),
    updatedAt: session.updatedAt || new Date().toISOString(),
    messages: messages,
  };
}

function loadSessions() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      chatSessions = [];
      activeSessionId = null;
      return;
    }
    const parsed = JSON.parse(raw);
    chatSessions = Array.isArray(parsed)
      ? parsed.map(normalizeSession).filter(Boolean)
      : [];
    chatSessions.sort(function (a, b) {
      return new Date(b.updatedAt) - new Date(a.updatedAt);
    });
    activeSessionId = chatSessions[0] ? chatSessions[0].id : null;
  } catch {
    chatSessions = [];
    activeSessionId = null;
  }
}

function saveSessions() {
  chatSessions = chatSessions
    .map(normalizeSession)
    .filter(Boolean)
    .sort(function (a, b) {
      return new Date(b.updatedAt) - new Date(a.updatedAt);
    })
    .slice(0, MAX_SESSIONS);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(chatSessions));
}

function getActiveSession() {
  return chatSessions.find(function (session) {
    return session.id === activeSessionId;
  }) || null;
}

function touchSession(session) {
  session.updatedAt = new Date().toISOString();
  session.title = sessionTitle(session);
  chatSessions = [session].concat(chatSessions.filter(function (item) {
    return item.id !== session.id;
  }));
  saveSessions();
}

function createSession() {
  const session = {
    id: createId(),
    title: "New chat",
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    messages: [],
  };
  chatSessions = [session].concat(chatSessions);
  activeSessionId = session.id;
  saveSessions();
  renderHistoryList();
  return session;
}

function appendMessage(role, content) {
  const node = document.createElement("div");
  node.className = "message " + role;
  if (role === "assistant") {
    node.innerHTML = renderMarkdown(content || "");
  } else {
    node.textContent = content || "";
  }
  chatMessagesEl.appendChild(node);
  chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
  return node;
}

function updateMessage(node, content) {
  if (node.classList.contains("assistant")) {
    node.innerHTML = renderMarkdown(content || "");
  } else {
    node.textContent = content || "";
  }
  chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
}

function normalizedHistory(messages) {
  return messages.slice(-MAX_CONTEXT_MESSAGES);
}

function renderEntries(items) {
  entriesEl.innerHTML = items.map(function (item) {
    return [
      '<article class="card">',
      "<h3>" + escapeHtml(item.title) + "</h3>",
      '<div class="meta">Type: ' + escapeHtml(item.type) + " | Source: " + escapeHtml(item.source) + "</div>",
      '<div class="tags">Tags: ' + escapeHtml((item.tags || []).join(", ") || "none") + "</div>",
      "</article>",
    ].join("");
  }).join("");
}

function renderSources(items) {
  if (!items.length) {
    sourcesEl.innerHTML = '<div class="card">No source matched</div>';
    return;
  }

  sourcesEl.innerHTML = items.map(function (item) {
    return [
      '<article class="card">',
      "<h3>" + escapeHtml(item.title) + "</h3>",
      '<div class="meta">Type: ' + escapeHtml(item.type) + " | Section: " + escapeHtml(item.section) + " | Source: " + escapeHtml(item.source) + "</div>",
      '<div class="tags">Tags: ' + escapeHtml((item.tags || []).join(", ") || "none") + "</div>",
      '<div class="snippet">' + escapeHtml(item.text).replace(/\n/g, "<br>") + "</div>",
      "</article>",
    ].join("");
  }).join("");
}

function renderWelcome() {
  chatMessagesEl.innerHTML = "";
  appendMessage("assistant", "Knowledge base loaded. You can start asking questions.");
}

function renderSessionMessages(session) {
  chatMessagesEl.innerHTML = "";
  if (!session || !session.messages.length) {
    renderWelcome();
    return;
  }

  session.messages.forEach(function (message) {
    appendMessage(message.role, message.content);
  });
}

function renderHistoryList() {
  if (!historyListEl) {
    return;
  }

  if (!chatSessions.length) {
    historyListEl.innerHTML = '<div class="history-empty">No history yet</div>';
    return;
  }

  historyListEl.innerHTML = chatSessions.map(function (session) {
    return [
      '<button class="history-item ' + (session.id === activeSessionId ? "active" : "") + '" data-session-id="' + session.id + '">',
      '<div class="history-title">' + escapeHtml(session.title) + "</div>",
      '<div class="history-meta">' + formatTime(session.updatedAt) + " · " + session.messages.length + " messages</div>",
      "</button>",
    ].join("");
  }).join("");

  historyListEl.querySelectorAll(".history-item").forEach(function (button) {
    button.addEventListener("click", function () {
      activeSessionId = button.dataset.sessionId;
      renderSessionMessages(getActiveSession());
      renderHistoryList();
    });
  });
}

function startNewChat() {
  activeSessionId = null;
  questionEl.value = "";
  sourcesEl.innerHTML = "";
  renderWelcome();
  renderHistoryList();
}

function addMessageToSession(session, role, content) {
  session.messages.push({ role: role, content: content });
  if (session.messages.length > MAX_MESSAGES_PER_SESSION) {
    session.messages = session.messages.slice(-MAX_MESSAGES_PER_SESSION);
  }
  touchSession(session);
}

function fetchJson(url, method, body) {
  var options = {
    method: method || "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : null,
  };
  return fetch(url, options).then(function (response) {
    if (!response.ok) {
      return response.text().then(function (detail) {
        throw new Error(detail || ("request failed: " + response.status));
      });
    }
    return response.json();
  });
}

function loadEntries() {
  return fetchJson("/api/entries").then(function (data) {
    renderEntries(data.items || []);
  });
}

async function streamChat(question) {
  if (currentController) {
    currentController.abort();
  }

  const userQuestion = question.trim();
  if (!userQuestion) {
    return;
  }

  sourcesEl.innerHTML = "";

  let session = getActiveSession();
  if (!session) {
    session = createSession();
  }

  if (
    chatMessagesEl.childElementCount === 1 &&
    chatMessagesEl.firstElementChild &&
    chatMessagesEl.firstElementChild.textContent === "Knowledge base loaded. You can start asking questions."
  ) {
    chatMessagesEl.innerHTML = "";
  }

  appendMessage("user", userQuestion);
  addMessageToSession(session, "user", userQuestion);
  questionEl.value = "";

  const assistantNode = appendMessage("assistant", "Thinking...");
  const controller = new AbortController();
  currentController = controller;
  let answer = "";
  let responseError = "";
  let started = false;

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: userQuestion,
        history: normalizedHistory(session.messages).slice(0, -1),
      }),
      signal: controller.signal,
    });

    if (!response.ok || !response.body) {
      throw new Error("request failed: " + response.status);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const chunkResult = await reader.read();
      if (chunkResult.done) break;
      buffer += decoder.decode(chunkResult.value, { stream: true });

      while (buffer.indexOf("\n\n") !== -1) {
        const index = buffer.indexOf("\n\n");
        const rawEvent = buffer.slice(0, index);
        buffer = buffer.slice(index + 2);
        if (!rawEvent.startsWith("data: ")) continue;
        const payload = JSON.parse(rawEvent.slice(6));

        if (payload.type === "sources") {
          renderSources(payload.sources || []);
        } else if (payload.type === "chunk") {
          answer += payload.content || "";
          if (!started) {
            started = true;
            updateMessage(assistantNode, "");
          }
          updateMessage(assistantNode, answer);
        } else if (payload.type === "error") {
          responseError = payload.content || "request failed";
          updateMessage(assistantNode, "Request failed: " + responseError);
        }
      }
    }

    const finalAnswer = responseError ? ("Request failed: " + responseError) : (answer || "No relevant material found in the knowledge base.");
    updateMessage(assistantNode, finalAnswer);
    addMessageToSession(session, "assistant", finalAnswer);
  } catch (error) {
    if (error.name === "AbortError") {
      updateMessage(assistantNode, "The previous answer was interrupted.");
    } else {
      const finalAnswer = "Request failed: " + error.message;
      updateMessage(assistantNode, finalAnswer);
      addMessageToSession(session, "assistant", finalAnswer);
    }
  } finally {
    currentController = null;
    renderHistoryList();
  }
}

function clearHistory() {
  const confirmed = window.confirm("Clear all saved chat history?");
  if (!confirmed) {
    return;
  }

  chatSessions = [];
  activeSessionId = null;
  localStorage.removeItem(STORAGE_KEY);
  renderHistoryList();
  renderWelcome();
}

askBtn.addEventListener("click", function () {
  streamChat(questionEl.value);
});

questionEl.addEventListener("keydown", function (event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    streamChat(questionEl.value);
  }
});

searchBtn.addEventListener("click", function () {
  const question = questionEl.value.trim();
  if (!question) return;
  appendMessage("user", question);
  appendMessage("assistant", "Local search complete. Check the sources panel.");
  fetchJson("/api/search", "POST", { question: question })
    .then(function (data) {
      renderSources(data.items || []);
    })
    .catch(function (error) {
      appendMessage("assistant", "Request failed: " + error.message);
    });
});

reloadBtn.addEventListener("click", function () {
  appendMessage("assistant", "Reloading the knowledge base...");
  fetchJson("/api/reload", "POST")
    .then(function () {
      return loadEntries();
    })
    .then(function () {
      appendMessage("assistant", "Knowledge base reloaded.");
    })
    .catch(function (error) {
      appendMessage("assistant", "Reload failed: " + error.message);
    });
});

if (newChatBtn) {
  newChatBtn.addEventListener("click", startNewChat);
}

if (clearHistoryBtn) {
  clearHistoryBtn.addEventListener("click", clearHistory);
}

loadSessions();
renderHistoryList();
if (activeSessionId) {
  renderSessionMessages(getActiveSession());
} else {
  renderWelcome();
}
loadEntries();
