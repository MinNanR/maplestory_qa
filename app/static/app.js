const askBtn = document.getElementById("askBtn");
const searchBtn = document.getElementById("searchBtn");
const reloadBtn = document.getElementById("reloadBtn");
const questionEl = document.getElementById("question");
const sourcesEl = document.getElementById("sources");
const entriesEl = document.getElementById("entries");
const chatMessagesEl = document.getElementById("chatMessages");

let currentController = null;
const chatHistory = [];

function escapeHtml(text) {
  return text
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
  const safe = escapeHtml(text);
  return marked.parse(safe, { breaks: true });
}

async function fetchJson(url, method = "GET", body = null) {
  const response = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : null,
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `request failed: ${response.status}`);
  }
  return response.json();
}

function renderSources(items) {
  if (!items.length) {
    sourcesEl.innerHTML = '<div class="card">没有命中资料</div>';
    return;
  }

  sourcesEl.innerHTML = items
    .map(
      (item) => `
        <article class="card">
          <h3>${item.title}</h3>
          <div class="meta">类型：${item.type} | 章节：${item.section} | 来源：${item.source}</div>
          <div class="tags">标签：${(item.tags || []).join(", ") || "无"}</div>
          <div class="snippet">${item.text.replace(/\n/g, "<br>")}</div>
        </article>
      `
    )
    .join("");
}

function renderEntries(items) {
  entriesEl.innerHTML = items
    .map(
      (item) => `
        <article class="card">
          <h3>${item.title}</h3>
          <div class="meta">类型：${item.type} | 来源：${item.source}</div>
          <div class="tags">标签：${(item.tags || []).join(", ") || "无"}</div>
        </article>
      `
    )
    .join("");
}

function appendMessage(role, content = "") {
  const node = document.createElement("div");
  node.className = `message ${role}`;
  if (role === "assistant") {
    node.innerHTML = renderMarkdown(content);
  } else {
    node.textContent = content;
  }
  chatMessagesEl.appendChild(node);
  chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
  return node;
}

function updateMessage(node, content) {
  if (node.classList.contains("assistant")) {
    node.innerHTML = renderMarkdown(content);
  } else {
    node.textContent = content;
  }
  chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
}

function normalizedHistory() {
  return chatHistory.slice(-8);
}

async function loadEntries() {
  const data = await fetchJson("/api/entries");
  renderEntries(data.items);
}

async function streamChat(question) {
  if (currentController) {
    currentController.abort();
  }

  sourcesEl.innerHTML = "";
  appendMessage("user", question);
  chatHistory.push({ role: "user", content: question });

  const assistantNode = appendMessage("assistant", "正在思考...");
  questionEl.value = "";

  const controller = new AbortController();
  currentController = controller;
  let answer = "";
  let started = false;

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        history: normalizedHistory().slice(0, -1),
      }),
      signal: controller.signal,
    });
    if (!response.ok || !response.body) {
      throw new Error(`request failed: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      while (buffer.includes("\n\n")) {
        const index = buffer.indexOf("\n\n");
        const rawEvent = buffer.slice(0, index);
        buffer = buffer.slice(index + 2);
        if (!rawEvent.startsWith("data: ")) continue;
        const payload = JSON.parse(rawEvent.slice(6));

        if (payload.type === "sources") {
          renderSources(payload.sources || []);
        } else if (payload.type === "chunk") {
          answer += payload.content;
          if (!started) {
            started = true;
            updateMessage(assistantNode, "");
          }
          updateMessage(assistantNode, answer);
        } else if (payload.type === "error") {
          updateMessage(assistantNode, `请求失败：${payload.content}`);
        }
      }
    }

    chatHistory.push({ role: "assistant", content: answer || "当前知识库里没有找到相关资料。" });
  } catch (error) {
    if (error.name === "AbortError") {
      updateMessage(assistantNode, "上一条回答已中断。");
    } else {
      updateMessage(assistantNode, `请求失败：${error.message}`);
    }
  } finally {
    currentController = null;
  }
}

askBtn.addEventListener("click", async () => {
  const question = questionEl.value.trim();
  if (!question) return;
  await streamChat(question);
});

questionEl.addEventListener("keydown", async (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    const question = questionEl.value.trim();
    if (!question) return;
    await streamChat(question);
  }
});

searchBtn.addEventListener("click", async () => {
  const question = questionEl.value.trim();
  if (!question) return;
  appendMessage("user", question);
  appendMessage("assistant", "已执行本地检索，请查看引用资料。");
  try {
    const data = await fetchJson("/api/search", "POST", { question });
    renderSources(data.items || []);
  } catch (error) {
    appendMessage("assistant", `请求失败：${error.message}`);
  }
});

reloadBtn.addEventListener("click", async () => {
  appendMessage("assistant", "正在重载知识库...");
  try {
    await fetchJson("/api/reload", "POST");
    await loadEntries();
    appendMessage("assistant", "知识库已重载。");
  } catch (error) {
    appendMessage("assistant", `重载失败：${error.message}`);
  }
});

appendMessage("assistant", "知识库已加载，可以开始提问。");
loadEntries();
