"use strict";

/* ============================================================
   MapleStory Knowledge Agent — 前端逻辑
   - 流式对话（后端 /api/chat/stream，SSE 事件流）
     协议解析在 sse.js（window.SSEStream），本文件只决定
     「每种事件怎么渲染」，不关心分帧/缓冲/多字节字符切分
   - Markdown 渲染（marked，本地 vendor）
   - 清空对话并生成新的 conversation_id
   ============================================================ */

// 当前会话 ID：清空对话时会重新生成
let conversationId = crypto.randomUUID();

/* 后端 turn.stage 的取值 -> 界面上人看的阶段名（error 事件里会带回来） */
const STAGE_LABELS = {
    init: "初始化",
    generation: "生成回答",
    tool_call: "调用工具",
};

const chat = document.getElementById("chat");
const form = document.getElementById("chat-form");
const input = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const clearBtn = document.getElementById("clear-btn");
const sessionBadge = document.getElementById("session-badge");
const welcomeEl = document.getElementById("welcome");

let streaming = false;

/* ---------- Markdown 配置 ---------- */
if (window.marked) {
    marked.setOptions({
        gfm: true,
        breaks: true,
        mangle: false,
        headerIds: false,
    });

    // 链接新窗口打开（marked 12 经典签名：link(href, title, text)）
    const renderer = new marked.Renderer();
    renderer.link = (href, title, text) => {
        const safeHref = (href || "").replace(/"/g, "&quot;");
        const safeTitle = (title || "").replace(/"/g, "&quot;");
        const titleAttr = safeTitle ? ` title="${safeTitle}"` : "";
        return `<a href="${safeHref}"${titleAttr} target="_blank" rel="noopener noreferrer">${text}</a>`;
    };
    marked.use({ renderer });
}

/* ---------- 工具函数 ---------- */

function escapeHtml(str) {
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function renderMarkdown(text) {
    if (window.marked) {
        try {
            return marked.parse(text || "");
        } catch (e) {
            return `<p>${escapeHtml(text || "")}</p>`;
        }
    }
    // 兜底：无 marked 时退化为纯文本
    return `<p>${escapeHtml(text || "")}</p>`;
}

function scrollToBottom(smooth = false) {
    chat.scrollTo({
        top: chat.scrollHeight,
        behavior: smooth ? "smooth" : "auto",
    });
}

function updateSessionBadge() {
    const short = conversationId.slice(0, 8);
    sessionBadge.textContent = `会话 ${short}`;
    sessionBadge.title = `当前会话 ID：${conversationId}`;
}

/* ---------- 消息渲染 ---------- */

function addMessage(role, content = "") {
    welcomeEl?.remove();

    const isUser = role === "user";
    const message = document.createElement("div");
    message.className = `message ${isUser ? "user" : "assistant"}`;

    const avatar = document.createElement("div");
    avatar.className = `avatar ${isUser ? "user-avatar" : "assistant-avatar"}`;
    avatar.textContent = isUser ? "你" : "🍁";

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    message.appendChild(isUser ? bubble : avatar);
    message.appendChild(isUser ? avatar : bubble);

    if (isUser) {
        // 用户消息按纯文本显示，避免注入
        bubble.textContent = content;
    } else {
        const body = document.createElement("div");
        body.className = "markdown-body";
        body.innerHTML = renderMarkdown(content);
        bubble.appendChild(body);
    }

    chat.appendChild(message);
    scrollToBottom();
    return { message, bubble, body: bubble.querySelector(".markdown-body") };
}

/* ---------- 流式请求 ---------- */

async function sendMessage(message) {
    addMessage("user", message);

    const el = addMessage("assistant");
    const body = el.body || el.bubble;

    const cursor = document.createElement("span");
    cursor.className = "stream-cursor";

    // 阶段提示：文本由后端的 stage / tool_call / tool_result 事件共同驱动：
    //   「思考中」→「🔍 正在调用 <工具>…」→「✅ 已返回…」→「正在生成回答…」
    const hint = document.createElement("span");
    hint.className = "thinking";
    const hintText = document.createElement("span");
    hintText.textContent = "思考中";
    const dots = document.createElement("span");
    dots.className = "dots";
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement("span");
        dot.className = "dot";
        dots.appendChild(dot);
    }
    hint.appendChild(hintText);
    hint.appendChild(dots);
    body.appendChild(hint);

    const setHint = (text) => {
        hintText.textContent = text || "思考中";
    };

    streaming = true;
    sendBtn.disabled = true;

    let full = "";            // 累积的 token，仅用于边收边渲染
    let finalText = "";       // final 事件里的权威完整答案
    let serverError = null;   // 后端通过 error 事件报的错（不是网络异常）
    let gotFinal = false;

    try {
        if (!window.SSEStream) {
            throw new Error("sse.js 未加载，无法解析事件流");
        }

        const response = await fetch("/api/chat/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ conversation_id: conversationId, message }),
        });

        if (!response.ok || !response.body) {
            throw new Error(`请求失败（HTTP ${response.status}）`);
        }

        await window.SSEStream.consumeSSE(response, (event) => {
            const payload = event.payload || {};

            switch (event.type) {
                case "stage":
                    // 后端只在工具调用前后发 stage（name="tool_call"，start/end）。
                    // start 那条要忽略：它和 tool_call 事件几乎同时到达，会用通用文案
                    // 盖掉刚设好的「🔍 正在调用 <工具名>…」，结果工具名从来来不及显示。
                    if (payload.name === "tool_call" && payload.state !== "end") break;
                    // end：转入生成。后端目前没有 generation 阶段的 stage，
                    // 这条 end 就是"工具跑完了，等模型作答"的信号。
                    setHint(payload.state === "end" ? "正在生成回答…" : payload.message || "思考中");
                    break;

                case "token":
                    full += payload.text || "";
                    hint.remove();
                    body.innerHTML = renderMarkdown(full);
                    body.appendChild(cursor);
                    scrollToBottom();
                    break;

                case "tool_call":
                    // 工具模式下**每一次** LLM 调用的正文都会走 token 事件，其中决策轮那段
                    // 是模型的"过程独白"（如"我先查一下知识库的目录结构。"），不是答案的一部分。
                    // 不清掉的话它会和最终答案粘成一段 —— 实测 71% 的轮次都有这段前导文本。
                    // 清掉是安全的：final 事件仍会用权威答案覆盖 full。
                    full = "";
                    body.innerHTML = "";
                    // hint 原本挂在 body 上，被 innerHTML 清掉后引用还在，重新挂上即可
                    if (!hint.parentNode) body.appendChild(hint);
                    setHint(`🔍 正在调用 ${payload.tool || "工具"}…`);
                    break;

                case "tool_result": {
                    // 工具回来了要给一条具体反馈：现在单次工具只要几十毫秒，无感；
                    // 但接了外部搜索（秒级）之后，没有反馈就会像卡住。
                    const ids = (payload.structured || {}).chunk_ids;
                    const hits = Array.isArray(ids) && ids.length ? `，命中 ${ids.length} 个片段` : "";
                    const chars = payload.chars ? `（${payload.chars} 字符）` : "";
                    setHint(`✅ ${payload.tool || "工具"} 已返回${chars}${hits}`);
                    break;
                }

                case "final":
                    // final 携带完整答案，是权威文本（不要只依赖 token 拼接）
                    finalText = payload.text || "";
                    gotFinal = true;
                    break;

                case "error":
                    serverError = payload;
                    break;

                default:
                    // usage 等事件前端不消费（留给评测/观测）
                    break;
            }
        });

        hint.remove();
        cursor.remove();

        if (finalText) full = finalText;

        if (serverError) {
            // stage 由后端带回来（turn.stage），映射成人看的阶段名；认不出就原样显示
            const stageLabel = STAGE_LABELS[serverError.stage] || serverError.stage;
            const where = stageLabel ? `（${stageLabel} 阶段）` : "";
            body.innerHTML = `<p>⚠️ 处理失败${escapeHtml(where)}：${escapeHtml(serverError.message || "未知错误")}</p>`;
            el.message.classList.add("error");
        } else if (full.trim()) {
            body.innerHTML = renderMarkdown(full);
            if (!gotFinal) {
                // 流在 final 之前就结束了：回答可能被截断，明确告知而不是静默接受
                const note = document.createElement("p");
                note.className = "stream-note";
                note.textContent = "（连接中断，以上回答可能不完整）";
                body.appendChild(note);
            }
            attachCopyButtons(body);
            scrollToBottom(true);
        } else {
            body.innerHTML = "<p>（无回复内容）</p>";
        }
    } catch (err) {
        hint.remove();
        cursor.remove();
        body.innerHTML = `<p>⚠️ ${escapeHtml(err.message || "网络异常，请稍后重试")}</p>`;
        el.message.classList.add("error");
    } finally {
        streaming = false;
        sendBtn.disabled = false;
        input.focus();
    }
}

/* ---------- 代码块复制按钮 ---------- */

function attachCopyButtons(container) {
    container.querySelectorAll("pre").forEach((pre) => {
        if (pre.querySelector(".code-copy-btn")) return;

        const codeEl = pre.querySelector("code");
        if (!codeEl) return;

        // 标记语言角标
        const cls = (codeEl.className || "").match(/language-(\S+)/);
        if (cls) {
            const lang = document.createElement("span");
            lang.className = "code-lang";
            lang.textContent = cls[1];
            pre.appendChild(lang);
        }

        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "code-copy-btn";
        btn.textContent = "复制";
        btn.addEventListener("click", async () => {
            try {
                await navigator.clipboard.writeText(codeEl.textContent || "");
                btn.textContent = "已复制 ✓";
                btn.classList.add("copied");
                setTimeout(() => {
                    btn.textContent = "复制";
                    btn.classList.remove("copied");
                }, 1500);
            } catch (e) {
                btn.textContent = "复制失败";
            }
        });
        pre.appendChild(btn);
    });
}

/* ---------- 清空对话 ---------- */

async function clearConversation() {
    if (streaming) return;

    const confirmed = window.confirm(
        "确定要清空当前对话吗？\n清空后将开启一个全新的会话。"
    );
    if (!confirmed) return;

    // 通知后端清理旧会话（可选，失败不影响前端）
    try {
        await fetch("/api/chat/clear", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ conversation_id: conversationId }),
        });
    } catch (e) {
        // 忽略：后端内存清理失败不影响新会话
    }

    // 清空界面并恢复欢迎页
    chat.innerHTML = "";
    chat.appendChild(welcomeEl);

    // 生成新的会话 ID
    conversationId = crypto.randomUUID();
    updateSessionBadge();
    input.focus();
}

/* ---------- 输入框自动增高 ---------- */

function autoResize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
}

input.addEventListener("input", autoResize);

/* ---------- 表单提交 ---------- */

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const message = input.value.trim();
    if (!message || streaming) return;

    input.value = "";
    autoResize();
    await sendMessage(message);
});

/* Enter 发送，Shift+Enter 换行 */
input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        form.requestSubmit();
    }
});

/* ---------- 欢迎页快捷提问 ---------- */

document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
        input.value = chip.textContent.trim();
        autoResize();
        input.focus();
    });
});

clearBtn.addEventListener("click", clearConversation);

/* ---------- 初始化 ---------- */

updateSessionBadge();
input.focus();
