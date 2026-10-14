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
const stopBtn = document.getElementById("stop-btn");
const clearBtn = document.getElementById("clear-btn");
const sessionBadge = document.getElementById("session-badge");
const welcomeEl = document.getElementById("welcome");

let streaming = false;

// 本轮请求的取消句柄 + "是不是用户按了停止"。
// 停止的实现是 abort 掉 fetch：连接一断，服务端就会收到 CancelledError，
// 标记 interrupted、落 trace，并（在投影层）把半截的工具组整组丢弃 ——
// 所以前端不需要等任何"确认停止"的事件，本地立刻收尾即可。
let activeController = null;
let stoppedByUser = false;

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

/* ---------- 执行过程时间线（让 agent 循环看得见） ---------- */

/** 工具参数是 JSON 字符串：只截一小段给人看，长文本交给 CSS 省略号。 */
function shortArgs(args) {
    if (!args) return "";
    const text = typeof args === "string" ? args : JSON.stringify(args);
    return text.length > 80 ? text.slice(0, 80) + "…" : text;
}

/**
 * 步骤时间线：把 tool_call / tool_result / stage 三类事件渲染成
 * 「第 N 步 · 调用了什么 · 回来了多少」。
 *
 * 为什么挂在 bubble 上而不是 markdown-body 里：正文每次 token 都会整块重渲染
 * （body.innerHTML = ...），挂在里面会被反复清掉 —— 老代码的 hint 就得手动
 * 重新 append 才能活下来。
 *
 * 没有工具调用的普通回答**不会**显示这块（只有出现步骤行才 show）。
 */
function createTracker(bubble) {
    const wrap = document.createElement("div");
    wrap.className = "steps";
    wrap.hidden = true;

    const head = document.createElement("div");
    head.className = "steps-head";
    const title = document.createElement("span");
    title.className = "steps-title";
    title.textContent = "执行过程";
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "steps-toggle";
    toggle.textContent = "收起";
    toggle.addEventListener("click", () => {
        const collapsed = wrap.classList.toggle("collapsed");
        toggle.textContent = collapsed ? "展开" : "收起";
    });
    head.appendChild(title);
    head.appendChild(toggle);

    const list = document.createElement("div");
    list.className = "steps-list";

    const status = document.createElement("div");
    status.className = "steps-status";
    status.textContent = "思考中…";

    wrap.appendChild(head);
    wrap.appendChild(list);
    wrap.appendChild(status);
    bubble.insertBefore(wrap, bubble.firstChild);

    const rows = new Map();     // step -> { row, items }
    const pending = new Map();  // 工具名 -> [chip, ...]（LIFO：结果贴回最近一次调用）

    function show() {
        wrap.hidden = false;
    }

    function rowFor(step) {
        show();
        const key = step || 1;
        if (!rows.has(key)) {
            const row = document.createElement("div");
            row.className = "step-row";
            const no = document.createElement("span");
            no.className = "step-no";
            no.textContent = `第 ${key} 步`;
            const items = document.createElement("div");
            items.className = "step-items";
            row.appendChild(no);
            row.appendChild(items);
            list.appendChild(row);
            rows.set(key, { row, items });
        }
        return rows.get(key);
    }

    return {
        setStatus(text) {
            // 只更新文字，不 show：普通回答不该因为"生成中"就冒出时间线
            status.textContent = text;
        },
        /** 决策轮的过程独白：留着比丢掉好（丢掉要靠 final 覆盖，留着可复盘）。 */
        addThought(step, text) {
            const { items } = rowFor(step);
            const div = document.createElement("div");
            div.className = "step-thought";
            div.textContent = text;
            items.appendChild(div);
        },
        addToolCall(p) {
            const { items } = rowFor(p.step);
            const chip = document.createElement("div");
            chip.className = "tool-chip";
            const name = document.createElement("span");
            name.className = "tool-name";
            name.textContent = `🔍 ${p.tool || "工具"}`;
            const args = document.createElement("span");
            args.className = "tool-args";
            args.textContent = shortArgs(p.args);
            chip.appendChild(name);
            chip.appendChild(args);
            items.appendChild(chip);
            const key = p.tool || "";
            if (!pending.has(key)) pending.set(key, []);
            pending.get(key).push(chip);
        },
        addToolResult(p) {
            const queue = pending.get(p.tool || "") || [];
            const chip = queue.pop();
            if (!chip) return;
            const ids = (p.structured || {}).chunk_ids;
            const hits = Array.isArray(ids) && ids.length ? ` · 命中 ${ids.length} 片段` : "";
            const chars = p.chars ? ` · ${p.chars} 字符` : "";
            // succeeded 是后端显式给的（失败也要回灌一条 tool 消息）——
            // 不要去嗅探 result 的文案前缀
            const failed = p.succeeded === false;
            chip.classList.add(failed ? "failed" : "done");
            const meta = document.createElement("span");
            meta.className = "tool-meta";
            meta.textContent = failed ? "❌ 失败" : `✅${chars}${hits}`;
            chip.appendChild(meta);
        },
        summary() {
            let tools = 0;
            rows.forEach(({ items }) => {
                tools += items.querySelectorAll(".tool-chip").length;
            });
            return rows.size ? `共 ${rows.size} 步 · 调用工具 ${tools} 次` : "";
        },
        finish(note, collapse = true) {
            if (wrap.hidden) return;   // 没发生过工具调用 → 这块不该出现
            status.textContent = note || "完成";
            if (collapse) {
                wrap.classList.add("collapsed");
                toggle.textContent = "展开";
            }
            const s = this.summary();
            if (s) title.textContent = `执行过程 · ${s}`;
        },
    };
}

/* ---------- 流式请求 ---------- */

async function sendMessage(message) {
    addMessage("user", message);

    const el = addMessage("assistant");
    const body = el.body || el.bubble;
    const tracker = createTracker(el.bubble);

    const cursor = document.createElement("span");
    cursor.className = "stream-cursor";
    body.appendChild(cursor);

    streaming = true;
    stoppedByUser = false;
    sendBtn.disabled = true;
    stopBtn.hidden = false;

    const controller = new AbortController();
    activeController = controller;

    let full = "";            // 累积的 token，仅用于边收边渲染
    let finalText = "";       // final 事件里的权威完整答案
    let finishReason = null;  // 收口原因：no_tool_calls / steps_exhausted / ...
    let serverError = null;   // 后端通过 error 事件报的错（不是网络异常）
    let aborted = false;      // 后端发过 aborted（服务端侧中断）
    let gotFinal = false;

    try {
        if (!window.SSEStream) {
            throw new Error("sse.js 未加载，无法解析事件流");
        }

        const response = await fetch("/api/chat/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ conversation_id: conversationId, message }),
            signal: controller.signal,
        });

        if (!response.ok || !response.body) {
            throw new Error(`请求失败（HTTP ${response.status}）`);
        }

        await window.SSEStream.consumeSSE(response, (event) => {
            const payload = event.payload || {};

            switch (event.type) {
                case "stage":
                    // 现在 generation 阶段有显式起止了（以前只能靠 tool_call 的 end 猜）
                    if (payload.name === "generation") {
                        tracker.setStatus(payload.state === "start" ? "生成中…" : "本轮生成结束");
                    } else if (payload.name === "tool_call") {
                        tracker.setStatus(payload.state === "start" ? "执行工具…" : "工具执行完毕");
                    }
                    break;

                case "token":
                    full += payload.text || "";
                    body.innerHTML = renderMarkdown(full);
                    body.appendChild(cursor);
                    scrollToBottom();
                    break;

                case "tool_call":
                    // 决策轮的正文是"过程独白"（不是答案）。老做法是直接丢掉，
                    // 这里改进：收进时间线，正文清空（final 仍是权威答案）。
                    if (full.trim()) tracker.addThought(payload.step, full.trim());
                    full = "";
                    body.innerHTML = "";
                    body.appendChild(cursor);
                    tracker.addToolCall(payload);
                    tracker.setStatus(`正在调用 ${payload.tool || "工具"}…`);
                    break;

                case "tool_result":
                    tracker.addToolResult(payload);
                    tracker.setStatus(`${payload.tool || "工具"} 已返回`);
                    break;

                case "final":
                    // final 携带完整答案，是权威文本（不要只依赖 token 拼接）
                    finalText = payload.text || "";
                    finishReason = payload.finish_reason || null;
                    gotFinal = true;
                    break;

                case "aborted":
                    aborted = true;
                    break;

                case "error":
                    serverError = payload;
                    break;

                default:
                    // usage 等事件前端不消费（留给评测/观测）
                    break;
            }
        });

        cursor.remove();
        if (finalText) full = finalText;

        const stopped = stoppedByUser || aborted;

        if (serverError) {
            // stage 由后端带回来（turn.stage），映射成人看的阶段名；认不出就原样显示
            const stageLabel = STAGE_LABELS[serverError.stage] || serverError.stage;
            const where = stageLabel ? `（${stageLabel} 阶段）` : "";
            body.innerHTML = `<p>⚠️ 处理失败${escapeHtml(where)}：${escapeHtml(serverError.message || "未知错误")}</p>`;
            el.message.classList.add("error");
            tracker.finish("失败", false);   // 失败时展开过程，便于排查
        } else if (full.trim()) {
            body.innerHTML = renderMarkdown(full);
            if (stopped || !gotFinal) {
                // 明确告知"这是半截答案"，而不是静默接受
                const note = document.createElement("p");
                note.className = "stream-note";
                note.textContent = stopped
                    ? "（已停止生成，以上为已收到的部分）"
                    : "（连接中断，以上回答可能不完整）";
                body.appendChild(note);
            }
            attachCopyButtons(body);
            tracker.finish(
                stopped ? "已停止" : finishReason ? `收口原因：${finishReason}` : "完成"
            );
            scrollToBottom(true);
        } else if (stopped) {
            body.innerHTML = "<p>（已停止，尚未生成内容）</p>";
            tracker.finish("已停止");
        } else {
            body.innerHTML = "<p>（无回复内容）</p>";
            tracker.finish("无输出");
        }
    } catch (err) {
        cursor.remove();
        if (stoppedByUser || (err && err.name === "AbortError")) {
            // 用户主动停止：保留已收到的内容，绝不能报成"网络异常"
            if (finalText) full = finalText;
            if (full.trim()) {
                body.innerHTML = renderMarkdown(full);
                const note = document.createElement("p");
                note.className = "stream-note";
                note.textContent = "（已停止生成，以上为已收到的部分）";
                body.appendChild(note);
                attachCopyButtons(body);
            } else {
                body.innerHTML = "<p>（已停止，尚未生成内容）</p>";
            }
            tracker.finish("已停止");
        } else {
            body.innerHTML = `<p>⚠️ ${escapeHtml(err.message || "网络异常，请稍后重试")}</p>`;
            el.message.classList.add("error");
            tracker.finish("失败", false);
        }
    } finally {
        streaming = false;
        activeController = null;
        sendBtn.disabled = false;
        stopBtn.hidden = true;
        input.focus();
    }
}

/**
 * 停止生成：abort 掉 fetch。
 *
 * 不依赖后端再回一个"已停止"事件 —— 连接断了就送不到了。服务端的收尾
 * （标记 interrupted、落 trace、投影丢弃半截工具组）由断连本身触发。
 */
function stopStreaming() {
    if (!activeController) return;
    stoppedByUser = true;
    activeController.abort();
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
    // 正在流式时点"清空"：先把这一轮停掉，而不是静默无反应
    if (streaming) stopStreaming();

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
stopBtn.addEventListener("click", stopStreaming);

/* ---------- 初始化 ---------- */

updateSessionBadge();
input.focus();
