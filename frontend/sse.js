"use strict";

/* ============================================================
   SSE 协议层（frontend/sse.js）
   ------------------------------------------------------------
   职责：把 text/event-stream 的字节流还原成「后端事件对象」。
   只做协议解析，不含任何 UI 逻辑（UI 决策在 app.js）。

   后端帧格式见 backend/main.py::encode_sse：
       event: <type>\n
       data: {"seq":n,"type":"<type>","payload":{...}}\n
       \n

   为什么必须缓冲：TCP 分片与 SSE 帧边界毫无关系。一次 read() 可能
   拿到半个帧、多个帧、或恰好一个帧；也可能把「中文的一个 UTF-8 字符」
   切成两半。所以：
     1) 字节层用 TextDecoder(..., {stream:true}) 增量解码，避免切碎多字节字符；
     2) 文本层攒到空行（\n\n）才算一帧解析。
   ============================================================ */

(function (root) {
    /** 创建增量 SSE 解析器：push(chunk) 返回本次能完整解析出的帧数组。 */
    function createSSEParser() {
        let buffer = "";

        function normalize() {
            // \r\n 与孤立的 \r 都视为换行；末尾孤立的 \r 可能是 \r\n 被切开，
            // 先留到下一块再处理（否则会把一个换行算成两个，破坏帧边界）。
            buffer = buffer.replace(/\r\n/g, "\n").replace(/\r(?!$)/g, "\n");
        }

        function parseFrame(text) {
            let event = "";
            const dataLines = [];
            for (const line of text.split("\n")) {
                if (!line || line.startsWith(":")) continue;   // 空行 / 注释行
                const idx = line.indexOf(":");
                const field = idx === -1 ? line : line.slice(0, idx);
                let value = idx === -1 ? "" : line.slice(idx + 1);
                if (value.startsWith(" ")) value = value.slice(1);   // 规范：冒号后仅一个空格不算数据
                if (field === "event") event = value;
                else if (field === "data") dataLines.push(value);
                // id / retry：本项目不使用，忽略
            }
            if (!event && !dataLines.length) return null;
            return { event: event, data: dataLines.join("\n") };
        }

        return {
            push: function (chunk) {
                buffer += chunk;
                normalize();
                const frames = [];
                let sep = buffer.indexOf("\n\n");
                while (sep !== -1) {
                    const raw = buffer.slice(0, sep);
                    buffer = buffer.slice(sep + 2);
                    const frame = parseFrame(raw);
                    if (frame) frames.push(frame);
                    sep = buffer.indexOf("\n\n");
                }
                return frames;
            },
            /** 流结束时调用：处理末尾没有以空行收尾的残留帧。 */
            flush: function () {
                const rest = buffer;
                buffer = "";
                if (!rest.trim()) return [];
                const frame = parseFrame(rest);
                return frame ? [frame] : [];
            },
        };
    }

    /** 把一帧解码成后端事件 {seq, type, payload}；解析失败返回 null。 */
    function decodeFrame(frame) {
        if (!frame || !frame.data) return null;
        let obj;
        try {
            obj = JSON.parse(frame.data);
        } catch (err) {
            return null;   // 脏帧：跳过，不影响后续帧
        }
        if (!obj || typeof obj !== "object") return null;
        return {
            seq: obj.seq,
            type: obj.type || frame.event,
            payload: obj.payload || {},
        };
    }

    /**
     * 消费一个 SSE 响应：逐块读取 → 解析 → 按事件回调（含 seq 乱序/重复的容忍由调用方决定）。
     * onEvent 里抛出的异常会终止读取，用于调用方主动中止。
     */
    async function consumeSSE(response, onEvent) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        const parser = createSSEParser();

        function emit(frames) {
            for (const frame of frames) {
                const event = decodeFrame(frame);
                if (event) onEvent(event);
            }
        }

        try {
            for (;;) {
                const result = await reader.read();
                if (result.done) break;
                emit(parser.push(decoder.decode(result.value, { stream: true })));
            }
            emit(parser.push(decoder.decode()));   // 冲掉 decoder 内部残留字节
            emit(parser.flush());
        } finally {
            if (typeof reader.releaseLock === "function") reader.releaseLock();
        }
    }

    const api = { createSSEParser: createSSEParser, decodeFrame: decodeFrame, consumeSSE: consumeSSE };

    if (typeof module !== "undefined" && module.exports) module.exports = api;   // 供 node 侧单测
    root.SSEStream = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
