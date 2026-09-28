# 001 — 工具化（台阶 1）

| 项 | 值 |
|---|---|
| 时间 | 2026-09-27 ~ 2026-09-28 |
| 用例集 | 64 条 / 70 轮（`cases_hash = 88563922907ada5f`，三次 run 完全一致 → 逐格可比） |
| 基线 | `run_dir/c1fc63ab`（管道模式，2026-09-20，`code_hash=4e4556215f19826c`） |
| 工具化 v1 | `run_dir/0d78f59a`（2026-09-28 07:16，`code_hash=906926f268cfd430`） |
| 工具化 v2 | `run_dir/92359523`（2026-09-28 13:12，`code_hash=b59c1d0760fb743f`） |
| 结论 | **达成**：通过率 74% → **79%**，`ctx_recall` 0.70 → 0.85，全量成本 1.39× |

---

## 1. 假设

原来的链路是固定管道，没有 agent 的四个本质特征（模型自主决策、循环、可观测、可评测）：

```
用户问题 → [LLM#1 解析意图、选 knowledge_id] → [本地 BM25 选片段注入 system] → [LLM#2 流式作答]
```

**假设 H1**：把"选哪些知识"从 analyzer 的固定分支改成模型自主的工具调用，能让**证据质量**提升——因为模型可以按需下钻（先看目录、再定位文档、再取原文），而不是被一次性的片段投影卡在 6000 字符预算里。

**假设 H2**：基线 74% 的通过率里，有相当一部分是"**蒙对**"——模型凭参数记忆答对了，证据其实没进上下文。工具化会先**暴露**这一点（指标短期下降），再靠证据质量把它补回来。

验收口径（8 条，见 §6）在动手前就定好了，其中最关键的一条是：**`tools_enabled=false` 与基线逐格对齐**——但这条后来因为"没保留旧链路"而改为与录制基线对比。

---

## 2. 改动

### 2.1 结构

| 层 | 变化 |
|---|---|
| `backend/tool/` | **新增**：`dispatcher.py`（注册表 + 调度 + 参数校验 + schema 生成）、`tools_impl/knowledge.py`（4 个知识工具） |
| `backend/agent/orchestrator.py` | 删掉 analyzer 分支，改成"决策轮（带 tools）→ 执行工具 → 回灌 → 下一轮"的循环；`max_tool_rounds` 控制预算 |
| `backend/models.py` | `ChatMessage` 改成 `Annotated[Union[...], Field(discriminator="role")]` 类型别名；新增各 role 的消息类与 `ToolCall` |
| `backend/llm/client.py` | 新增 `stream_chat_with_tools`（流式累积 `tool_calls` 分片）；`_as_message` 显式挑字段 |
| `backend/observability/trace.py` | 新增 `finish_reason` / `tool_output` / `KIND_TOOL` 的实际使用；`cached_tokens` 进 span 摘要 |
| `eval/` | 证据口径改由工具事件推导；工具指标；基线对比 CLI；用例筛选 CLI |

### 2.2 工具集（4 个）

| 工具 | 作用 | 上游能力复用 |
|---|---|---|
| `read_knowledge_catalog` | 看有哪些分类 | `knowledge.parse_meta_file` |
| `read_knowledge_sub_catalog(folder)` | 看该分类下有哪些条目（拿到 doc_id） | `knowledge.build_sub_catalog` |
| `retrieval_in_document(doc_id, retrieval_texts[])` | 在一篇文档内取原文片段 | `retrieval.retrieve_chunks_in_doc` |
| `retrieval_among_document(doc_ids[], retrieval_text)` | 跨文档横向比较 | `retrieval.retrieve_chunks` |

设计成"目录 → 子目录 → 原文"的递进式，是为了**让模型自己确认该读哪一篇**，而不是猜 doc_id。

### 2.3 三个关键设计决定

**① `structured` 通道：dispatcher 保持领域无关**

工具返回 `ToolOutput(text, structured)`：`text` 给模型，`structured` 给观测/评测。

最初我把 `evidence_chunk_ids` 直接加在 `ToolCallResult` 上——**这是错的**：`chunk` 是知识库的领域概念，长在通用调度层上，将来接外部搜索就得再来个 `evidence_urls`、接数据库再来个 `evidence_row_ids`。改成的形状是 MCP `CallToolResult = content + structuredContent + isError` 的思路：

```python
@dataclass
class ToolOutput:
    text: str                        # 回灌模型
    structured: dict[str, Any] = {}  # dispatcher 只透传，不解释键名
```

于是**加新工具族时 dispatcher 零改动**：
```python
# 知识检索
ToolOutput(text=block, structured={"chunk_ids": [...]})
# 将来的外部搜索
ToolOutput(text=rendered, structured={"urls": [...]})
```
键名由工具自己声明，"chunk_ids → 文档级召回"这条领域知识留在评测层（它本来就是知识问答评测器）。

**② 工具消息落会话历史（最初我反对，后来改口）**

我一开始坚持"工具消息只活在 run 的工作副本里"。这个判断**在管道模式下成立**（那时有会话知识池提供跨轮证据），但 analyzer 删掉之后**池子就没有写入者了**——"不落历史"等于"跨轮完全没有证据"。

硬约束是：**`assistant(tool_calls)` 与它的 tool 消息组是不可分割的单元**（缺一条 provider 直接 400）。所以"不落 tool 消息"实际等价于"连 assistant(tool_calls) 也不能落"，那就只剩答案文本。粒度错了，不是方向错了。

现在的做法：**存储层保留完整 transcript 供跨轮复用；投影层的预算裁剪还没做**（记在 §7 待办）。

**③ 收口轮：不带 tools + 一条 user 提示**

终止保证是"轮次用尽的那一轮不带 `tools`"——模型无法再返回 `tool_calls`，循环必然收口。但实测发现这样会让模型**把工具调用协议当正文吐出来**（见 §4）。修法是加一条只存在于该次请求的 `UserMessage`：

```
系统提示：本轮工具调用次数已达上限。请基于已经获得的信息直接给出最终答案；
不要再请求调用工具，也不要输出工具调用的格式。
```

用 `UserMessage` 而不是 `SystemMessage`：非首条 system 消息在部分 OpenAI 兼容实现上不被接受；且它**只活在收口那一次请求**，不落历史（否则下一轮模型会以为用户说过这句话）。

### 2.4 先建回归网，再动生产代码

这一步是本迭代**最重要的一步**。`smoke_orchestrator.py` 重写成 8 个独立用例（逐用例收集失败、最后汇总，而不是 fail-fast），覆盖：事件序列、第一轮必须带 tools、收口轮不带 tools、消息配对不变式、工具失败仍是观察、参数校验失败、去重、LLM 失败契约、会话历史、trace 落盘。

它一次跑完就抓出了 **3 个让每一轮都必然失败的 P0**：

| # | 问题 | 症状 |
|---|---|---|
| 1 | `await client.stream_chat(...)` | 是 async generator，`await` 直接 `TypeError` |
| 2 | `AssistantMessage(reasoning_content=None)` | pydantic 校验失败（`LLMSpan.reasoning_content` 默认 None 表示"provider 没给"） |
| 3 | `tool_acc` 在流被消费**之前**读取 | generator 的函数体要等第一次 `__anext__` 才执行 → 永远读到空 dict → **模型要求调工具、编排器却当成直接作答**，工具链路静默失效（不报错） |

第 3 条最危险：不抛异常、不写日志，只表现为"每轮工具调用次数 = 0"，会让人去怀疑系统提示词或工具描述。

---

## 3. 指标：三轮对比

| 指标 | 基线 c1fc63ab | v1 0d78f59a | **v2 92359523** |
|---|---|---|---|
| **通过率** | 74% | 67% | **79%** |
| `doc_recall`（期望文档命中率） | 0.97 | 0.95 | 0.95 |
| `ctx_recall`（片段锚点命中率） | 0.70 | 0.89 | **0.85** |
| `contain_ok`（答案断言通过率） | 0.74 | 0.67 | **0.79** |
| LLM 调用 / 轮 | 2.00 | 4.27 | 4.57 |
| 工具调用 / 轮 | 0.00 | 3.51 | 3.90 |
| 工具返回字符 / 轮 | 0 | 27,102 | **15,053** |
| input tokens / 轮 | 3,613 | 38,829 | **27,675** |
| prompt cache 命中率 | 15.6% | 61.6% | **75.1%** |
| 全量成本（70 轮） | $1.43 | $2.88 | **$1.99** |
| 端到端 p50 / p95 | 6273 / 25166 | 6716 / 12754 | 7957 / 23532 |
| **协议泄漏轮数** | 0 | **6** | **0** |

工具侧（v2）：调用 246 次、失败率 **0%**、耗时占端到端 **~0%**（30ms/轮）、去重拦下 12 次。分布：`retrieval_in_document` 133、`read_knowledge_sub_catalog` 76、`read_knowledge_catalog` 61、`retrieval_among_document` 3。

**v2 归因矩阵**：

```
OK                                     50
文档对了但片段没取够（片段级召回）      7
证据齐但答案没用上（生成问题）          6   ← 下一阶段主攻
蒙对（最危险的绿）                      2
不该说却说了（命中禁词）                2
该检索却没检索（未调用检索工具）        1
答案内容不符（本轮无证据断言）          1
工具取错文档（选错 doc_id / 关键词）    1
```

### 结论

- **`ctx_recall` 0.70 → 0.85**，且"文档对了但片段没取够"从 **16 轮降到 7 轮**。这正是 H1 的账：递进式下钻让证据取全了。
- **`contain_ok` 0.74 → 0.79**，首次超过基线；通过率 74% → 79%。
- **H2 得到印证**：v1 阶段 10 轮"基线过、这次挂"里，有 8 轮的形态是 `doc=1.0 ctx=1.0` 却 contain 失败，而基线那几轮是 `ctx=0.0` 却 PASS——基线在**凭记忆蒙对**。换成"证据齐但答案没组织好"是可审计性提升过程中的正常代价，但它确实吃掉了账面分数。

---

## 4. 两个根因定位

### 4.1 协议泄漏：根因是"轮次不够"，不是模型坏

v1 有 6 轮答案里出现工具调用协议文本（长度 181~359 字符，根本不是答案），基线 0 轮。相关性是**完全**的：

| | 收口调用（`llm_calls=6`） | 其他所有调用 |
|---|---|---|
| 出现协议泄漏 | **6** | **0** |

14 次收口调用里 6 次（43%）出问题。而且 `39% 的轮次（27/70）用满了 ≥5 次调用`——模型走"目录→子目录→原文"本来就要 3 轮，`max_tool_rounds=5` 太紧。

修法（v2，两点一起做，泄漏降到 0）：
1. `max_tool_rounds: 5 → 8`
2. 收口轮带上有针对性的 `UserMessage`（明说"不要再请求调用工具，也不要输出工具调用的格式"）

v2 的 `llm_calls` 分布变健康了：`{4:26, 5:19, 6:9, 7:4, 8:3}`——大多数轮 4 次就收尾，只有 3 轮真正用满。

### 4.2 成本：账面 3.2×，实际 1.39×

报告里的"每轮成本均值"曾经是把**全部 input 按未命中价**算的上界。把 `cached_tokens` 接进来之后：

```
92359523:  in 1,937,269   cached 1,454,464 (75.1%)   -> $1.99
0d78f59a:  in 2,718,031   cached 1,673,088 (61.6%)   -> $2.88
c1fc63ab:  in   252,885   cached    39,419 (15.6%)   -> $1.43
```

**工具模式虽然 input 总量大，但前缀（system + 工具 schema + 历史）重复率高，缓存命中率 75%，而管道模式只有 15.6%**（analysis 与 generation 前缀不同，analyzer prompt 又短）。所以工具模式的边际成本远低于它的账面 token 量——真实成本是基线的 **1.39×**，不是 3.2×。

代价是模型/工具结构变了，但要警惕：**这个结论依赖"前缀稳定"**。一旦开始给 system 注入随轮次变化的临时提示（比如收口提示写在 system 里而不是末尾的 user），前缀缓存就会失效。收口提示放在**消息末尾**而非 system，除了 provider 兼容性，也是为这个考虑。

### 4.3 目录工具才是成本大头（一个被误诊的问题）

最初的诊断是"工具输出没有预算"。实测后发现是错的：

| 工具 | 每次返回字符 | 合计 | 占比 |
|---|---|---|---|
| **read_knowledge_sub_catalog** | **29538（恒定）** | 1,161,002 | **61%** |
| retrieval_in_document | p50 4949 / max 6733 | 596,645 | 31% |
| read_knowledge_catalog | 2069 | 126,209 | 7% |
| retrieval_among_document | ~6700 | 13,297 | 1% |

`retrieval_in_document` 的 max 6733 = `knowledge_max_chars`(6000) + 片段头/分隔符开销 —— **预算一直是生效的**。真正没预算的是目录工具：`job_skill/meta.md` 有 53 个条目，每条平均 557 字符（最长条目 keyword 275 + description 437），所以 `build_sub_catalog` 恒定返回 29538 字符，而模型只需要其中的 **doc_id**。

修法：压缩字段（keyword/description 是给 BM25 匹配用的，不该原样喂给模型），而不是粗暴截断（截断会把某些条目的 id 直接切掉，模型就拿不到 doc_id 了）。效果：sub_catalog 29538 → **p50 1137（−96%）**，全部工具输出 1,897,153 → **1,053,732 字符（−44%）**。

---

## 5. 评测侧的配套改动

工具化之后有几处评测语义必须跟着改，否则会拿到一份**读不出结论的报告**：

| 改动 | 不动会怎样 |
|---|---|
| `did_retrieve` 改由 `tool_call` 事件推导 | 旧口径读 `stage.name == "retrieval"`，该事件已不存在 → 恒为 False → 所有期望检索的轮 `retrieval_ok=False` → 通过率塌到 ~30%，归因矩阵清一色"检索选错文档"（指错方向） |
| `injected_chunk_ids` / `_texts` 从 `tool_result.structured.chunk_ids` 取 | `doc_recall` / `ctx_recall` 恒为 0.00 |
| `tool_result` 事件带 `structured`（只带 id，不带正文） | 评测无法知道"模型实际取到了哪些证据"；正文若进事件流会击穿 `<1500` 字符守卫 |
| span 摘要加 `cached_tokens` | 成本只能按未命中价算（上界），无法判断"成本可控" |
| `_span_cost` 用 `(input - cached) * 全价 + cached * 缓存价` | `prompt_tokens` **已包含**命中部分，写成 `input*全价 + cached*缓存价` 会把命中部分计两次——量级接近"全部按全价"，看起来"合理"，很难发现 |
| 基线对比 CLI：`python -m eval.metrics <run> [基线run]` | `render(baseline=...)` 参数写好了却从来没人传，两次 run 无法对比 |
| 用例筛选 CLI：`--case / --pattern / --limit` | 调一条用例也要跑 64 条（十几分钟、$2） |
| `SNAPSHOT_SETTINGS` 加 `max_tool_rounds` / `tool_timeout_s` | meta 解释不了两次 run 的行为差异 |
| `prompt_hash` 纳入**工具定义** | 工具 description 就是 prompt；漏掉它，改一次描述就无法解释历史分数 |

有一条踩过的坑值得单独写下来：**归因矩阵的每一格必须语义正确**。第一版 `_bucket` 里 3 轮 `neg-out-of-scope-*`（doc/ctx 都是 `None`）因为 `doc_ok = (None or ...) >= 1.0` 平凡为真，被归进了"证据齐但答案没用上（生成问题）"——实际成因是"该拒答却答了"。归因矩阵是决定下一步改什么的依据，**错一格就会指错方向**。修正后多了两格：`不该说却说了（命中禁词）`（`must_not_contain` 命中优先判定）和 `答案内容不符（本轮无证据期望）`（无证据层期望的轮，不参与证据归因）。

---

## 6. 验收清单

| # | 验收项 | 结论 |
|---|---|---|
| 1 | 回归不退化 | ✅ **79% vs 74%**（修好 6 轮、退化 3 轮，净 +3） |
| 2 | 工具真的被用 | ✅ 3.90 次/轮、66/70 轮调过工具、失败率 0% |
| 3 | 不必要调用少 | ✅ 4 个 `expect_no_retrieval` 用例都正确没调工具 |
| 4 | 目标指标改善 | ✅ `ctx_recall` +0.15、`contain_ok` +0.05 |
| 5 | 成本可控 | ✅ 实际 1.39×（账面 3.2×，差异来自 prompt cache） |
| 6 | 失败可归因 | ✅ 归因矩阵 + 【工具】块定位了协议泄漏与目录工具成本 |
| 7 | 离线可回归 | ✅ `smoke_orchestrator.py` 8/8（不联网、不要 API key） |
| 8 | 可解释 | ✅ meta 含 code/prompt/cases hash、价目快照、工具配置 |

**判定：达成，进入台阶 2。**

---

## 7. 遗留与下一步

### 遗留（已知、未做）

| 项 | 现状 | 影响 |
|---|---|---|
| 历史投影裁剪 | 工具消息无界落历史（存储层保留完整 transcript） | 多轮用例 token 是单轮的 3~5 倍；`multi_*` 类 token/轮 40k+ |
| 跨轮证据保留 | `【多轮跨轮证据】均值=0.50 最低=0.00` | 第 2 轮看不到第 1 轮的片段（只知道答案文本） |
| 工具输出统一上限 | `ToolCallRequest.max_output_chars` 仍未使用 | 单次工具输出最大 7658 字符（`retrieval_among_document`） |
| `tool_timeout_s` | 仍是 600s | 实际工具耗时 p50 40ms/轮；600s 等于没有超时 |
| 跨文档粗搜捷径 | `retrieval.retrieve_documents` 无人调用 | 导航占掉 50% 的工具调用（61+76=137 次 vs 原文 136 次） |

### 下一步（台阶 2：Agent 循环）

按"离答案最近"排序：

1. **6 轮"证据齐但答案没用上"**——`doc=1.0 ctx=1.0` 但 contain 失败，问题在答案有没有用上证据，属于生成层/自我校验，不是检索。
2. **多轮跨轮证据 0.50**——需要决定"证据要不要跨轮投影"（把 doc_ids 汇成池、按当前问题重投影）。
3. **跨文档粗搜工具 A/B**——`retrieve_documents`（BM25 跑在目录条目上）可以把"目录→子目录→原文"3 轮压到 1 轮，同时降成本、减少协议泄漏风险。**这是一个可测量的假设，不该靠论证定胜负**；判据是 `工具/轮`、`input tokens/轮`、泄漏轮数、通过率。

---

## 8. 如何复现

```powershell
# 离线回归（零成本、不联网）
.\.venv\Scripts\python.exe smoke_orchestrator.py

# 单条录制（调试期）
.\.venv\Scripts\python.exe -m eval.runner --case job-adele-hexa

# 全量录制
.\.venv\Scripts\python.exe -m eval.runner

# 打分 + 与基线逐格对比
.\.venv\Scripts\python.exe -m eval.metrics run_dir\<新run_id> run_dir\c1fc63ab
```

`meta.cases_hash` 不同时不要直接比总体均值——单条/子集录制的 `cases_hash` 必然不同，只有逐轮明细可比。
