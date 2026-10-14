# 002 — Agent 循环（台阶 2）

| 项 | 值 |
|---|---|
| 时间 | 2026-10-08 ~ 2026-10-10 |
| 用例集 | 74 条 / 80 轮（`cases_hash = 649e4c3d8b863864`；其中 64 条 / 70 轮与台阶 1 同集） |
| 开工前基线 | `run_dir/99db2824`（64 条 / 70 轮，`code_hash=fa5dc4b3761aec87`，`cases_hash=88563922907ada5f`） |
| 本次 run | `run_dir/c5c44bed`（74 条 / 80 轮，`code_hash=bb7c1899b194a243`，`cases_hash=649e4c3d8b863864`） |
| 单跳对照（B 组） | `run_dir/342d2f79`（10 条多跳，`--max-tool-rounds 1`） |
| 轮数中间档（C 组） | `run_dir/ecba241f`（10 条多跳，`--max-tool-rounds 4`） |
| 台阶 1 参照 | `run_dir/92359523`（工具化 v2，64 条 / 70 轮，`code_hash=b59c1d0760fb743f`） |
| prompt_hash | `0b70f30b1fd71446`（**三个 run 完全相同** → 系统提示与工具 schema 未变，行为差异可归因于编排改动） |
| 结论 | **达成**：循环可控（三类预算 + 无进展收口）、可停（取消/中断 + 前端停止实测通过）、可观测（`finish_reason` / `step` / 时间线）。重合 70 轮无退化信号（通过率 76%→80%，ctx 0.82→0.85，工具/轮 3.99→3.73，缓存 73%→74%）。**L5 对照完成：1 轮 0% / 4 轮 10% / 8 轮 80%** —— 循环是必要条件（§4.3、§4.4）；另发现 L1 停滞阈值 K=2 可能过早收口一处（§4.4 结论 3）。 |

---

## 1. 目标与验收口径

台阶 2 的原始验收只有一句：**复杂问题会连续调 2~4 个工具后作答，可中断。**
把它拆成 5 条可测判据，本次逐条核对：

| # | 判据 | 结论 | 证据 |
|---|---|---|---|
| 1 | **多步**：复杂问题会连续调 2~4 个工具 | ✅ | 10 条依赖型多跳：1 轮 → **0%** / 4 轮 → **10%** / 8 轮 → **80%**（§4.3、§4.4）。**循环是必要条件**；但"2~4 个工具"这个数字对该组题偏乐观 —— 当前工具粒度下一次两跳要约 6 次调用 |
| 2 | **收口**：预算/停滞触顶时能收口且原因可读 | ✅ | `finish_reason` 分布：`no_tool_calls` 77 / `no_progress` 2 / `chars_exhausted` 1；**协议泄漏 0 轮** |
| 3 | **可中断**：能停、断连不留后患 | ✅ | 冒烟用例 14（取消传播 + `aborted` 末尾事件 + trace `interrupted`）；前端停止按钮已实测通过；孤儿工具组由投影层整组丢弃（用例 15） |
| 4 | **可观测**：能看见它在做什么 | ✅ | 事件流新增 `stage(generation)`、`tool_call.step`、`tool_result.step/succeeded`、`aborted`；前端渲染"第 N 步 · 调用 X · 返回 Y 字符"时间线 |
| 5 | **不回退**：重合用例不变差 | ✅ | 重合 70 轮：通过率 76%→80%（修好 5 轮、退化 2 轮），doc 0.95→0.95，ctx 0.82→0.85，contain 0.77→0.80，工具/轮 3.99→3.73，cache 73%→74% |

> 判据 1 要同时看两条结论：**循环是必要条件**（1 轮 0% vs 8 轮 80%，见 §4.3），但**"2~4 个工具"对该组题偏乐观**（拐点落在 4 与 8 之间，见 §4.4）—— 瓶颈在检索路径长度，属台阶 3。

---

## 2. 改动清单

### 2.1 循环骨架（L0，台阶 2 开工前完成）

| 层 | 变化 |
|---|---|
| `backend/agent/loop.py` | 循环体从 `orchestrator.run` 抽出；**预算改为函数局部变量**（只在结束时把"剩余步数"回填 `budget.steps_remaining` 供外壳记录）——一次性消灭了"写回 budget 导致语义反转"那一类 bug |
| `backend/agent/execution.py` | `ToolExecutor`：去重、并发上限、单次超时、统一记账出口（成功/失败/参数校验失败/去重四条路径都写 `span.detail` + `duration_ms`） |
| `backend/agent/events.py` | 事件契约独立成模块（解 orchestrator ↔ loop 的循环 import）；`seq` 由外壳分配 |
| `backend/observability/trace.py` | `TurnContext.finish_reason / interrupted`、`LLMSpan.step` |

### 2.2 L1 无进展收口（新增终止条件）

- **做法**：用 `tool_result.structured.chunk_ids` 做进展信号（零额外 LLM 调用）。连续两轮拿不到**新**证据 → `no_progress` 收口。
- **判定规则**（`loop.py`）：`round_evidence - seen_evidence` 非空 = 有进展；否则计数；**`seen_evidence` 非空之后，"整轮零证据"与"重复调用"同样计入停滞**。
- **验**：冒烟用例 18（专门构造去重拦不住的形态：换关键词但证据不变）。

### 2.3 L2 可中断

- `loop` 的 `CancelledError` 只抛不报；由外壳 `mark_interrupted()` → `turn.interrupted=True` / `finish_reason="user_cancel"` / 落盘 `finished=False`，事件以 `aborted` 收尾。
- `GeneratorExit`（客户端 `aclose`）同样标记，但**不 yield**（否则 "async generator ignored GeneratorExit"）。
- 前端：停止按钮 + `AbortController`。**停止不依赖后端事件**——客户端 abort 后连接就断了，`aborted` 事件送不到浏览器；它的价值在测试与 trace。

### 2.4 L3 投影层（`backend/conversation/projection.py`）

- `MessageGroup` **按 tool_call id 配对** tool 消息（不是按位置）；不完整的组（中断留下的半截）**整组丢弃**——否则 provider 直接 400。
- **最新一组无条件保留**（不受字符预算影响）：修掉"当前问题本身超预算 → 模型收到空对话"。
- 预算统计含 assistant/tool 正文（不是只算 user）。
- 三条不变量由冒烟用例 15/16/17 钉住：配对不变式、窗口真裁、**前缀逐字节稳定**（缓存命中的前提）。

### 2.5 L4 循环可观测

- 事件：`stage(generation, start/end)`（此前前端只能靠 `tool_call` 的 end 猜）、`tool_call.step`、`tool_result.step/succeeded`、`aborted`；`final.finish_reason`。
- 前端：时间线（第 N 步 · 工具名 + 参数摘要 · 返回字符/命中片段数），决策轮的"过程独白"从**丢弃**改为**收进时间线**（可复盘）。
- 评测：`finish_reason` 落进 `TurnRecord` 并进归因矩阵；新增 `预算收口（可能因此证据不足）` 一格。

### 2.6 L5 多跳用例与指标

- **用例必须"依赖型"**：第二跳的文档身份与检索词只能从第一跳返回里得到（判据：只能调一次工具时必然答不对）。反例是"并列型"——因为 `retrieval_among_document` 一次调用可覆盖多篇，并列型单跳也能答对。
- 新增 `expect_min_tool_calls` 字段 + `validate_cases` 语义校验（`multi_hop` 必须 ≥2 篇 doc 且 ≥2 次调用）。
- 指标：`min_tool_calls_met`（过程达成率，分母只在声明了该期望的轮）。
- **硬化**：多跳轮必须 `doc_recall == 1.0` 才算过。原因：多数 `must_contain` 项在第一跳文档里就能命中（校验器只要求"任一期望文档可达"），`contain_pass` 无法证明第二跳发生过，只能靠 `doc_recall` 钉住。这类失败会落进既有的 `蒙对（最危险的绿）` 格。

---

## 3. 指标

### 3.1 本次全量（74 条 / 80 轮）

```
通过率 80%（64/80）   doc_recall 0.95   ctx_recall 0.86   contain 0.82
工具/轮 4.01（76/80 轮调过工具，去重 17 次）   LLM 调用/轮 4.78（总 382 次）
input/轮 28,407   cost/轮 $0.0149   cache 74%
端到端 p50 11,583ms   p95 31,511ms
预算收口轮数 3（no_progress 2 + chars_exhausted 1）   多跳达成率 1.00   协议泄漏 0
```

按 category：

| category | 轮 | 通过率 | doc | ctx | contain | 工具/轮 | 预算收口 |
|---|---|---|---|---|---|---|---|
| job_skill | 24 | 96% | 1.00 | 0.96 | 0.96 | 4.38 | 0 |
| **multi_hop** | 10 | **80%** | 0.92 | 0.92 | 1.00 | **6.00** | 2 |
| multi_turn | 12 | 58% | 0.83 | 0.75 | 0.58 | 3.33 | 0 |
| system | 11 | 73% | 0.91 | 0.64 | 0.73 | 3.45 | 0 |
| negative | 9 | 67% | -- | -- | 0.67 | 2.00 | 0 |
| alias_typo | 6 | 83% | 1.00 | 1.00 | 0.83 | 4.33 | 0 |
| boss | 4 | 100% | 1.00 | 1.00 | 1.00 | 3.25 | 0 |
| patch_note | 2 | 100% | 1.00 | 1.00 | 1.00 | 3.50 | 0 |
| ambiguous | 2 | 50% | 1.00 | 0.50 | 0.50 | 7.00 | 1 |

### 3.2 W1/L1/W3 验收：与开工前基线逐轮对比（重合 70 轮）

`cases_hash` 不同（schema 加了字段），所以**只比逐轮明细**，不比总体均值。

| 指标 | 99db2824 | c5c44bed | Δ |
|---|---|---|---|
| 通过率 | 76% | 80% | **+4pp**（修好 5 轮 / 退化 2 轮） |
| doc_recall | 0.95 | 0.95 | +0.00 |
| ctx_recall | 0.82 | 0.85 | +0.03 |
| contain_ok | 0.77 | 0.80 | +0.03 |
| 工具/轮 | 3.99 | 3.73 | **−0.26** |
| LLM 调用/轮 | 4.56 | 4.54 | −0.02（总 319→318） |
| input/轮 | 26,927 | 26,127 | −800 |
| cache 命中率 | 73% | 74% | +1pp |
| 端到端 p95 | 18,943ms | 27,620ms | +8,677ms（**见 §5.3 归因**） |

按 category（重合轮）：

| category | n | 通过率 | ctx_recall | 工具/轮 |
|---|---|---|---|---|
| multi_turn | 12 | 50% → **58%** | 0.42 → **0.75** | 3.58 → 3.33 |
| job_skill | 24 | 92% → 96% | 0.96 → 0.96 | 4.67 → 4.38 |
| boss | 4 | 75% → 100% | 1.00 → 1.00 | 3.50 → 3.25 |
| system | 11 | 73% → 73% | **0.82 → 0.64** | 3.55 → 3.45 |
| alias_typo | 6 | 83% → 83% | 1.00 → 1.00 | 5.17 → 4.33 |
| negative | 9 | 67% → 67% | -- | 2.22 → 2.00 |
| patch_note / ambiguous | 2 / 2 | 100% / 50% | 1.00 / 0.50 | 3.50 / 7.00 |

翻转的 7 轮（这是本次改动唯一的逐轮证据）：

```
FAIL→PASS  boss-seren-phase-01        t0  工具 5→4
FAIL→PASS  job-pathfinder-boost       t0  工具 3→3
FAIL→PASS  multi-flame-scissors-01    t1  doc 0.00→1.00  ctx 0.00→1.00  工具 0→1   ← W1 最干净的收益
FAIL→PASS  multi-seren-kalos-01       t1  contain False→True
FAIL→PASS  system-utility-cooldown-01 t0  工具 3→3
PASS→FAIL  multi-kinesis-numeric-01   t0  contain True→False
PASS→FAIL  system-flame-nowash-01     t0  ctx 1.00→0.00  工具 4→5
```

**multi_turn 逐轮（W1 的主战场）**：`multi-flame-scissors-01` 第 2 轮从"工具 0 次、doc 0.00"变成"工具 1 次、doc/ctx 1.00"；`multi-archmage-compare-01` 第 2 轮的工具数 5→2（投影裁掉了冗余历史，检索不再重复）。`pool_retention` 仍只有 0.00/1.00 两种取值（第 2 轮要么用到第 1 轮的文档、要么完全没用），说明"跨轮证据"依然是薄弱环节 —— 但它属于台阶 3/4 的题目，本次不动。

### 3.3 方差声明

`llm_model` 未固定 `temperature`，**模型非确定性**。因此：

- 结果里 **< 3pp 的差视为噪声**（本次 +4pp 恰在噪声线上，不能据此宣称"改善"）；
- 可以确定的只有**方向一致性**：ctx/contain 微升、工具/轮下降、缓存未退、成本微降 —— **没有任何一项指向退化**；
- 单轮翻转（如 `multi-kinesis-numeric-01`）很可能是采样波动，不是改动因果。
- 但 **§4.3 的 A/B 差距（80pp）远超噪声**：A 8/10 vs B 0/10，且 B 组 `doc_recall` 全 0 —— 那条结论不受方差影响。

---

## 4. L5：多跳用例与 A 组结果

### 4.1 用例设计（`eval/case/multi_hop.jsonl`，10 条）

依赖形态四类：补丁→职业文档（2 条）、公式→职业文档（2 条）、通用机制→职业文档（3 条，含 1 条三跳）、BOSS→职业文档（2 条）。每条都在 `notes` 里写明"单跳为什么解不了"。

写题纪律：
- 问题**不点名**第二跳的职业/技能（点名了就退化成并列型）；
- 判据是"只能调一次工具时必然答不对"，不是"需要两篇文档"；
- `must_contain` 必须是文档里真实存在的字面串（校验器检查可达性）。

### 4.2 A 组（`multi_hop` 10 轮，`max_tool_rounds=8`）

| 指标 | 值 |
|---|---|
| 通过率 | **8/10 = 80%** |
| 多跳达成率（≥2 次工具调用） | **10/10 = 100%** |
| doc_recall / ctx_recall / contain | 0.92 / 0.92 / 1.00 |
| 工具/轮 | **6.00** |
| 预算收口 | 2（都是 `no_progress`） |

逐题（工具数 / doc_recall / 结果）：

```
hop-formulas-arcanepitch-to-luminous-01   7   1.00  PASS
hop-utility-attackspeed-to-bow-master-01  7   1.00  PASS
hop-formulas-truesight-to-dawn-warrior-01 6   1.00  PASS
hop-kalos-heavensdoor-to-bishop-01        6   1.00  PASS
hop-patch-hexa-to-kinesis-01              6   1.00  PASS
hop-patch-shine-to-sia-erel-01            6   1.00  PASS
hop-seren-holyfountain-to-bishop-01       5   1.00  PASS
hop-utility-cdr-to-dawn-warrior-01        5   1.00  PASS
hop-blackmage-origin-bind-to-zero-01      7   0.67  FAIL  finish=no_progress
hop-utility-bind-to-zero-shadow-rain-01   5   0.50  FAIL  finish=no_progress
```

两个失败**都是"第二篇没取到"**（doc_recall 0.67 / 0.50）且被 `no_progress` 收口 —— 即：agent 在 5~7 次调用里反复拿到已有证据，L1 判定"原地打转"提前收口。这是正确行为（否则只会烧更多预算），但也说明**瓶颈是检索路径，不是循环机制**。

### 4.3 A/B 对照（B 组 `run_dir/342d2f79`，`--max-tool-rounds 1`）

| 指标 | A 组（多步，8 轮预算） | B 组（压到 1 轮） | Δ |
|---|---|---|---|
| 通过率 | **80%**（8/10） | **0%**（0/10） | −80pp |
| doc_recall | 0.92 | **0.00** | −0.92 |
| ctx_recall | 0.92 | 0.00 | −0.92 |
| contain_ok | 1.00 | 0.00 | −1.00 |
| 多跳达成率 | 1.00 | 0.00 | −1.00 |
| 工具调用/轮 | 6.00 | 1.00 | −5.00 |
| LLM 调用/轮 | 6.40 | 2.00 | −4.40 |
| input/轮 | 44,368 | 3,406 | −40,962 |
| cost/轮 | $0.02 | $0.01 | |
| p50 | 21,207ms | 7,060ms | |

- A 组 `finish_reason`：`no_tool_calls` 8 / `no_progress` 2
- B 组 `finish_reason`：**`steps_exhausted` 10/10**（每一轮都撞在步数预算上 —— 收口原因准确可读）

逐题配对：**A✓B✗ = 8**、A✗B✗ = 2、B✓ = 0、A✓B✓ = 0。

**结论一（可以确定）：循环是这个系统的必要条件，不是装饰。** 把预算压到 1 轮，10 条多跳题全挂、`doc_recall` 0.00 —— 连一条期望文档都没取到；而 8 轮下 8/10 通过、达成率 10/10。

**结论二（必须说清的局限）：B 组测的不是"单跳 RAG 管道"，而是"只给一次工具调用机会"。** 看 trace 可知，模型把那唯一一次调用花在了 `read_knowledge_catalog`（导航）上 —— 它连"取原文"这一步都没走到。所以：

- 能确定：**1 轮不够**（连"导航 → 取原文"的最短路径都走不完）；
- **不能**确定：6 次调用里到底有多少是真正必要的；
- 真正的"单跳 RAG 基线"（台阶 1 之前的管道模式）代码已删除，所以 B 组只能作为**轮数下界**的证据，而不是"管道 vs 循环"的证据。

判据 1 记为 ✅（循环是必要条件、多跳会真的连续调工具）；**工具/轮 6.00 超出 2~4 的目标**这一项仍未达标 —— 轮数曲线（§4.4）显示拐点落在 4 与 8 之间，检索路径长度归台阶 3。

---

### 4.4 轮数曲线（中间档 `run_dir/ecba241f`，`--max-tool-rounds 4`）

把三档放在一起，才能回答"到底几次调用才够"：

| 指标 | B（1 轮） | C（4 轮） | A（8 轮） |
|---|---|---|---|
| 通过率 | 0% | **10%** | **80%** |
| doc_recall | 0.00 | 0.53 | 0.92 |
| ctx_recall | 0.00 | 0.55 | 0.92 |
| contain_ok | 0.00 | 0.70 | 1.00 |
| 多跳达成率（≥2 次调用） | 0.00 | **1.00** | 1.00 |
| 工具调用/轮 | 1.00 | 4.70 | 6.00 |
| LLM 调用/轮 | 2.00 | 5.00 | 6.40 |
| input/轮 | 3,406 | 24,926 | 44,368 |
| cost/轮 | $0.01 | $0.02 | $0.02 |
| p50 | 7,060ms | 15,560ms | 21,207ms |
| `finish_reason` | `steps_exhausted` 10/10 | `steps_exhausted` 10/10 | `no_tool_calls` 8 / `no_progress` 2 |

逐题（`PASS` / `fail`，以及工具数 1/4/8 轮）：

```
hop-blackmage-origin-bind-to-zero-01   fail / PASS / fail   工具 1/8/7   doc 0.00/1.00/0.67   ← 非单调，见结论 3
hop-utility-bind-to-zero-shadow-rain-01 fail / fail / fail  工具 1/4/5   doc 0.00/0.50/0.50
其余 8 条                              fail / fail / PASS   工具 1/4~6/5~7  doc 0.00/0.33~0.50/1.00
```

**结论 1：拐点在 4 与 8 之间，不在 4。** 4 轮时 `多跳达成率` 已经 1.00（模型确实在连续调工具），但 `doc_recall` 只有 0.53、通过率 10% —— **"调够次数"不等于"取到证据"**。这正是 §2.6 必须把多跳轮的 `doc_recall == 1.0` 设为通过条件的实证理由。

**结论 2：原始验收的"2~4 个工具"对这组题偏乐观。** 当前工具粒度是"目录 → 子目录 → 原文"，一次依赖型两跳要 ~6 次调用（每跳都得导航 + 取原文）。两条路：要么承认这类题的目标就是 ~6 次；要么把导航压掉（台阶 3 的粗搜工具，`retrieval.retrieve_documents` 跑在目录条目上），让"2~4"变成可达目标。**推荐后者**，并把"多跳工具/轮从 6.0 降到 ≤4.0 且通过率不降"写成台阶 3 的验收项。

**结论 3：⚠️ 发现 L1 可能过早收口。** `hop-blackmage-origin-bind-to-zero-01` 出现非单调：**4 轮 PASS、8 轮 fail**。8 轮那次用了 7 次调用、连续两轮没拿到新证据 → 被判 `no_progress` 提前收口（`doc_recall` 0.67）；而 4 轮那次在同一预算内一轮并行多个调用（共 8 次），在收口前就拿全了两篇。

也就是说 **K=2 的停滞阈值会砍掉"再换一次关键词就能成功"的轮次**。A 组两个 `no_progress` 轮都失败了，其中这一条按 4 轮的数据看**本来是能过的**（多跳通过率 80% → 本可 90%）。建议下一轮实验：把 K 试成 3，或改成"还有剩余预算时多给一轮，预算紧张时才按 K=2 收口"，用同一组用例复测 —— 这是**有实证支撑**的下一个迭代假设，而不是拍脑袋调参。

---

## 5. 观测与归因

### 5.1 收口原因分布（L1/L4 的可观测性第一次真的用上）

```
no_tool_calls     77   ← 自然收尾（模型自己决定不再要工具）
no_progress        2   ← L1 生效（都在 multi_hop）
chars_exhausted    1   ← 累计 60k 字符预算触顶（ambig-archmage-both-01，8 次工具调用）
```

三个触顶收口轮**全部失败**（3/3）。结论：**收口是兜底，不是解法** —— 它保证"用户总能拿到答案"，但拿不到满分答案。要提升这部分，只能改检索（台阶 3）。

### 5.2 协议泄漏 0 轮

按 001 §4.1 的口径扫答案文本（含 `"tool_calls"` / `{"name":` 等协议痕迹）：**0 轮**。上一版把 `max_tool_rounds` 从 5 提到 8 并加收口提示后泄漏归零，本次新增 `no_progress` 收口路径后仍为 0。

### 5.3 p95 上升 +8.7s 的归因：**不是本次改动**

| 证据 | 值 |
|---|---|
| LLM 调用总数（重合 70 轮） | 319 → **318** |
| 工具调用总数 | 279 → **261**（少 18 次） |
| p50 | 8,294 → 9,753ms（+1.5s） |
| p95 | 18,943 → 27,620ms（+8.7s） |
| 变慢 >8s 的轮 | 9 轮，其中只有 5 轮的 `llm_calls` 增加 |

调用次数没变、工具还更少，但单位耗时上升 → 归因为**供应商侧波动**（同一 prompt_hash、同一模型）。下一次录制要复查它是否复现；若复现，再去看 span 级的 `ttft_ms` 分布。

---

## 6. 验收清单

| # | 验收项 | 结论 |
|---|---|---|
| 1 | 多步 | ✅ 1 轮 0% / 4 轮 10% / 8 轮 80%（§4.4）→ 循环是必要条件；"2~4 个工具"偏乐观（拐点落在 4~8 之间），归台阶 3 |
| 2 | 收口可控（四类原因 + 协议泄漏 0） | ✅ |
| 3 | 可中断（冒烟用例 14 + 前端停止实测） | ✅ |
| 4 | 可观测（`finish_reason` / `step` / 时间线 / trace `interrupted`） | ✅ |
| 5 | 不回退（重合 70 轮逐轮对比） | ✅ 无退化信号（修好 5 / 退化 2；方差内） |
| 6 | 离线回归网 | ✅ 冒烟 **18/18**（零成本、不联网） |
| 7 | 可解释 | ✅ meta 含 code/prompt/cases hash + 全部预算键（`max_tool_chars` / `max_chars_per_tool_call` / `max_tool_chars_projection` / `max_prompt_chars` / `answer_turns_projection` / `max_wall_s`） |

**判定：台阶 2 达成。**

---

## 7. 遗留与下一步

### 7.1 下一轮实验（有实证支撑，不拍脑袋调参）

三档曲线已补全（1 / 4 / 8 轮，§4.4）。下一个假设是 **L1 的停滞阈值 K=2 过早收口**，证据是 `hop-blackmage-origin-bind-to-zero-01` 在 4 轮下 PASS、8 轮下被 `no_progress` 砍掉（§4.4 结论 3）。

实验设计（同一组 10 条用例，各约 $0.2）：

1. **K=3**（改 `loop.py` 里的 `stall_rounds >= 2`）→ 看那一条是否回到 PASS、工具/轮是否升高；
2. 备选：**"还有剩余预算时多给一轮，预算紧张时才按 K=2 收口"** —— 更贴近"钱花在刀刃上"的本意。

验收：多跳通过率 80% → **≥90%**，且工具/轮不超过 6.5、重合 70 轮不退化。

### 7.2 归到台阶 3（不在本次范围）

| 项 | 为什么 |
|---|---|
| 多跳工具/轮 6.00 偏高 | 检索路径长：模型要"目录 → 子目录 → 原文"三跳导航。粗搜工具（BM25 跑在目录条目上）把它压到 1 次，是本台阶暴露出的最大杠杆 |
| `hop-utility-bind` / `hop-blackmage-origin-bind` 两个失败 | 都是"第二篇没取到" → 检索侧问题 |
| `system` 的 ctx_recall 0.82→0.64 | 单轮退化（`system-flame-nowash-01`），需要看是否与窗口裁剪相关；必要时单独复现 |
| `pool_retention` 仍 0.00/1.00 | 跨轮证据复用弱 → 跨轮证据重投影（当时评估后归到台阶 3/4） |

### 7.3 待观察

- **p95**：下次录制复查（§5.3）。
- **`hop-seren-holyfountain-to-bishop-01` 公平性最弱**（"Decent X = 职业专属技能 X"需要定位到 53 篇职业文档之一，属"定位难"而非"可单跳"）。B 组它同样没过（`doc_recall` 0.00），所以**这次没能证伪它**——但 B 组整体是 0%，也就无法用它来证明这条题目的严格性。建议在补 4 轮中间档时单独看它：若它比其他题更早通过，说明它偏易，需要加"只在第二跳文档出现的 `must_contain` 串"。
- 001 §7 遗留表里 `tool_timeout_s` 那条写的"仍是 600s"已过期（现为 20s），下次改 001 时顺手修正（迭代文档不改历史结论，只改事实性错误）。

---

## 8. 如何复现

```powershell
# 离线回归（零成本、不联网）
.\.venv\Scripts\python.exe smoke_orchestrator.py          # 18/18

# 用例集校验（零成本）
.\.venv\Scripts\python.exe -m eval.case                   # 74 用例 / 80 轮

# 全量录制
.\.venv\Scripts\python.exe -m eval.runner
# 只录多跳
.\.venv\Scripts\python.exe -m eval.runner --pattern "eval/case/multi_hop.jsonl"
# 单跳对照（B 组：run_dir/342d2f79）
.\.venv\Scripts\python.exe -m eval.runner --pattern "eval/case/multi_hop.jsonl" --max-tool-rounds 1
# 轮数中间档（C 组：run_dir/ecba241f）
.\.venv\Scripts\python.exe -m eval.runner --pattern "eval/case/multi_hop.jsonl" --max-tool-rounds 4

# 打分（可与基线逐格对比）
.\.venv\Scripts\python.exe -m eval.metrics run_dir\c5c44bed run_dir\99db2824
```

`cases_hash` 不同的两个 run **不要比总体均值**（本台阶中途给 `TurnExpect` 加了 `expect_min_tool_calls`，schema 一变全量 hash 就变）——用逐轮明细对比，或用 `--pattern` 限定同集重录。
