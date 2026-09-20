"""指标判定、聚合与报告（打分侧，纯函数、零成本、可反复重算）。

两种入口共用同一套实现：
    summary = score(run_record)          # runner 落盘后内存直传，只为即时反馈
    summary = score(load_run(path))      # 从录制重算，改断言/加指标不用重跑 LLM
两者必须等价，由 round-trip 测试保证。

分层：
    judge()     逐轮判定（检索层 / 生成层）      -> TurnVerdict
    aggregate() 聚合（跳过 None 的项，失败进分母）-> Summary
    render()    只负责排版
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from eval.case import norm
from eval.record import CaseRecord, RunRecord, TurnRecord, load_run


@dataclass
class TurnVerdict:
    case_id: str
    doc_recall: float | None
    ctx_recall: float | None
    contain_pass: bool | None
    retrieval_ok: bool | None
    doc_precision: float | None
    missing: list[str] = field(default_factory=list)      # must_contain 未命中的项
    bad_hits: list[str] = field(default_factory=list)     # must_not_contain 命中的项
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    failed: bool = False

    # 报告分组用
    turn_idx: int = 0
    category: str = ""
    tags: list[str] = field(default_factory=list)
    pool_retention: float | None = None                   # 多轮：前几轮文档的保留率
    reason: str = ""                                      # 失败原因（error 文本 / not_run）

    # token / 成本（None = 未知，不要与 0 混）
    llm_calls: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    usage_estimated: bool = False
    cost_usd: float | None = None
    spans: list[dict] = field(default_factory=list)       # 每 span 摘要（含算好的 cost_usd）

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)

    @property
    def gen_ms(self) -> float | None:
        """首字之后的生成耗时。ttft 缺失（没有 token 事件）时不可得。"""
        return None if self.ttft_ms is None else max(0.0, self.latency_ms - self.ttft_ms)

    @property
    def passed(self) -> bool:
        """该轮是否通过。

        注意：三项断言全为 None（该轮没有任何期望）时视为通过 —— 它没有考任何东西，
        不该被算作失败；要发现这种"空用例"，看 n_asserted 与 contain_pass is None 的计数。
        """
        if self.failed:
            return False
        if self.contain_pass is False:
            return False
        if self.retrieval_ok is False:
            return False
        return True


@dataclass
class Summary:
    category: str = ""
    count: int = 0                                        # 参与统计的轮数（失败轮也在内）
    failed: int = 0
    passed: int = 0
    doc_recall: float | None = None
    ctx_recall: float | None = None
    contain_pass: float | None = None
    p50_ms: float | None = None                           # 生成耗时（latency - ttft）
    p95_ms: float | None = None
    latency_p50_ms: float | None = None                   # 端到端耗时
    latency_p95_ms: float | None = None
    tokens_in_per_turn: float | None = None
    tokens_out_per_turn: float | None = None
    cost_per_turn: float | None = None
    analysis_cost_share: float | None = None              # 分析阶段占总成本比例
    estimated_turns: int = 0                              # token 是估算值的轮数


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------


def judge(case_snapshot: dict, turn_record: TurnRecord) -> TurnVerdict:
    turn_idx = turn_record.turn_idx
    turns = case_snapshot.get("turns") or []
    if turn_idx >= len(turns) or not isinstance(turns[turn_idx], dict):
        raise ValueError(
            "录制里的 case['turns'] 不是逐轮期望字典 —— 该录制早于 expect_per_turn 改造，"
            "请重跑一次录制（旧录制只有扁平期望，无法按轮判定）"
        )
    expect_info: dict = turns[turn_idx]

    got_doc = set(turn_record.injected_doc_ids)
    expect_doc = set(expect_info.get("expect_doc_ids") or [])
    expect_doc_any = set(expect_info.get("expect_doc_ids_any") or [])

    doc_recall = len(expect_doc & got_doc) / len(expect_doc) if expect_doc else None

    # 片段层：锚点在【本轮注入的片段文本】里搜；分母是锚点数，不是片段数
    anchors = expect_info.get("expect_context_contains") or []
    if not anchors:
        ctx_recall = None
    elif turn_record.injected_chunk_texts:
        texts = [norm(t) for t in turn_record.injected_chunk_texts]
        hit = sum(1 for a in anchors if any(norm(a) in t for t in texts))
        ctx_recall = hit / len(anchors)
    else:
        # 有 chunk_ids 却没有 chunk_texts → 录制缺字段，记 None（不是 0.0），
        # 否则会把"没数据"报成"没命中"。
        ctx_recall = None if turn_record.injected_chunk_ids else 0.0

    if expect_info.get("expect_no_retrieval"):
        retrieval_ok = not turn_record.did_retrieve
    elif expect_doc or expect_doc_any or anchors:
        retrieval_ok = turn_record.did_retrieve
    else:
        retrieval_ok = None

    # 文档级精确率：分母必须是抽到的【文档】，不是片段（否则交集恒为空）
    got_doc_all = set(turn_record.injected_doc_ids)
    doc_precision = len(expect_doc & got_doc_all) / len(got_doc_all) if (expect_doc and got_doc_all) else None

    answer = norm(turn_record.answer)
    mc = expect_info.get("must_contain") or []            # ALL
    mca = expect_info.get("must_contain_any") or []       # ANY
    mnc = expect_info.get("must_not_contain") or []       # NOT ANY

    missing = [s for s in mc if norm(s) not in answer]
    bad_hits = [s for s in mnc if norm(s) in answer]
    any_ok = (not mca) or any(norm(s) in answer for s in mca)

    contain_pass = None if not (mc or mca or mnc) else (not missing) and any_ok and (not bad_hits)

    return TurnVerdict(
        case_id=case_snapshot.get("id", ""),
        doc_recall=doc_recall,
        ctx_recall=ctx_recall,
        contain_pass=contain_pass,
        retrieval_ok=retrieval_ok,
        doc_precision=doc_precision,
        missing=missing,
        bad_hits=bad_hits,
        latency_ms=turn_record.latency_ms,
        ttft_ms=turn_record.ttft_ms,
        failed=turn_record.error is not None,
        turn_idx=turn_idx,
        category=case_snapshot.get("category", ""),
        tags=list(case_snapshot.get("tags") or []),
        reason=turn_record.error or "",
        llm_calls=turn_record.llm_calls,
        input_tokens=turn_record.input_tokens,
        output_tokens=turn_record.output_tokens,
        usage_estimated=turn_record.usage_estimated,
        spans=[dict(s) for s in (turn_record.spans or [])],
    )


def _attach_retention(
    verdicts: list[TurnVerdict], expect_turns: list, record_turns: list[TurnRecord]
) -> None:
    """多轮专用：第 k 轮还留着前几轮期望过的文档吗（量化 _knowledge_pool 的跨轮保留）。"""
    prior: set[str] = set()
    for k in range(1, len(expect_turns)):
        prev = expect_turns[k - 1]
        if isinstance(prev, dict):
            prior |= set(prev.get("expect_doc_ids") or [])
        if k >= len(record_turns):
            break
        got = set(record_turns[k].injected_doc_ids)
        verdicts[k].pool_retention = len(prior & got) / len(prior) if prior else None


def judge_case(case_record: CaseRecord) -> list[TurnVerdict]:
    """逐轮判定。分母用【期望轮数】：被异常截断而没跑到的轮算失败，不能静默跳过。"""
    snapshot = case_record.case
    expect_turns = snapshot.get("turns") or []

    verdicts: list[TurnVerdict] = []
    for i in range(len(expect_turns)):
        if i < len(case_record.turns):
            verdicts.append(judge(snapshot, case_record.turns[i]))
        else:
            verdicts.append(
                TurnVerdict(
                    case_id=snapshot.get("id", ""),
                    doc_recall=None,
                    ctx_recall=None,
                    contain_pass=False,        # 没跑 = 没答对
                    retrieval_ok=None,
                    doc_precision=None,
                    failed=True,
                    turn_idx=i,
                    category=snapshot.get("category", ""),
                    tags=list(snapshot.get("tags") or []),
                    reason="not_run",
                )
            )

    _attach_retention(verdicts, expect_turns, case_record.turns)
    return verdicts


def judge_run(run: RunRecord) -> list[TurnVerdict]:
    verdicts = [v for case_record in run.cases for v in judge_case(case_record)]
    _attach_costs(verdicts, run.meta or {})
    return verdicts


# ---------------------------------------------------------------------------
# 成本：token 存进 span，金额在打分时算（按 meta 里的价目快照）
# ---------------------------------------------------------------------------


def _span_cost(span: dict, pricing: dict) -> float | None:
    """单个 span 的成本。None = 算不了（非 llm span / 没拿到 usage / 没配单价）。

    关键：**没拿到 usage 时返回 None，不要返回 0** —— 否则成本报表会
    把"不知道"显示成"免费"。
    """
    if span.get("kind") != "llm":
        return None
    pin = pricing.get("llm_price_input_per_mtok")
    pout = pricing.get("llm_price_output_per_mtok")
    if pin is None or pout is None:
        return None
    if span.get("input_tokens") is None and span.get("output_tokens") is None:
        return None
    return ((span.get("input_tokens") or 0) * pin
            + (span.get("output_tokens") or 0) * pout) / 1_000_000


def _attach_costs(verdicts: list[TurnVerdict], meta: dict) -> None:
    pricing = meta.get("pricing") or {}
    for v in verdicts:
        total, known = 0.0, False
        for span in v.spans:
            cost = _span_cost(span, pricing)
            span["cost_usd"] = cost
            if cost is not None:
                total += cost
                known = True
        v.cost_usd = total if known else None


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------


def mean_of(verdicts: list[TurnVerdict], field: str) -> float | None:
    xs = [getattr(v, field) for v in verdicts if getattr(v, field) is not None]
    return sum(xs) / len(xs) if xs else None


def pct(values: list[float], q: float) -> float | None:
    """nearest-rank 分位数。样本少时它比插值诚实（n=10 的 p95 就是最大值）。"""
    if not values:
        return None
    xs = sorted(values)
    k = max(1, math.ceil(q * len(xs)))
    return xs[k - 1]


def aggregate(verdicts: list[TurnVerdict], category: str = "") -> Summary:
    # not_run 的轮 latency_ms 是 0.0，会把延迟分布拉低 —— 它们必须排除。
    ran = [v for v in verdicts if v.reason != "not_run"]
    gen = [v.gen_ms for v in ran if v.gen_ms is not None]
    lat = [v.latency_ms for v in ran if v.latency_ms is not None]

    ins = [v.input_tokens for v in verdicts if v.input_tokens is not None]
    outs = [v.output_tokens for v in verdicts if v.output_tokens is not None]
    costs = [v.cost_usd for v in verdicts if v.cost_usd is not None]

    # 分析阶段占总成本的比例：这是"每轮重跑意图分析划不划算"的直接依据
    analysis_cost = sum(s.get("cost_usd") or 0.0 for v in verdicts for s in v.spans
                        if s.get("name") == "analysis")
    all_span_cost = sum(s.get("cost_usd") or 0.0 for v in verdicts for s in v.spans)

    return Summary(
        category=category,
        count=len(verdicts),                              # 失败轮也计入分母
        failed=sum(1 for v in verdicts if v.failed),
        passed=sum(1 for v in verdicts if v.passed),
        doc_recall=mean_of(verdicts, "doc_recall"),
        ctx_recall=mean_of(verdicts, "ctx_recall"),
        contain_pass=mean_of(verdicts, "contain_pass"),
        p50_ms=pct(gen, 0.5),
        p95_ms=pct(gen, 0.95),
        latency_p50_ms=pct(lat, 0.5),
        latency_p95_ms=pct(lat, 0.95),
        tokens_in_per_turn=(sum(ins) / len(ins)) if ins else None,
        tokens_out_per_turn=(sum(outs) / len(outs)) if outs else None,
        cost_per_turn=(sum(costs) / len(costs)) if costs else None,
        analysis_cost_share=(analysis_cost / all_span_cost) if all_span_cost else None,
        estimated_turns=sum(1 for v in verdicts if v.usage_estimated),
    )


def score(run: RunRecord) -> tuple[list[TurnVerdict], Summary]:
    verdicts = judge_run(run)
    return verdicts, aggregate(verdicts)


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

DASH = "--"          # 只用 GBK 能编码的字符：Windows 控制台打不出 ✓ / ⚠ 会直接崩


def _w(text: str) -> int:
    """终端显示宽度：CJK 全角字符占 2 列。

    直接用 len() 补空格会让中文标签所在的列比 ASCII 行多占几列，整张表错位。
    """
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int, right: bool = False) -> str:
    gap = max(0, width - _w(text))
    return (" " * gap + text) if right else (text + " " * gap)


def _row(cells: list[tuple[str, int, bool]]) -> str:
    """cells: [(文本, 列宽, 是否右对齐)]"""
    return "    " + "".join(_pad(t, w, r) for t, w, r in cells)


def _num(x: float | None, nd: int = 2) -> str:
    return DASH if x is None else f"{x:.{nd}f}"


def _ms(x: float | None) -> str:
    return DASH if x is None else f"{x:.0f}"


def _ratio(x: float | None) -> str:
    return DASH if x is None else f"{x * 100:.0f}%"


def _bucket(v: TurnVerdict) -> str | None:
    """把不通过的轮归到具体环节 —— 决定你下一步该改检索还是改 prompt。"""
    if v.reason == "not_run":
        # 根本没执行：它既没有检索也没有答案，不能算进任何诊断格
        return "未执行（被前轮中断）"
    if v.contain_pass is None:
        return None
    doc_ok = v.doc_recall is None or v.doc_recall >= 1.0
    ctx_ok = v.ctx_recall is None or v.ctx_recall >= 1.0
    if doc_ok and ctx_ok:
        return "OK" if v.contain_pass else "证据齐但答案没用上（生成问题）"
    if doc_ok and not ctx_ok:
        return "文档对了但片段没进预算（片段级召回）"
    return "蒙对（最危险的绿）" if v.contain_pass else "检索选错文档（意图解析/候选）"


def _group_table(title: str, groups: dict[str, list[TurnVerdict]]) -> list[str]:
    head = ("分组", 22, False), ("轮数", 6, True), ("失败", 6, True), ("通过率", 8, True), \
           ("doc_recall", 12, True), ("ctx_recall", 12, True), ("contain", 9, True), \
           ("生成p50", 10, True), ("总延迟p50", 11, True)
    lines = [title, _row(head)]
    for name in sorted(groups):
        s = aggregate(groups[name], category=name)
        rate = s.passed / s.count if s.count else None
        lines.append(
            _row([
                (name, 22, False), (str(s.count), 6, True), (str(s.failed), 6, True),
                (_ratio(rate), 8, True), (_num(s.doc_recall), 12, True),
                (_num(s.ctx_recall), 12, True), (_num(s.contain_pass), 9, True),
                (_ms(s.p50_ms), 10, True), (_ms(s.latency_p50_ms), 11, True),
            ])
        )
    return lines


def render(
    turn_verdicts: list[TurnVerdict],
    summary: Summary,
    baseline: Summary | None = None,
) -> str:
    if not turn_verdicts:
        return "（没有任何轮次可统计：录制为空或用例集为空）"

    out: list[str] = []
    line = "=" * 78
    out.append(line)
    out.append(f"评测报告  轮数={summary.count}  失败={summary.failed}  通过率={_ratio(summary.passed / summary.count if summary.count else None)}")
    out.append(line)

    # 1) 总体（含与基线的差值）
    out.append("【总体】")
    rows = [
        ("doc_recall  期望文档命中率", summary.doc_recall, "doc_recall"),
        ("ctx_recall  片段锚点命中率", summary.ctx_recall, "ctx_recall"),
        ("contain_ok  答案断言通过率", summary.contain_pass, "contain_pass"),
    ]
    for label, value, attr in rows:
        base = getattr(baseline, attr) if baseline else None
        delta = ""
        if value is not None and base is not None:
            d = value - base
            # 指标波动 < 3% 视为噪声，不值得为它改架构
            delta = f"d={d:+.2f}" + ("  (噪声内)" if abs(d) < 0.03 else "  (显著)")
        out.append(_row([(label, 34, False), (_num(value), 8, True),
                         (delta, 22 if delta else 0, False)]))
    out.append(_row([("生成耗时 latency-ttft", 34, False),
                     (f"p50={_ms(summary.p50_ms)}", 16, False),
                     (f"p95={_ms(summary.p95_ms)}", 0, False)]))
    out.append(_row([("端到端耗时", 34, False),
                     (f"p50={_ms(summary.latency_p50_ms)}", 16, False),
                     (f"p95={_ms(summary.latency_p95_ms)}", 0, False)]))
    out.append(f"    样本数 n={summary.count}（p95 在 n<20 时就是最大值，别当分位数读）")
    out.append("")

    # 2) 按 category
    by_cat: dict[str, list[TurnVerdict]] = {}
    for v in turn_verdicts:
        by_cat.setdefault(v.category or "(未分类)", []).append(v)
    out.extend(_group_table("【按 category】", by_cat))
    out.append("")

    # 3) 按轮次（多轮专用：能看出"上下文变长后变差"）
    by_turn: dict[str, list[TurnVerdict]] = {}
    for v in turn_verdicts:
        by_turn.setdefault(f"turn{v.turn_idx}", []).append(v)
    if len(by_turn) > 1:
        out.extend(_group_table("【按轮次】", by_turn))
        out.append("")

    # 3.5) 成本
    out.append("【成本】")
    out.append(_row([("每轮 token 均值（in/out）", 34, False),
                     (f"{_num(summary.tokens_in_per_turn, 1)}/{_num(summary.tokens_out_per_turn, 1)}", 0, False)]))
    if summary.cost_per_turn is None:
        out.append("    每轮成本  --  （未配置单价：settings 里的 llm_price_*_per_mtok 为空，"
                   "或本轮完全没拿到 usage）")
    else:
        out.append(_row([("每轮成本均值", 34, False),
                         (f"${summary.cost_per_turn:.6f}", 0, False)]))
        out.append(_row([("分析阶段占总成本", 34, False),
                         (_ratio(summary.analysis_cost_share), 0, False)]))
    if summary.estimated_turns:
        out.append(f"    注意：{summary.estimated_turns} 轮的 token 是**估算值**（usage_estimated=True），"
                   "成本不可与真实值直接比较")
    for name in sorted(by_cat):
        s = aggregate(by_cat[name], category=name)
        if s.tokens_in_per_turn is None and s.tokens_out_per_turn is None:
            continue
        tok = (s.tokens_in_per_turn or 0) + (s.tokens_out_per_turn or 0)
        cost = f"${s.cost_per_turn:.6f}" if s.cost_per_turn is not None else DASH
        out.append(_row([("    " + name, 34, False),
                         (f"tok/turn={tok:.0f}", 18, False), (cost, 0, False)]))
    out.append("")

    # 4) 归因矩阵
    buckets = Counter(b for b in (_bucket(v) for v in turn_verdicts) if b)
    out.append("【归因矩阵】（只统计有答案断言的轮）")
    if buckets:
        for name, n in buckets.most_common():
            out.append(_row([(name, 40, False), (str(n), 5, True)]))
    else:
        out.append("    （没有轮次带 must_contain / must_contain_any / must_not_contain 断言）")
    out.append("")

    # 5) 红线：regression 标签必须全绿
    reg = [v for v in turn_verdicts if "regression" in v.tags]
    if reg:
        reg_pass = sum(1 for v in reg if v.passed)
        flag = "通过" if reg_pass == len(reg) else "未通过  <-- 红线"
        out.append(f"【红线】regression 标签：{reg_pass}/{len(reg)} {flag}")
    else:
        out.append("【红线】没有 regression 标签的用例 —— 建议给你的回归用例打上该标签")
    out.append("")

    # 6) 多轮池保留
    ret = [v.pool_retention for v in turn_verdicts if v.pool_retention is not None]
    if ret:
        out.append(f"【多轮池保留】均值={_num(sum(ret) / len(ret))}  "
                   f"最低={_num(min(ret))}  n={len(ret)}（<1.0 说明知识池丢了前轮的文档）")
        out.append("")

    # 7) 逐轮明细
    out.append("【逐轮明细】")
    out.append(_row([("case", 34, False), ("轮", 4, True), ("doc", 6, True), ("ctx", 6, True),
                     ("cont", 6, True), ("ret", 6, True), ("tok", 8, True), ("llm", 5, True),
                     ("总延迟", 9, True), ("生成", 8, True), ("  备注", 0, False)]))
    for v in turn_verdicts:
        cont = DASH if v.contain_pass is None else ("PASS" if v.contain_pass else "FAIL")
        note = []
        if v.reason:
            note.append(v.reason[:40])
        if v.missing:
            note.append(f"缺 must_contain {len(v.missing)} 项")
        if v.bad_hits:
            note.append(f"命中 must_not_contain {len(v.bad_hits)} 项")
        if v.retrieval_ok is False:
            note.append("该检索却没检索")
        out.append(_row([
            (v.case_id[:34], 34, False), (str(v.turn_idx), 4, True),
            (_num(v.doc_recall), 6, True), (_num(v.ctx_recall), 6, True),
            (cont, 6, True), (_num(v.pool_retention), 6, True),
            (str(v.total_tokens) if v.total_tokens is not None else DASH, 8, True),
            (str(v.llm_calls) if v.llm_calls is not None else DASH, 5, True),
            # 没执行的轮没有有意义的耗时，显示 -- 而不是 0
            (DASH if v.reason == "not_run" else _ms(v.latency_ms), 9, True),
            (_ms(v.gen_ms), 8, True),
            (("  " + "; ".join(note)) if note else "", 0, False),
        ]))

    # 8) 仍然缺的东西要显式说明，不要静默给 0
    out.append("")
    out.append("【暂缺】prompt cache 命中数（cached_tokens 字段已预留，provider 不返回时为空）；"
               "分析阶段内部的候选检索没有独立 span（已包含在 analysis 的耗时里）")
    out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 入口 2：从录制文件重算
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import sys

    args = (argv or sys.argv)[1:]
    if not args:
        print("用法: python -m eval.metrics <run.json 或 run 目录>")
        return 2

    run = load_run(args[0])
    try:
        verdicts, summary = score(run)
    except ValueError as e:
        print(f"无法打分：{e}")
        return 1
    print(f"run_id={run.run_id}  started_at={run.started_at}")
    meta = run.meta or {}
    print(
        "meta: commit={git_commit} dirty={git_dirty} code_hash={code_hash} "
        "cases_hash={cases_hash} prompt_hash={prompt_hash}".format(
            git_commit=meta.get("git_commit"), git_dirty=meta.get("git_dirty"),
            code_hash=meta.get("code_hash"), cases_hash=meta.get("cases_hash"),
            prompt_hash=meta.get("prompt_hash"),
        )
    )
    print()
    print(render(verdicts, summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
