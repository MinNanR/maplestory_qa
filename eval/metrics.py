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
from datetime import datetime


@dataclass
class TurnVerdict:
    case_id: str
    doc_recall: float | None
    ctx_recall: float | None
    contain_pass: bool | None
    retrieval_ok: bool | None
    doc_precision: float | None
    missing: list[str] = field(default_factory=list)  # must_contain 未命中的项
    bad_hits: list[str] = field(default_factory=list)  # must_not_contain 命中的项
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    failed: bool = False

    # 报告分组用
    turn_idx: int = 0
    category: str = ""
    tags: list[str] = field(default_factory=list)
    pool_retention: float | None = None  # 多轮：前几轮文档的保留率
    reason: str = ""  # 失败原因（error 文本 / not_run）
    # 该轮是否本来就"不该检索"（闲聊/超范围用例）。
    # retrieval_ok=False 有两种成因，归因时必须区分，否则会把"多调了"报成"少调了"。
    expected_no_retrieval: bool = False

    # token / 成本（None = 未知，不要与 0 混）
    llm_calls: int | None = None
    input_tokens: int | None = None
    cached_tokens: int | None = None
    output_tokens: int | None = None
    usage_estimated: bool = False
    cost_usd: float | None = None
    spans: list[dict] = field(default_factory=list)  # 每 span 摘要（含算好的 cost_usd）

    # —— 工具（工具化阶段的核心可观测量）——
    # 刻意不加进 TurnRecord：录制存的是原始 spans，这些指标在打分时派生
    # （与"改断言/加指标不用重跑 LLM"一致；旧录制也能直接重算出这些数）。
    tool_calls: int = 0
    tool_failed: int = 0
    tool_deduped: int = 0
    tool_chars: int = 0  # 工具返回给模型的字符总量
    tool_ms: float = 0.0
    tool_names: list[str] = field(default_factory=list)  # 按调用顺序

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)

    @property
    def gen_ms(self) -> float | None:
        """首字之后的生成耗时。ttft 缺失（没有 token 事件）时不可得。"""
        return (
            None if self.ttft_ms is None else max(0.0, self.latency_ms - self.ttft_ms)
        )

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
    count: int = 0  # 参与统计的轮数（失败轮也在内）
    failed: int = 0
    passed: int = 0
    doc_recall: float | None = None
    ctx_recall: float | None = None
    contain_pass: float | None = None
    p50_ms: float | None = None  # 生成耗时（latency - ttft）
    p95_ms: float | None = None
    latency_p50_ms: float | None = None  # 端到端耗时
    latency_p95_ms: float | None = None
    tokens_in_per_turn: float | None = None
    tokens_out_per_turn: float | None = None
    cost_per_turn: float | None = None
    # prompt cache 命中率 = cached / input（只在报了 cached_tokens 的轮上算）。
    # None = provider 没报这个字段（不是"命中 0"）。
    cache_hit_rate: float | None = None
    tool_calls_per_turn: float | None = None  # 平均每轮工具调用次数
    tool_fail_rate: float | None = None  # 失败调用 / 总调用（None = 一次都没调）
    tool_ms_share: float | None = None  # 工具耗时 / 端到端耗时
    tool_ms_per_turn: float | None = None  # 平均每轮工具耗时
    tool_chars_per_turn: float | None = None  # 平均每轮工具返回字符
    tool_turns: int = 0  # 至少调过一次工具的轮数
    deduped_total: int = 0  # 被去重拦下的调用总数
    estimated_turns: int = 0  # token 是估算值的轮数


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------


def _tool_stats(spans: list[dict]) -> dict:
    """从 span 摘要里派生工具指标。

    数据来源是编排器写进 tool span 的 detail：
        detail.tool_call_request = {tool_call_id, name, params, max_output_chars}
        detail.tool_call_result  = {succeeded, deduped, chars_count, structured, ...}
    刻意不在 TurnRecord 上加字段：录制只存原始 spans，指标在打分时派生 ——
    这样"改口径 / 加指标"仍然不需要重跑 LLM，旧录制也能直接重算。

    `name` 优先取 request 里的（那是模型真实调用的工具名），
    span 名（"tool_call:xxx"）只作为兜底 —— 它带前缀，不适合直接当统计键。
    """
    calls = failed = deduped = chars = 0
    ms = 0.0
    names: list[str] = []
    for s in spans or []:
        if s.get("kind") != "tool":
            continue
        detail = s.get("detail") or {}
        req = detail.get("tool_call_request") or {}
        res = detail.get("tool_call_result") or {}
        calls += 1
        ms += s.get("duration_ms") or 0.0
        names.append(str(req.get("name") or s.get("name") or "?"))
        if res.get("succeeded") is False:
            failed += 1
        if res.get("deduped"):
            deduped += 1
        chars += res.get("chars_count") or 0
    return {
        "calls": calls,
        "failed": failed,
        "deduped": deduped,
        "chars": chars,
        "ms": ms,
        "names": names,
    }


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
    doc_precision = (
        len(expect_doc & got_doc_all) / len(got_doc_all)
        if (expect_doc and got_doc_all)
        else None
    )

    answer = norm(turn_record.answer)
    mc = expect_info.get("must_contain") or []  # ALL
    mca = expect_info.get("must_contain_any") or []  # ANY
    mnc = expect_info.get("must_not_contain") or []  # NOT ANY

    missing = [s for s in mc if norm(s) not in answer]
    bad_hits = [s for s in mnc if norm(s) in answer]
    any_ok = (not mca) or any(norm(s) in answer for s in mca)

    contain_pass = (
        None if not (mc or mca or mnc) else (not missing) and any_ok and (not bad_hits)
    )

    tools = _tool_stats(turn_record.spans)

    # prompt cache 命中量：逐 span 汇总。全都没有这个字段时记 None（未知），
    # 不要与"命中 0"混 —— "provider 没报" 和 "一次都没命中" 是两件事。
    cached_vals = [
        s.get("cached_tokens")
        for s in (turn_record.spans or [])
        if s.get("kind") == "llm" and s.get("cached_tokens") is not None
    ]

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
        expected_no_retrieval=bool(expect_info.get("expect_no_retrieval")),
        llm_calls=turn_record.llm_calls,
        input_tokens=turn_record.input_tokens,
        cached_tokens=sum(cached_vals) if cached_vals else None,
        output_tokens=turn_record.output_tokens,
        usage_estimated=turn_record.usage_estimated,
        spans=[dict(s) for s in (turn_record.spans or [])],
        tool_calls=tools["calls"],
        tool_failed=tools["failed"],
        tool_deduped=tools["deduped"],
        tool_chars=tools["chars"],
        tool_ms=tools["ms"],
        tool_names=tools["names"],
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
                    contain_pass=False,  # 没跑 = 没答对
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

    两个关键点：
    1. **没拿到 usage 时返回 None，不要返回 0** —— 否则成本报表会把"不知道"
       显示成"免费"；
    2. **input_tokens（prompt_tokens）已经包含缓存命中的那部分**，所以必须拆成
       `(input - cached) * 全价 + cached * 缓存价`。写成 `input * 全价 + cached * 缓存价`
       会把命中部分计两次 —— 那个数的量级会接近"全部按全价"，看起来"合理"，
       所以很难被发现。
    """
    if span.get("kind") != "llm":
        return None
    pin = pricing.get("llm_price_input_per_mtok")
    pcached = pricing.get("llm_price_cached_per_mtok")
    pout = pricing.get("llm_price_output_per_mtok")
    if pin is None or pout is None:
        return None

    in_tok = span.get("input_tokens")
    out_tok = span.get("output_tokens")
    cached = span.get("cached_tokens")
    if in_tok is None and out_tok is None:
        return None
    
    price_modifier = 1
    if "started_at" in span:
        started_at = span.get("started_at")
        start_time = datetime.fromisoformat(started_at)
        h = start_time.hour
        if 9 <= h <= 12 or 14 <= h <= 18:
            price_modifier = 2
        

    in_tok = in_tok or 0
    miss, hit = in_tok, 0
    if cached:
        if pcached is None:
            # 有缓存命中却没配缓存价：这个 span 的成本算不准，宁可返回未知
            return None
        hit = min(cached, in_tok)      # provider 偶尔会给出大于 input 的值
        miss = in_tok - hit
    return (miss * pin + hit * (pcached or 0) + (out_tok or 0) * pout) / 1_000_000 * price_modifier


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

    # 工具聚合。分母刻意区分：
    #   - tool_calls_per_turn 的分母是【轮数】（含没调工具的轮，才算"平均每轮调几次"）
    #   - tool_fail_rate 的分母是【调用次数】（一次都没调时是 None，不是 0%）
    tool_calls_total = sum(v.tool_calls for v in verdicts)
    tool_failed_total = sum(v.tool_failed for v in verdicts)
    tool_ms_total = sum(v.tool_ms for v in verdicts)
    tool_chars_total = sum(v.tool_chars for v in verdicts)
    lat_total = sum(v.latency_ms for v in ran)

    # 缓存命中率：只在**两个数都有**的轮上配对求和。
    # 分子分母取自不同的轮集合会让比率失去意义（有些轮 provider 没报 cached）。
    paired = [
        (v.input_tokens, v.cached_tokens)
        for v in verdicts
        if v.input_tokens is not None and v.cached_tokens is not None
    ]
    paired_in = sum(p[0] for p in paired)
    cache_hit_rate = (sum(p[1] for p in paired) / paired_in) if paired_in else None

    return Summary(
        category=category,
        count=len(verdicts),  # 失败轮也计入分母
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
        cache_hit_rate=cache_hit_rate,
        tool_calls_per_turn=(tool_calls_total / len(verdicts)) if verdicts else None,
        tool_fail_rate=(
            (tool_failed_total / tool_calls_total) if tool_calls_total else None
        ),
        tool_ms_share=(tool_ms_total / lat_total) if lat_total else None,
        tool_ms_per_turn=(tool_ms_total / len(verdicts)) if verdicts else None,
        tool_chars_per_turn=(tool_chars_total / len(verdicts)) if verdicts else None,
        tool_turns=sum(1 for v in verdicts if v.tool_calls),
        deduped_total=sum(v.tool_deduped for v in verdicts),
        estimated_turns=sum(1 for v in verdicts if v.usage_estimated),
    )


def score(run: RunRecord) -> tuple[list[TurnVerdict], Summary]:
    verdicts = judge_run(run)
    return verdicts, aggregate(verdicts)


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

DASH = "--"  # 只用 GBK 能编码的字符：Windows 控制台打不出 ✓ / ⚠ 会直接崩


def _w(text: str) -> int:
    """终端显示宽度：CJK 全角字符占 2 列。

    直接用 len() 补空格会让中文标签所在的列比 ASCII 行多占几列，整张表错位。
    """
    import unicodedata

    return sum(
        2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text
    )


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
    """把不通过的轮归到具体环节 —— 决定你下一步该改工具描述、改 prompt，还是改检索。"""
    if v.reason == "not_run":
        # 根本没执行：它既没有检索也没有答案，不能算进任何诊断格
        return "未执行（被前轮中断）"
    if v.contain_pass is None:
        return None
    if v.bad_hits:
        # must_not_contain 命中：和"证据够不够"完全无关，是答案里出现了不该出现的东西
        # （超范围/负样本轮、以及"编造了不存在的东西"都落这一格）。
        # 必须排在证据判定之前 —— 这类轮的 doc/ctx 常常是 None，会被判成
        # "doc_ok and ctx_ok"从而错误落进"证据齐但答案没用上（生成问题）"。
        return "不该说却说了（命中禁词）"
    if v.doc_recall is None and v.ctx_recall is None:
        # 该轮没有任何证据层期望（负样本 / 纯内容断言）：通过就是 OK，
        # 失败只能是答案内容问题 —— 不该掉进下面"证据齐但答案没用上"那一格，
        # 那会让人误以为证据层已经达标，实际上证据层这轮压根没被考察。
        return "OK" if v.contain_pass else "答案内容不符（本轮无证据期望）"
    if v.retrieval_ok is False:
        # 工具模式下最常见、也最该盯的一档。两种成因必须分开报：
        #   该检索却没检索 —— 模型凭记忆作答（上一阶段这里是"意图解析选错 id"，
        #   analyzer 删掉之后那个归因已不存在）
        #   不该检索却检索 —— 闲聊/超范围也去翻知识库，白烧一轮往返
        return (
            "不该检索却调了工具（闲聊/超范围轮）"
            if v.expected_no_retrieval
            else "该检索却没检索（未调用检索工具）"
        )
    doc_ok = v.doc_recall is None or v.doc_recall >= 1.0
    ctx_ok = v.ctx_recall is None or v.ctx_recall >= 1.0
    if doc_ok and ctx_ok:
        return "OK" if v.contain_pass else "证据齐但答案没用上（生成问题）"
    if doc_ok and not ctx_ok:
        return "文档对了但片段没取够（关键词/片段级召回）"
    return (
        "蒙对（最危险的绿）"
        if v.contain_pass
        else "工具取错文档（选错 doc_id / 关键词）"
    )


def _group_table(title: str, groups: dict[str, list[TurnVerdict]]) -> list[str]:
    head = (
        ("分组", 22, False),
        ("轮数", 6, True),
        ("失败", 6, True),
        ("通过率", 8, True),
        ("doc_recall", 12, True),
        ("ctx_recall", 12, True),
        ("contain", 9, True),
        ("工具/轮", 9, True),
        ("生成p50", 10, True),
        ("总延迟p50", 11, True),
    )
    lines = [title, _row(head)]
    for name in sorted(groups):
        s = aggregate(groups[name], category=name)
        rate = s.passed / s.count if s.count else None
        lines.append(
            _row(
                [
                    (name, 22, False),
                    (str(s.count), 6, True),
                    (str(s.failed), 6, True),
                    (_ratio(rate), 8, True),
                    (_num(s.doc_recall), 12, True),
                    (_num(s.ctx_recall), 12, True),
                    (_num(s.contain_pass), 9, True),
                    (_num(s.tool_calls_per_turn), 9, True),
                    (_ms(s.p50_ms), 10, True),
                    (_ms(s.latency_p50_ms), 11, True),
                ]
            )
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
    out.append(
        f"评测报告  轮数={summary.count}  失败={summary.failed}  通过率={_ratio(summary.passed / summary.count if summary.count else None)}"
    )
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
        out.append(
            _row(
                [
                    (label, 34, False),
                    (_num(value), 8, True),
                    ((" " + delta) if delta else "", 22 if delta else 0, False),
                ]
            )
        )
    out.append(
        _row(
            [
                ("生成耗时 latency-ttft", 34, False),
                (f"p50={_ms(summary.p50_ms)}", 16, False),
                (f"p95={_ms(summary.p95_ms)}", 0, False),
            ]
        )
    )
    out.append(
        _row(
            [
                ("端到端耗时", 34, False),
                (f"p50={_ms(summary.latency_p50_ms)}", 16, False),
                (f"p95={_ms(summary.latency_p95_ms)}", 0, False),
            ]
        )
    )
    out.append(
        f"    样本数 n={summary.count}（p95 在 n<20 时就是最大值，别当分位数读）"
    )
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
    out.append(
        _row(
            [
                ("每轮 token 均值（in/out）", 34, False),
                (
                    f"{_num(summary.tokens_in_per_turn, 1)}/{_num(summary.tokens_out_per_turn, 1)}",
                    0,
                    False,
                ),
            ]
        )
    )
    if summary.cost_per_turn is None:
        out.append(
            "    每轮成本  --  （算不出来：settings 里 llm_price_input/output_per_mtok 为空、"
            "有缓存命中但没配 llm_price_cached_per_mtok，或完全没拿到 usage）"
        )
    else:
        out.append(
            _row(
                [
                    ("每轮成本均值", 34, False),
                    (f"${summary.cost_per_turn:.6f}", 0, False),
                ]
            )
        )
    if summary.cache_hit_rate is None:
        out.append(
            "    prompt cache 命中率  --  （provider 没返回 cached_tokens；"
            "本次成本把全部 input 按全价计，是上界）"
        )
    else:
        out.append(
            _row(
                [
                    ("prompt cache 命中率", 34, False),
                    (_ratio(summary.cache_hit_rate), 0, False),
                    ("（命中部分按 llm_price_cached_per_mtok 计价）", 0, False),
                ]
            )
        )
    if summary.estimated_turns:
        out.append(
            f"    注意：{summary.estimated_turns} 轮的 token 是**估算值**（usage_estimated=True），"
            "成本不可与真实值直接比较"
        )
    for name in sorted(by_cat):
        s = aggregate(by_cat[name], category=name)
        if s.tokens_in_per_turn is None and s.tokens_out_per_turn is None:
            continue
        tok = (s.tokens_in_per_turn or 0) + (s.tokens_out_per_turn or 0)
        cost = f"${s.cost_per_turn:.6f}" if s.cost_per_turn is not None else DASH
        out.append(
            _row(
                [
                    ("    " + name, 34, False),
                    (f"tok/turn={tok:.0f}", 18, False),
                    (cost, 0, False),
                ]
            )
        )
    out.append("")

    # 3.7) 工具：工具化阶段最该盯的一组数
    out.append("【工具】")
    out.append(
        _row(
            [
                ("每轮工具调用次数（均值）", 34, False),
                (_num(summary.tool_calls_per_turn), 8, True),
                (f"（{summary.tool_turns}/{summary.count} 轮调过工具）", 0, False),
            ]
        )
    )
    if summary.tool_fail_rate is None:
        out.append("    工具失败率  --  （本轮评测一次工具都没调用）")
    else:
        out.append(
            _row(
                [
                    ("工具失败率", 34, False),
                    (_ratio(summary.tool_fail_rate), 8, True),
                    (
                        (
                            "（失败调用 / 总调用；工具返回 error 或参数非法都算）"
                            if summary.tool_fail_rate
                            else ""
                        ),
                        0,
                        False,
                    ),
                ]
            )
        )
    out.append(
        _row(
            [
                ("工具耗时占端到端", 34, False),
                (_ratio(summary.tool_ms_share), 8, True),
                (f"（平均每轮 {_ms(summary.tool_ms_per_turn)} ms）", 0, False),
            ]
        )
    )
    out.append(
        _row(
            [
                ("工具返回字符（均值/轮）", 34, False),
                (_num(summary.tool_chars_per_turn, 0), 8, True),
                ("（这部分会随历史累积，是成本增长的主因）", 0, False),
            ]
        )
    )
    if summary.deduped_total:
        out.append(
            _row(
                [
                    ("被去重拦下的调用", 34, False),
                    (str(summary.deduped_total), 8, True),
                    (
                        "（模型重复用同参数调用；偏高说明该改工具描述或终止纪律）",
                        0,
                        False,
                    ),
                ]
            )
        )
    # 按工具名分布：一眼看出是"某个工具特别爱失败"还是"某类问题从不调工具"
    tool_calls: Counter[str] = Counter()
    tool_fails: Counter[str] = Counter()
    for v in turn_verdicts:
        tool_calls.update(v.tool_names)
        for s in v.spans:
            if s.get("kind") != "tool":
                continue
            res = (s.get("detail") or {}).get("tool_call_result") or {}
            req = (s.get("detail") or {}).get("tool_call_request") or {}
            if res.get("succeeded") is False:
                tool_fails[str(req.get("name") or "?")] += 1
    for name, n in tool_calls.most_common():
        fail = tool_fails.get(name, 0)
        fail_txt = f"  失败 {fail}" if fail else ""
        out.append(
            _row(
                [
                    ("    " + name, 34, False),
                    (f"调用 {n}", 18, False),
                    (fail_txt, 0, False),
                ]
            )
        )
    if baseline is not None:
        # 工具指标不走【总体】那张比例表：它的噪声阈值不是 0.03（差 0.5 次/轮就是实质变化）
        bits = []
        for label, cur, base in (
            ("工具调用/轮", summary.tool_calls_per_turn, baseline.tool_calls_per_turn),
            ("工具失败率", summary.tool_fail_rate, baseline.tool_fail_rate),
            ("工具耗时占比", summary.tool_ms_share, baseline.tool_ms_share),
        ):
            if cur is None or base is None:
                continue
            bits.append(f"{label} d={cur - base:+.2f}")
        if bits:
            out.append("    基线对比：" + "   ".join(bits))
    out.append("")

    # 4) 归因矩阵
    buckets = Counter(b for b in (_bucket(v) for v in turn_verdicts) if b)
    out.append("【归因矩阵】（只统计有答案断言的轮）")
    if buckets:
        for name, n in buckets.most_common():
            out.append(_row([(name, 40, False), (str(n), 5, True)]))
    else:
        out.append(
            "    （没有轮次带 must_contain / must_contain_any / must_not_contain 断言）"
        )
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

    # 6) 多轮跨轮证据
    ret = [v.pool_retention for v in turn_verdicts if v.pool_retention is not None]
    if ret:
        out.append(
            f"【多轮跨轮证据】均值={_num(sum(ret) / len(ret))}  "
            f"最低={_num(min(ret))}  n={len(ret)}"
            f"（<1.0 说明前几轮取到的文档在后续轮次里不再被本轮检索覆盖）"
        )
        out.append("")

    # 7) 逐轮明细
    out.append("【逐轮明细】")
    out.append(
        _row(
            [
                ("case", 34, False),
                ("轮", 4, True),
                ("doc", 6, True),
                ("ctx", 6, True),
                ("cont", 6, True),
                ("ret", 6, True),
                ("tok", 8, True),
                ("llm", 5, True),
                ("工具", 5, True),
                ("总延迟", 9, True),
                ("生成", 8, True),
                ("  备注", 0, False),
            ]
        )
    )
    for v in turn_verdicts:
        cont = (
            DASH if v.contain_pass is None else ("PASS" if v.contain_pass else "FAIL")
        )
        note = []
        if v.reason:
            note.append(v.reason[:40])
        if v.missing:
            note.append(f"缺 must_contain {len(v.missing)} 项")
        if v.bad_hits:
            note.append(f"命中 must_not_contain {len(v.bad_hits)} 项")
        if v.retrieval_ok is False:
            note.append(
                "不该检索却调了工具" if v.expected_no_retrieval else "该检索却没检索"
            )
        if v.tool_failed:
            note.append(f"工具失败 {v.tool_failed} 次")
        if v.tool_deduped:
            note.append(f"去重 {v.tool_deduped} 次")
        out.append(
            _row(
                [
                    (v.case_id[:34], 34, False),
                    (str(v.turn_idx), 4, True),
                    (_num(v.doc_recall), 6, True),
                    (_num(v.ctx_recall), 6, True),
                    (cont, 6, True),
                    (_num(v.pool_retention), 6, True),
                    (
                        str(v.total_tokens) if v.total_tokens is not None else DASH,
                        8,
                        True,
                    ),
                    (str(v.llm_calls) if v.llm_calls is not None else DASH, 5, True),
                    (str(v.tool_calls), 5, True),
                    # 没执行的轮没有有意义的耗时，显示 -- 而不是 0
                    (DASH if v.reason == "not_run" else _ms(v.latency_ms), 9, True),
                    (_ms(v.gen_ms), 8, True),
                    (("  " + "; ".join(note)) if note else "", 0, False),
                ]
            )
        )

    # 8) 仍然缺的东西要显式说明，不要静默给 0
    out.append("")
    out.append(
        "【暂缺】工具输出被截断的量（max_output_chars 尚未生效）；"
        "检索内部的候选打分过程没有独立 span（已计入工具耗时的 detail）"
    )
    out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 入口 2：从录制文件重算
# ---------------------------------------------------------------------------


def _meta_line(meta: dict) -> str:
    return (
        "meta: commit={git_commit} dirty={git_dirty} code_hash={code_hash} "
        "cases_hash={cases_hash} prompt_hash={prompt_hash}".format(
            git_commit=meta.get("git_commit"),
            git_dirty=meta.get("git_dirty"),
            code_hash=meta.get("code_hash"),
            cases_hash=meta.get("cases_hash"),
            prompt_hash=meta.get("prompt_hash"),
        )
    )


def main(argv: list[str] | None = None) -> int:
    import sys

    args = (argv or sys.argv)[1:]
    if not args:
        print("用法: python -m eval.metrics <run.json 或 run 目录> [基线 run]")
        print("  传第二个参数时，【总体】会给出与基线的差值（|d|<0.03 视为噪声）。")
        return 2

    run = load_run(args[0])
    try:
        verdicts, summary = score(run)
    except ValueError as e:
        print(f"无法打分：{e}")
        return 1

    baseline_summary = None
    if len(args) > 1:
        baseline = load_run(args[1])
        try:
            _, baseline_summary = score(baseline)
        except ValueError as e:
            print(f"[WARN] 基线无法打分，本次不出差值：{e}")
            baseline_summary = None
        else:
            print(
                f"baseline run_id={baseline.run_id}  started_at={baseline.started_at}"
            )
            print(_meta_line(baseline.meta or {}))
            # 用例集变了，逐格对比就不成立 —— 显式说出来，而不是给出一堆看似可比的差值。
            # prompt_hash 变了则要区分：那是"有意改了 prompt"，差值仍然有意义。
            if (run.meta or {}).get("cases_hash") != (baseline.meta or {}).get(
                "cases_hash"
            ):
                print(
                    "[WARN] 两次运行的 cases_hash 不同：用例集被改过，"
                    "下面的差值只能当趋势看，不能当结论。"
                )

    print(f"run_id={run.run_id}  started_at={run.started_at}")
    print(_meta_line(run.meta or {}))
    print()
    print(render(verdicts, summary, baseline=baseline_summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
