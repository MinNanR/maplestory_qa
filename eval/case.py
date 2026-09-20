"""评测用例：加载、归一化（扁平写法 → 逐轮期望）、以及用例集自身的元校验。

为什么要"逐轮期望"
------------------
多轮用例的期望必须绑定到具体的轮次，否则同一个 expect 会被套到每一轮上：

    turns: ["塞伦有哪些阶段机制？", "那阿黛尔该怎么应对？"]
    扁平 expect_doc_ids: ["seren", "skill_adele"]

第 1 轮只会注入 seren，于是 recall 恒为 0.5 —— 那不是系统的问题，是断言套错了轮次。
（2026-09-17 那次录制的两条 multi_turn FAIL 全部由此产生。）

于是本模块把用例统一归一化成「每轮一份期望」：

- 单轮用例：继续用扁平写法（扁平字段 = 第 1 轮的期望），可读性最好；
- 多轮用例：必须写 `expect_per_turn`，逐轮给出期望。

加载后的 Case.turns 一律是 list[TurnExpect]，runner 落盘的快照也是这个形状，
metrics 因此可以直接按轮次断言，不需要再猜。
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from glob import glob
from pathlib import Path

from backend.config import settings
from backend.knowledge.knowledge import KNOWLEDGE_CATALOG, get_knowledge
from backend.knowledge.retrieval import split_document

CASE_GLOB = "eval/case/*.jsonl"

# 用例里所有"期望"字段。扁平写法与 expect_per_turn 写法共用同一批名字。
EXPECT_FIELDS = (
    "expect_doc_ids",
    "expect_doc_ids_any",
    "expect_context_contains",
    "must_contain",
    "must_contain_any",
    "must_not_contain",
)


@dataclass
class TurnExpect:
    """单轮的期望。空列表 = 该轮不对此项断言。"""

    query: str
    expect_doc_ids: list[str] = field(default_factory=list)
    expect_doc_ids_any: list[str] = field(default_factory=list)
    expect_context_contains: list[str] = field(default_factory=list)
    must_contain: list[str] = field(default_factory=list)
    must_contain_any: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    # 不该检索的轮（超范围/闲聊）：断言 did_retrieve 为 False
    expect_no_retrieval: bool = False


@dataclass
class Case:
    id: str
    category: str
    tags: list[str]
    turns: list[TurnExpect]
    notes: str = ""
    source: str = ""          # 这条用例从哪来的（真实失败文件名 / 手工构造 / 语料反向生成）


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------


def _as_str_list(value, where: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValueError(f"{where} 必须是字符串列表，实际是 {type(value).__name__}")
    return list(value)


def _build_turn(query, raw_turn: dict, where: str) -> TurnExpect:
    if not isinstance(raw_turn, dict):
        raise ValueError(f"{where} 必须是对象，实际是 {type(raw_turn).__name__}")
    known = set(EXPECT_FIELDS) | {"expect_no_retrieval"}
    unknown = set(raw_turn) - known
    if unknown:
        raise ValueError(f"{where} 含未知字段：{sorted(unknown)}（可用：{sorted(known)}）")
    return TurnExpect(
        query=query,
        **{f: _as_str_list(raw_turn.get(f), f"{where}.{f}") for f in EXPECT_FIELDS},
        expect_no_retrieval=bool(raw_turn.get("expect_no_retrieval", False)),
    )


def _normalize(raw: dict, where: str) -> Case:
    # 顶层字段白名单。pydantic dataclass 对未声明字段是「静默忽略」，
    # 于是把 must_contain 拼成 must_contains 会无声丢掉整条期望 —— 这里直接报错。
    known = {"id", "category", "tags", "turns", "notes", "source", "expect_per_turn",
             "expect_no_retrieval", *EXPECT_FIELDS}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{where} 含未知字段：{sorted(unknown)}（可用：{sorted(known)}）")

    for key in ("id", "category", "turns"):
        if key not in raw:
            raise ValueError(f"{where} 缺少必填字段 {key}")

    queries = raw["turns"]
    if not isinstance(queries, list) or not queries or not all(isinstance(q, str) for q in queries):
        raise ValueError(f"{where}.turns 必须是非空的字符串列表")

    per_turn = raw.get("expect_per_turn")
    flat_used = [f for f in EXPECT_FIELDS if raw.get(f)]

    if per_turn is not None:
        if not isinstance(per_turn, list) or len(per_turn) != len(queries):
            raise ValueError(
                f"{where}.expect_per_turn 必须与 turns 等长"
                f"（turns={len(queries)}, expect_per_turn={len(per_turn) if isinstance(per_turn, list) else 'N/A'}）"
            )
        turns = [
            _build_turn(q, t, f"{where}.expect_per_turn[{i}]") for i, (q, t) in enumerate(zip(queries, per_turn))
        ]
        if flat_used:
            # 两种写法同时在，容易误以为扁平字段会作为"所有轮的默认值"
            print(f"[WARN] {where}: 同时存在 expect_per_turn 与扁平 {flat_used}，扁平字段将被忽略")
    else:
        turns = [
            _build_turn(
                q,
                {f: raw.get(f) for f in EXPECT_FIELDS}
                | {"expect_no_retrieval": raw.get("expect_no_retrieval", False)},
                where,
            )
            for q in queries
        ]
        if len(turns) > 1:
            print(
                f"[WARN] {where}: 多轮用例使用了扁平 expect 字段，已全部套在「最后一轮」；"
                f"建议改用 expect_per_turn 逐轮声明"
            )

    return Case(
        id=raw["id"],
        category=raw["category"],
        tags=_as_str_list(raw.get("tags"), f"{where}.tags"),
        turns=turns,
        notes=raw.get("notes") or "",
        source=raw.get("source") or "",
    )


def load_cases(pattern: str = CASE_GLOB) -> list[Case]:
    """加载全部用例。结构性问题直接抛 ValueError（宁可在烧钱前失败）。"""
    cases: list[Case] = []
    seen: dict[str, str] = {}

    files = sorted(glob(pattern))
    if not files:
        raise ValueError(f"没有匹配到任何用例文件：{pattern}")

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                where = f"{path}:{line_no}"
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{where} JSON 解析失败：{e}") from e
                case = _normalize(raw, where)
                if case.id in seen:
                    raise ValueError(f"{where} 用例 id 重复：{case.id}（已在 {seen[case.id]} 出现）")
                seen[case.id] = where
                cases.append(case)

    print(f"已加载 {len(cases)} 个测试用例（{len(files)} 个文件）")
    return cases


def cases_hash(cases: list[Case]) -> str:
    """用例集指纹。写进 run meta：用例集一改，历史分数就不可直接比较。"""
    payload = json.dumps([asdict(c) for c in cases], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 元校验：先证明评测集本身没坏，再去信它的分数
# ---------------------------------------------------------------------------


def norm(s: str) -> str:
    """规范化。校验与运行时断言必须调用同一个函数，否则两边会分歧。"""
    s = unicodedata.normalize("NFKC", s or "")   # 全角→半角、（）→() 等
    return re.sub(r"\s+", "", s).casefold()      # 去掉全部空白、统一大小写


_CHUNK_DOMAIN: dict[str, str] | None = None


def build_chunk_domain(force: bool = False) -> dict[str, str]:
    """doc_id -> 该文档全部分块文本拼接并规范化（= 运行时可能被注入的内容域）。

    注意不能用原始文件全文：
      - front-matter 会被 chunker 剥掉，只在其中出现的内容永远不会被注入；
      - 片段文本会重复写入祖先标题路径，与原文不是逐字对应。
    """
    global _CHUNK_DOMAIN
    if _CHUNK_DOMAIN is not None and not force:
        return _CHUNK_DOMAIN

    domain: dict[str, str] = {}
    for doc_id, info in KNOWLEDGE_CATALOG.items():
        chunks = split_document(
            get_knowledge(doc_id) or "", doc_id, info["title"],
            min_chars=settings.chunk_min_chars,     # 必须与 retrieval 用的参数一致
            max_chars=settings.chunk_max_chars,
        )
        domain[doc_id] = norm("".join(c.text for c in chunks))
    _CHUNK_DOMAIN = domain
    return domain


def anchor_report(anchor: str, expect_docs: list[str], domain: dict[str, str]) -> dict:
    """锚点的两个结论：可达性（能不能通过）与区分度（通过了算不算数）。"""
    a = norm(anchor)
    expect = set(expect_docs)
    if not a:
        return {"status": "EMPTY", "reachable_in": [], "leaked_to": []}

    hit = {d for d, text in domain.items() if a in text}
    reachable_in = hit & expect        # 期望文档里命中的
    leaked_to = hit - expect           # 泄漏到其他文档的

    if not reachable_in:
        status = "UNREACHABLE"         # FATAL：期望文档里就没有 → 永远不可能通过
    elif leaked_to:
        status = "WEAK"                # WARN：别的文档也有 → 绿了也不能证明证据来自期望文档
    else:
        status = "STRONG"
    return {"status": status,
            "reachable_in": sorted(reachable_in),
            "leaked_to": sorted(leaked_to)}


def validate_cases(cases: list[Case]) -> list[str]:
    """返回 FATAL 问题列表（空 = 通过）。WEAK 只打印警告，不阻断。"""
    fatal: list[str] = []

    # 1) doc_id 存在性（覆盖逐轮 + any 两种字段）
    for case in cases:
        for i, turn in enumerate(case.turns):
            for field_name in ("expect_doc_ids", "expect_doc_ids_any"):
                for doc_id in getattr(turn, field_name):
                    if doc_id not in KNOWLEDGE_CATALOG:
                        fatal.append(
                            f"{case.id} turn{i}: {field_name} 引用了不存在的 doc_id: {doc_id!r}"
                        )

    # 2) 锚点可达性 + 区分度；3) must_contain 可达性
    #    注意：歧义用例只写 expect_doc_ids_any，所以比对集合要把两者并起来
    domain = build_chunk_domain()
    for case in cases:
        for i, turn in enumerate(case.turns):
            where = f"{case.id} turn{i}"
            allowed = sorted(set(turn.expect_doc_ids) | set(turn.expect_doc_ids_any))
            for anchor in turn.expect_context_contains:
                r = anchor_report(anchor, allowed, domain)
                if r["status"] in ("EMPTY", "UNREACHABLE"):
                    fatal.append(f"{where}: 锚点 {r['status']}（期望文档中不存在）: {anchor!r}")
                elif r["status"] == "WEAK":
                    leaked = r["leaked_to"]
                    print(
                        f"[WARN] {where}: 锚点无区分度，泄漏到 {len(leaked)} 篇其他文档"
                        f"（如 {', '.join(leaked[:3])}）: {anchor!r}"
                    )

            # 负样本轮（超范围/不可答）没有期望文档：断言的是"答案里不该有/该承认不知道"，
            # 没有文档可对照，跳过可达性检查（否则会把正确设计的负样本判成非法）。
            if not allowed:
                continue

            for field_name in ("must_contain", "must_contain_any"):
                for s in getattr(turn, field_name):
                    if not any(norm(s) in domain.get(d, "") for d in allowed):
                        fatal.append(
                            f"{where}: {field_name} 项在期望文档里不可达（用例永红）: {s!r}"
                        )

    return fatal


if __name__ == "__main__":
    loaded = load_cases()
    problems = validate_cases(loaded)
    if problems:
        print(f"\n用例集非法，共 {len(problems)} 个问题：")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print(f"\n用例集校验通过。cases_hash = {cases_hash(loaded)}")
    print(f"共 {len(loaded)} 个用例，{sum(len(c.turns) for c in loaded)} 轮")
