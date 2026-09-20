"""语料反向生成评测集（生成器；产物提交到 eval/case/*.jsonl）。

为什么不手写
------------
50 条用例里要嵌几百个断言字符串（doc_id、片段锚点、must_contain）。手写错得
**看不出来**：
- doc_id 拼错（`skill_arch-mage-fire-poison` 少一个结尾连字符 → 用例永久红）；
- 锚点没有区分度（`HEXA Enhancements` 在 53 篇里都有 → 用例永久绿）；
- 断言词出现在问题里（问「Psychic Shockwave 的数值」又断言答案含
  "Psychic Shockwave" → 断言自证成立，什么都没测）。

所以把这三件事机器化：
1. **专属字符串提取**：只保留"出现在目标文档、且不出现在其他文档"的片段；
2. **可达性**：每条断言都必须能在期望文档里找到（否则永远不可能通过）；
3. **非自证**：断言词不得出现在问题里。

用法：
    python -m eval.gen_cases      # 生成 + 自检
    python -m eval.case           # 再用评测集校验器复核
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from backend.knowledge.chunker import FRONT_MATTER_RE
from backend.knowledge.knowledge import KNOWLEDGE_CATALOG as CATALOG, get_knowledge
from eval.case import anchor_report, build_chunk_domain, norm

CASE_DIR = Path("eval/case")

# 职业文档里 53 篇一致的二级标题（只用于问题模板；**不能**当锚点：每篇都有）
SEC_BOOST = "5 转强化核心（Boost Node）"
SEC_HEXA_ENH = "6 转 HEXA 强化核心（HEXA Enhancements）"
SEC_MASTERY_NODE = "6 转精通核心分组（Mastery Node）"
SEC_CLASS_V = "5 转职业专属 V 技能（Class-Specific V Skills）"
SEC_HYPER = "Hyper 被动强化（Passive Skill Boost）"
SEC_SHARED_HEXA = "6 转通用 HEXA 技能（Shared HEXA Skills）"


# ---------------------------------------------------------------------------
# 基础原语
# ---------------------------------------------------------------------------


def _body(doc_id: str) -> str:
    raw = get_knowledge(doc_id) or ""
    m = FRONT_MATTER_RE.match(raw)
    return m.group(2) if m else raw


def _clean(line: str) -> str:
    s = line.strip()
    s = re.sub(r"^#{1,6}\s+", "", s)
    s = re.sub(r"^[-*+]\s+", "", s)
    s = s.replace("**", "").replace("`", "")
    return s.strip("| \t")


def section_body(doc_id: str, prefix: str) -> str:
    """取某个二级标题下的正文（prefix 只写标题开头，编号/后缀可变）。"""
    out, capture = [], False
    for line in _body(doc_id).splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            if capture:
                break
            capture = stripped[3:].strip().startswith(prefix)
            continue
        if capture:
            out.append(line)
    return "\n".join(out)


# 章节编号前缀：`3.2 Phase 1` / `1. 概述` / `12、联动活动`。模型答题时不会复述编号，
# 所以候选词要把编号剥掉，否则锚点/断言词会因为"多了个 3.2"而假失败。
# 注意：`15 星`、`0~30 星` 这类没有"点号编号"的不能剥。
_NUM_PREFIX = re.compile(r"^(?:\d+(?:\.\d+)+|\d+[.、])\s*")


def _strip_numbering(s: str) -> str:
    return _NUM_PREFIX.sub("", s).strip()


def _strip_decoration(s: str) -> str:
    """去掉首尾的装饰符号（表格里的 ⬢、项目符号、残留标点）。

    `⬢330 SAC` 这种词不值得断言：模型答题时不会带那个符号，而且它打不进
    GBK 控制台（会 UnicodeEncodeError）。
    """
    return re.sub(r"^[^\w\u4e00-\u9fff]+|[^\w\u4e00-\u9fff）)]+$", "", s).strip()


def _candidates(text: str) -> list[tuple[str, int]]:
    """抽候选断言词，返回 (词, 优先级)。

    优先级很关键：`### 技能名` 才是"名字"，而描述行里切出来的
    `Final Damage: +10%.` 虽然独占且含数字（打分很高），却根本不是技能名 ——
    拿它去断言"包含哪些技能"是错的。所以名字永远排在描述片段之前：
        0 = 小节标题（技能名）；1 = 列表/表格/分号分组切出来的词
    """
    found: list[tuple[str, int]] = []
    for line in text.splitlines():
        cleaned = _clean(line)
        if not cleaned:
            continue
        if line.strip().startswith("### "):                 # 技能名小节
            name = _strip_decoration(_strip_numbering(
                re.sub(r"（[^）]*）\s*$", "", cleaned).strip()))
            if 2 <= len(name) <= 44:
                found.append((name, 0))
            continue
        for tok in re.split(r"[；;、,，|/。]", cleaned):
            tok = _strip_decoration(_strip_numbering(tok.strip(" -—:：()（）[]*")))
            if 4 <= len(tok) <= 44:
                found.append((tok, 1))
    seen, uniq = set(), []
    for t, p in found:
        if t not in seen:
            seen.add(t)
            uniq.append((t, p))
    return uniq


def _score(term: str) -> float:
    """"稳健性"打分：越像"名字"越靠前。

    must_contain 是拿去做**答案**的字面断言，所以短、无标点、专名最稳；
    长句子/带范围数字的片段容易被模型改写 → 扣分。
    """
    length = len(term)
    s = 0.0
    if re.search(r"[A-Za-z]", term):
        s += 2.0
        s += 3.0 if 8 <= length <= 28 else (-2.0 if length > 40 else 0.0)
    else:
        s += 3.0 if 4 <= length <= 12 else (-2.0 if length > 20 else 0.0)
    if re.search(r"\d", term):
        s += 0.5
    if re.search(r"[：:（）()~＝=。.]", term):        # 片段/句子而非名字
        s -= 2.5
    return s + length / 50


def reachable(term: str, docs: list[str], domain: dict[str, str]) -> bool:
    return any(norm(term) in domain.get(d, "") for d in docs)


def exclusive(term: str, docs: list[str], domain: dict[str, str]) -> bool:
    """区分度：只出现在 docs 里（等价于 anchor_report 判 STRONG）。"""
    return anchor_report(term, docs, domain)["status"] == "STRONG"


def in_question(term: str, questions: list[str]) -> bool:
    """断言词出现在问题里 → 自证成立，必须排除。"""
    n = norm(term)
    return any(n and n in norm(q) for q in questions)


def pick_terms(cands: list[tuple[str, int]], docs: list[str], questions: list[str],
               domain: dict[str, str], *, need: int, hints: list[str] | None = None,
               max_len: int = 30) -> list[str]:
    """挑 must_contain_any 的词。

    注意：**不需要独占**（must_contain 是关于答案的断言，不是检索证据标记），
    只需要「在期望文档里存在」+「不出现在问题里」+「像名字」。
    手工 hints 优先（人挑的词最稳）。
    """
    ranked = [(t, -1) for t in (hints or [])] + list(cands)
    ranked.sort(key=lambda tp: (tp[1], -_score(tp[0])))
    picked: list[str] = []
    for term, _prio in ranked:
        if len(picked) >= need:
            break
        if len(term) > max_len:
            continue
        if in_question(term, questions) or not reachable(term, docs, domain):
            continue
        if any(norm(term) in norm(p) or norm(p) in norm(term) for p in picked):
            continue
        picked.append(term)
    return picked


def pick_anchor(cands: list[tuple[str, int]], docs: list[str], questions: list[str],
                domain: dict[str, str], hints: list[str] | None = None) -> str | None:
    """挑 expect_context_contains 的锚点。

    与 must_contain 相反：**必须独占**（只出现在期望文档里），否则"命中"证明不了
    任何事。长度上偏好具体一些（10~40 字符）。
    """
    ranked = [(t, -1) for t in (hints or [])] + list(cands)
    ranked.sort(key=lambda tp: (tp[1], (0 if 10 <= len(tp[0]) <= 40 else 1), -_score(tp[0])))
    for term, _prio in ranked:
        if not (6 <= len(term) <= 44):
            continue
        if in_question(term, questions) or not reachable(term, docs, domain):
            continue
        if exclusive(term, docs, domain):
            return term
    return None


def cn_name(doc_id: str) -> str:
    """'阿黛尔 Adele 技能数据' -> '阿黛尔'"""
    for tok in re.split(r"[\s（(]+", CATALOG[doc_id]["title"]):
        if re.search(r"[\u4e00-\u9fff]", tok):
            return tok
    return CATALOG[doc_id]["title"]


# ---------------------------------------------------------------------------
# 装配
# ---------------------------------------------------------------------------


@dataclass
class Built:
    id: str
    category: str
    tags: list[str]
    turns: list[str]
    per_turn: list[dict] = field(default_factory=list)
    notes: str = ""
    source: str = ""

    def to_dict(self) -> dict:
        d: dict = {"id": self.id, "category": self.category, "tags": self.tags, "turns": self.turns}
        if len(self.turns) == 1:
            d.update({k: v for k, v in self.per_turn[0].items() if v})
        else:
            d["expect_per_turn"] = self.per_turn
        if self.notes:
            d["notes"] = self.notes
        if self.source:
            d["source"] = self.source
        return d


class Builder:
    def __init__(self) -> None:
        self.domain = build_chunk_domain()
        self.cases: list[Built] = []
        self.problems: list[str] = []

    # ————————————————— 内部：为一个问题生成"该轮的期望" —————————————————
    def _turn(self, docs: list[str], question: str, *, section: str | None = None,
              section_doc: str | None = None, term_hints: list[str] | None = None,
              need: int = 3, must_have: list[str] | None = None, any_of: bool = False,
              extra: dict | None = None) -> dict:
        # docs 是"允许出现"的文档集合（多轮时含前几轮累积的池）；
        # 章节正文必须从**本轮目标文档**里取，不能取 docs[0]（多轮时会取错）。
        src = section_doc or docs[0]
        text = _body(src)
        if section:
            got = section_body(src, section)
            if got.strip():
                text = got
            else:
                self.problems.append(f"章节未命中，退回全文：{src} / {section}")
        pool = _candidates(text)
        terms = pick_terms(pool, docs, [question], self.domain, need=need, hints=term_hints)
        anchor = pick_anchor(pool, docs, [question], self.domain,
                             hints=list(term_hints or []) + terms)

        turn: dict = {}
        turn["expect_doc_ids_any" if any_of else "expect_doc_ids"] = list(docs)
        if anchor:
            turn["expect_context_contains"] = [anchor]
            # 锚点同时进 must_contain_any：同一串做双域断言，诊断最干净
            # （证据到场但答案没写 → 纯生成问题，排除了"锚点选得不好"的干扰）
            if anchor not in terms:
                terms = [anchor] + terms[: max(0, need - 1)]
        if terms:
            turn["must_contain_any"] = terms
        else:
            self.problems.append(f"抽不到断言词：{docs} / {question[:34]}")

        if must_have:
            ok = [t for t in must_have
                  if reachable(t, docs, self.domain) and not in_question(t, [question])]
            for t in must_have:
                if t not in ok:
                    self.problems.append(
                        f"must_contain 被丢弃（不可达或出现在问题里）：{t!r} @ {question[:30]}")
            if ok:
                turn["must_contain"] = ok
        if extra:
            turn.update(extra)
        return turn

    # ————————————————— 对外 API —————————————————
    def single(self, cid: str, category: str, tags: list[str], question: str, docs: list[str],
               *, section: str | None = None, notes: str = "", source: str = "语料反向生成",
               **kw) -> None:
        turn = self._turn(docs, question, section=section, **kw)
        self.cases.append(Built(cid, category, tags, [question], [turn], notes, source))

    def multi(self, cid: str, questions: list[str], docs_per_turn: list[list[str]],
              tags: list[str], notes: str = "",
              sections: list[str | None] | None = None,
              last_not_contain: list[str] | None = None) -> None:
        sections = sections or [None] * len(questions)
        per_turn: list[dict] = []
        for idx, (q, docs, sec) in enumerate(zip(questions, docs_per_turn, sections)):
            allowed = sorted({d for ds in docs_per_turn[: idx + 1] for d in ds})
            per_turn.append(self._turn(allowed, q, section=sec, section_doc=docs[0]))
        if last_not_contain:
            per_turn[-1]["must_not_contain"] = last_not_contain
        self.cases.append(Built(cid, "multi_turn", tags, questions, per_turn, notes,
                                "语料反向生成"))

    def raw(self, built: Built) -> None:
        self.cases.append(built)


# ---------------------------------------------------------------------------
# 各类用例
# ---------------------------------------------------------------------------

# 覆盖不同职业群、不同章节：考"文档选择"能否从 53 篇里选对。
# 问题里的职业名用 {cn} 占位 —— 由 meta 标题取出，避免手写错职业译名。
JOB_NODE_PLAN = [
    ("skill_adele", SEC_HEXA_ENH, "{cn}的 6 转 HEXA 强化核心包含哪些技能？"),
    ("skill_adele", SEC_MASTERY_NODE, "{cn}的 6 转精通核心分成几组？各组有哪些技能？"),
    ("skill_kinesis", SEC_BOOST, "{cn}的 5 转强化核心包含哪些技能？"),
    ("skill_hero", SEC_CLASS_V, "{cn}的 5 转职业专属 V 技能有哪些？"),
    ("skill_paladin", SEC_HYPER, "{cn}的 Hyper 被动强化有哪些？"),
    ("skill_dark-knight", SEC_SHARED_HEXA, "{cn}的 6 转通用 HEXA 技能有哪些？"),
    ("skill_bishop", SEC_HEXA_ENH, "{cn}的 6 转 HEXA 强化核心包含哪些技能？"),
    ("skill_night-lord", SEC_MASTERY_NODE, "{cn}的 6 转精通核心分成几组？"),
    ("skill_bow-master", SEC_BOOST, "{cn}的 5 转强化核心包含哪些技能？"),
    ("skill_shadower", SEC_CLASS_V, "{cn}的 5 转职业专属 V 技能有哪些？"),
    ("skill_phantom", SEC_HEXA_ENH, "{cn}的 6 转 HEXA 强化核心包含哪些技能？"),
    ("skill_luminous", SEC_MASTERY_NODE, "{cn}的 6 转精通核心分成几组？"),
    ("skill_evan", SEC_HYPER, "{cn}的 Hyper 被动强化有哪些？"),
    ("skill_mercedes", SEC_BOOST, "{cn}的 5 转强化核心包含哪些技能？"),
    ("skill_aran", SEC_CLASS_V, "{cn}的 5 转职业专属 V 技能有哪些？"),
    ("skill_kaiser", SEC_HEXA_ENH, "{cn}的 6 转 HEXA 强化核心包含哪些技能？"),
    ("skill_zero", SEC_MASTERY_NODE, "{cn}的 6 转精通核心分成几组？"),
    ("skill_khali", SEC_SHARED_HEXA, "{cn}的 6 转通用 HEXA 技能有哪些？"),
    ("skill_ark", SEC_BOOST, "{cn}的 5 转强化核心包含哪些技能？"),
    ("skill_hoyoung", SEC_CLASS_V, "{cn}的 5 转职业专属 V 技能有哪些？"),
    ("skill_mechanic", SEC_HEXA_ENH, "{cn}的 6 转 HEXA 强化核心包含哪些技能？"),
    ("skill_battle-mage", SEC_MASTERY_NODE, "{cn}的 6 转精通核心分成几组？"),
    ("skill_cadena", SEC_SHARED_HEXA, "{cn}的 6 转通用 HEXA 技能有哪些？"),
    ("skill_pathfinder", SEC_BOOST, "{cn}的 5 转强化核心包含哪些技能？"),
]

# 章节 -> 短码，用于生成**稳定**的用例 id（不能用 hash()：字符串 hash 每进程随机）
SECTION_CODE = {
    SEC_BOOST: "boost",
    SEC_HEXA_ENH: "hexa",
    SEC_MASTERY_NODE: "mastery",
    SEC_CLASS_V: "vskill",
    SEC_HYPER: "hyper",
    SEC_SHARED_HEXA: "sharedhexa",
}

# 系统/机制：问题里不出现断言词，断言从文档对应章节里抽
SYSTEM_PLAN = [
    ("system-starforce-cap-01", "星之力强化的星级上限是多少？普通装备和卓越装备有区别吗？",
     "starforce", "2. 星级上限", ["30", "卓越"]),
    ("system-starforce-cost-01", "星之力强化的费用能打折吗？有哪些折扣来源？",
     "starforce", "7. 强化费用与折扣", ["周日枫叶", "MVP"]),
    ("system-starforce-destroy-01", "星之力强化失败时装备会被破坏吗？破坏后怎么恢复？",
     "starforce", "8. 破坏与恢复", ["破坏率"]),
    ("system-flame-nowash-01", "哪些装备不能使用火花（附加选项）？",
     "flame-additional-options", "二、GMS 基础规则", ["副武器", "纹章", "图腾"]),
    ("system-flame-tiers-01", "火花道具（Rebirth Flame）有哪些家族？各有什么特点？",
     "flame-additional-options", "四、火花道具", ["Powerful", "Eternal", "Abyssal"]),
    ("system-damage-range-01", "伤害范围（Damage Range）是怎么计算的？",
     "damage-formulas", "三、伤害范围", ["武器倍率", "熟练度"]),
    ("system-damage-final-01", "最终伤害、BOSS 伤害和伤害% 之间是加算还是乘算？",
     "damage-formulas", "二、伤害核心组件", ["最终伤害", "BOSS"]),
    ("system-utility-statusres-01", "异常状态抗性是按什么公式计算的？有上限吗？",
     "utility-related", "一、异常状态抗性", ["log"]),
    ("system-utility-attackspeed-01", "攻击速度有上限吗？攻速加成是怎么换算的？",
     "utility-related", "三、攻击速度", ["软上限", "硬上限", "攻速"]),
    ("system-utility-cooldown-01", "技能冷却缩减（CDR）最多能减到多少？有下限吗？",
     "utility-related", "六、技能冷却", ["冷却", "CDR"]),
    ("system-upgrade-protect-01", "强化到 15 星以上失败会掉星吗？保护装备的费用是多少倍？",
     "upgrade-rule", None, ["保护", "倍"]),
]

# boss：两个 BOSS 文档的章节结构一致 → 锚点必须来自正文内容
BOSS_PLAN = [
    ("boss-seren-phase-01", "塞伦有哪些核心机制？", "seren", "三、核心机制"),
    ("boss-seren-entry-01", "塞伦各难度的入场条件是什么？", "seren", "一、入场信息"),
    ("boss-kalos-entry-01", "卡洛斯的入场条件是什么？有哪些难度？", "kalos", "一、入场信息"),
    ("boss-kalos-phase-01", "卡洛斯有哪些核心机制？", "kalos", "三、核心机制"),
]

PATCH_PLAN = [
    ("patch-271-collab-01", "V.271 版本联动了哪个作品？", "patch-271", "十二、联动活动"),
    ("patch-271-updates-01", "V.271 更新了哪些主要内容？", "patch-271", "二、新增内容"),
]


def build_job(b: Builder) -> None:
    for doc, sec, template in JOB_NODE_PLAN:
        q = template.format(cn=cn_name(doc))
        b.single(
            cid=f"job-{doc.replace('skill_', '')}-{SECTION_CODE[sec]}",
            category="job_skill",
            tags=["easy"],
            question=q,
            docs=[doc],
            section=sec,
            need=4,
            notes=f"考文档选择（从 53 篇职业文档里选中 {doc}）+ 该章节片段是否进预算",
        )


def build_system(b: Builder) -> None:
    for cid, q, doc, sec, hints in SYSTEM_PLAN:
        b.single(cid=cid, category="system", tags=["medium"], question=q, docs=[doc],
                 section=sec, term_hints=hints, need=3,
                 must_have=[h for h in hints if re.search(r"\d", h)][:2] or None,
                 notes="机制/数值类：must_contain 用文档里的原词或数字，考生成是否取到正确片段")


def build_boss(b: Builder) -> None:
    for cid, q, doc, sec in BOSS_PLAN:
        b.single(cid=cid, category="boss", tags=["medium"], question=q, docs=[doc],
                 section=sec, need=3,
                 notes="BOSS 文档只有 2 篇：考是否能和另一篇区分开")


def build_patch(b: Builder) -> None:
    for cid, q, doc, sec in PATCH_PLAN:
        b.single(cid=cid, category="patch_note", tags=["version-sensitive"],
                 question=q, docs=[doc], section=sec, need=2,
                 notes="版本相关：改版后需集中复查")


def build_alias(b: Builder) -> None:
    """别名 / 错别字 / 中英混写：考检索的鲁棒性（预期可能失败，标 hard）。"""
    plan = [
        ("alias-adele-en-01", "Adele 的 6 转 HEXA 强化核心包含哪些技能？", "skill_adele", SEC_HEXA_ENH),
        ("alias-kinesis-en-01", "Kinesis 的 5 转强化核心包含哪些技能？", "skill_kinesis", SEC_BOOST),
        ("alias-adele-typo-01", "艾黛尔的 6 转精通核心分成几组？", "skill_adele", SEC_MASTERY_NODE),
        ("alias-seren-typo-01", "赛伦有哪些核心机制？", "seren", "三、核心机制"),
        ("alias-fpmage-01", "火毒法师的 6 转 HEXA 强化核心包含哪些技能？",
         "skill_arch-mage-fire-poison-", SEC_HEXA_ENH),
        ("alias-ilmage-01", "冰雷法师的 5 转职业专属 V 技能有哪些？",
         "skill_arch-mage-ice-lightning-", SEC_CLASS_V),
    ]
    for cid, q, doc, sec in plan:
        b.single(cid=cid, category="alias_typo", tags=["hard"], question=q, docs=[doc],
                 section=sec, need=3,
                 notes="别名/错别字：同一事实的另一种写法。失败说明检索需要别名支持，不代表答案错")


def build_ambiguous(b: Builder) -> None:
    """歧义：一词语料里对应两篇文档，任一命中即算对。"""
    both = ["skill_arch-mage-fire-poison-", "skill_arch-mage-ice-lightning-"]
    b.single(cid="ambig-archmage-01", category="ambiguous", tags=["medium"],
             question="魔导师的 6 转 HEXA 强化核心包含哪些技能？",
             docs=both, section=SEC_HEXA_ENH, need=3, any_of=True,
             notes="「魔导师」在语料里对应火毒/冰雷两篇，任一命中即算对（expect_doc_ids_any）")
    b.single(cid="ambig-archmage-both-01", category="ambiguous", tags=["hard"],
             question="火毒魔导师和冰雷魔导师的 HEXA 专属技能分别是什么？",
             docs=both, section=None, need=4,
             notes="要求两篇都取到（expect_doc_ids 是两篇），考多文档召回")


def build_multi(b: Builder) -> None:
    b.multi(
        cid="multi-seren-adele-01",
        questions=["塞伦有哪些核心机制？", "那阿黛尔该怎么应对？"],
        docs_per_turn=[["seren"], ["skill_adele"]],
        sections=["三、核心机制", SEC_HEXA_ENH],
        tags=["hard"],
        notes="第 2 轮的 allowed 集含第 1 轮的 seren → 直接考知识池跨轮保留",
    )
    b.multi(
        cid="multi-kinesis-numeric-01",
        questions=["超能力者的 Psychic Shockwave 是什么技能？",
                   "它的强化被动在 Lv.30 提供多少最终伤害？"],
        docs_per_turn=[["skill_kinesis"], ["skill_kinesis"]],
        sections=[SEC_CLASS_V, SEC_HYPER],
        tags=["medium"],
        notes="同一文档内的追问：考指代（它）+ 数值片段召回",
    )
    b.multi(
        cid="multi-starforce-protect-01",
        questions=["星之力强化失败时装备会被破坏吗？", "那有什么办法保护装备？"],
        docs_per_turn=[["starforce"], ["starforce"]],
        sections=["4. 普通装备强化概率表", None],
        tags=["medium"],
        notes="机制追问：第 2 轮需要第 1 轮的上下文（保护规则在别的章节）",
    )
    b.multi(
        cid="multi-archmage-compare-01",
        questions=["火毒魔导师和冰雷魔导师的 HEXA 专属技能分别是什么？",
                   "那他们通用的 6 转 HEXA 技能一样吗？"],
        docs_per_turn=[["skill_arch-mage-fire-poison-", "skill_arch-mage-ice-lightning-"],
                       ["skill_arch-mage-fire-poison-", "skill_arch-mage-ice-lightning-"]],
        sections=[None, SEC_SHARED_HEXA],
        tags=["hard"],
        notes="多文档 + 追问：考两篇文档与通用节是否都在场",
    )
    b.multi(
        cid="multi-seren-kalos-01",
        questions=["塞伦各难度的入场条件是什么？", "卡洛斯呢？"],
        docs_per_turn=[["seren"], ["kalos"]],
        sections=["一、入场信息", "一、入场信息"],
        tags=["hard"],
        notes="指代（卡洛斯呢？）：第 2 轮换文档，考池里同时有两篇",
    )
    b.multi(
        cid="multi-flame-scissors-01",
        questions=["哪些装备不能使用火花？", "使用火花会影响交易吗？"],
        docs_per_turn=[["flame-additional-options"], ["flame-additional-options"]],
        sections=["二、GMS 基础规则", "四、火花道具"],
        tags=["medium"],
        notes="同文档跨章节追问：考片段级召回是否跟着问题走",
    )


def build_negative(b: Builder) -> None:
    """负样本：唯一能测出"幻觉倾向"的一类。目前语料里 0 条，必须补上。"""
    absent = ["没有", "未收录", "不包含", "无法", "不知道", "未涵盖", "没有相关"]

    def neg(cid: str, q: str, tags: list[str], notes: str,
            not_contain: list[str] | None = None, no_retrieval: bool = False,
            must_any: list[str] | None = None) -> None:
        turn = {
            "must_contain_any": must_any or absent,
            "must_not_contain": not_contain or [],
            "expect_no_retrieval": no_retrieval,
        }
        b.raw(Built(cid, "negative", tags, [q], [turn], notes, "手工构造"))

    neg("neg-out-of-scope-crawler-01",
        "帮我写一个爬取 maplestorywiki 网站所有技能页的 Python 脚本。",
        ["easy"],
        "超范围：不该检索知识库、不该产出代码",
        not_contain=["import ", "def ", "requests.get"], no_retrieval=True)
    neg("neg-out-of-scope-game-01",
        "推荐几款和冒险岛类似的横版游戏。",
        ["easy"], "超范围：与知识库无关",
        not_contain=["MapleStory 2", "DNF", "地下城与勇士"], no_retrieval=True)
    neg("neg-out-of-scope-debug-01",
        "这段 Python 代码报 KeyError 了，帮我看看哪里错了：d = {}; print(d['a'])",
        ["easy"], "超范围：通用编程求助",
        not_contain=["d.get(", "defaultdict"], no_retrieval=True)
    neg("neg-mobile-differs-01",
        "手游 MapleStory M 里阿黛尔的技能和端游一样吗？",
        ["hard"], "知识库只有 GMS 端游数据（技能文档来源是 maplestorywiki 端游页）",
        not_contain=["一样", "完全相同", "没有区别"])
    neg("neg-discontinued-jett-01",
        "Jett 的 HEXA 强化核心包含哪些技能？",
        ["hard"], "Jett 属于停用职业，meta.md 明确说明未收录 → 必须承认没有资料",
        not_contain=["HEXA Cleave", "精通核心 1", "Core 1"])
    neg("neg-fake-skill-01",
        "阿黛尔有一个叫「星辰破碎斩」的技能吗？效果是什么？",
        ["hard"], "幻觉探针：该技能不存在，答案必须否认而不是编造效果",
        must_any=["没有", "不存在", "未收录", "没找到", "并未"])
    neg("neg-live-patch-01",
        "今天的冒险岛维护公告说了什么？下次维护是什么时候？",
        ["hard"], "时效性：知识库最大到 V.271，没有实时公告",
        must_any=["无法", "没有", "不知道", "未收录", "不能"], not_contain=["今天维护"])
    neg("neg-account-01",
        "我的账号被封了，怎么申诉？",
        ["easy"], "超范围：账号/客服问题与知识库无关",
        no_retrieval=True)
    neg("neg-price-01",
        "现在拍卖行里黑色重生之焰多少钱一个？",
        ["hard"], "时效性 + 超出语料：文档里只有历史价格区间，没有实时行情",
        must_any=["无法", "没有", "不知道", "未收录", "实时"])


def build(b: Builder) -> None:
    build_job(b)
    build_system(b)
    build_boss(b)
    build_patch(b)
    build_alias(b)
    build_ambiguous(b)
    build_multi(b)
    build_negative(b)


# ---------------------------------------------------------------------------
# 写盘 + 自检
# ---------------------------------------------------------------------------

FILE_BY_CATEGORY = {
    "job_skill": "job_skill.jsonl",
    "system": "system.jsonl",
    "boss": "boss.jsonl",
    "patch_note": "patch_note.jsonl",
    "alias_typo": "alias.jsonl",
    "ambiguous": "ambiguous.jsonl",
    "multi_turn": "multiturn.jsonl",
    "negative": "negative.jsonl",
}


def main() -> int:
    b = Builder()
    build(b)

    # ① 自检：id 唯一、断言非自证、锚点有区分度、断言可达
    seen: set[str] = set()
    for c in b.cases:
        if c.id in seen:
            b.problems.append(f"用例 id 重复：{c.id}")
        seen.add(c.id)
        for t in c.per_turn:
            for key in ("must_contain", "must_contain_any", "expect_context_contains"):
                for s in t.get(key) or []:
                    if in_question(s, c.turns):
                        b.problems.append(f"{c.id}: {key} 项出现在问题里（自证成立）：{s!r}")

    # ② 写盘（按 category 分文件）
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    for old in CASE_DIR.glob("*.yaml"):
        old.unlink()                       # 清掉历史遗留的第二份真相源
    if (CASE_DIR / "json_skill.yaml").exists():
        (CASE_DIR / "json_skill.yaml").unlink()

    by_file: dict[str, list[Built]] = {}
    for c in b.cases:
        by_file.setdefault(FILE_BY_CATEGORY[c.category], []).append(c)

    for fname, cases in sorted(by_file.items()):
        path = CASE_DIR / fname
        with open(path, "w", encoding="utf-8") as f:
            for c in cases:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    # ③ 报告
    print(f"生成 {len(b.cases)} 条用例：")
    for fname, cases in sorted(by_file.items()):
        print(f"  {fname:<20} {len(cases):>3}")
    print()
    if b.problems:
        print(f"自检发现 {len(b.problems)} 个问题：")
        for p in b.problems:
            print("  -", p)
        return 1
    print("自检通过（断言非自证、锚点有区分度、断言可达）")
    print()
    print("提示：以上文件由本生成器**整文件覆盖**。手工从真实失败里挖的用例请写到")
    print("      eval/case/regression.jsonl（生成器不碰它，load_cases 会自动加载）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
