# smoke_retrieval.py  （仓库根目录，运行：.\.venv\Scripts\python.exe smoke_retrieval.py）
"""冒烟测试：知识检索层 + 两个检索工具，**不联网、不要 API key、不调 LLM**。

为什么要有这一份 —— d175e0da 那次全量录制里，retrieval_among_document（跨文档检索）
**6 次调用 6 次失败**，而 retrieval_in_document 122 次全成功。两个函数名字只差一个
单词，是被下面两条契约悄悄分岔的（都在 retrieve_chunks 的**同一行返回语句**上）：

    1. 返回形状：它返回的是 (chunk, score) 元组，而
       - 返回注解写的是 list[KnowledgeChunk]；
       - 两个调用方（工具层、会话层）都按 KnowledgeChunk 用；
       - build_knowledge_block 也只认 KnowledgeChunk。
       于是工具每次都在 build_knowledge_block 里炸：
       `'tuple' object has no attribute 'heading_path'`。
    2. 零命中兜底：第一遍是"每个文档至少一段"，那个片段可能对查询零命中
       （chunk_score 里根本没有它的条目），再用 `chunk_score[cid]` 取就直接
       KeyError —— 错误信息是一串 chunk_id，完全看不出是检索的问题。

断言口径：**同一份输入下，两个检索函数的输出形状必须一致**，且都必须能被
build_knowledge_block 与工具层直接消费。
"""

from __future__ import annotations

import asyncio
import sys
import traceback

from backend.config import settings
from backend.knowledge.chunker import KnowledgeChunk
from backend.knowledge.knowledge import KNOWLEDGE_CATALOG
from backend.knowledge.retrieval import (
    build_knowledge_block,
    retrieve_chunks,
    retrieve_chunks_in_doc,
)
from backend.tool.tools_impl.knowledge import (
    retrieval_among_document,
    retrieval_in_document,
)

# d175e0da 里真实失败的那两个文档（跨文档比较炎术士 / 冰雷法师）
ARCH_MAGE_IDS = sorted(d for d in KNOWLEDGE_CATALOG if "arch-mage" in d)
# 那次录制里实际发出去的 6 个关键词，4 个走"正常命中"、2 个走"零命中兜底"
REAL_QUERIES = ["HEXA 强化", "Boost Node", "HEXA", "Infernal Venom", "Frozen Lightning", "Common HEXA"]

CASES: list[tuple[str, object]] = []


def case(desc: str):
    def deco(fn):
        CASES.append((desc, fn))
        return fn

    return deco


def two_doc_ids() -> list[str]:
    assert len(ARCH_MAGE_IDS) >= 2, f"知识库里找不到两个 arch-mage 文档：{ARCH_MAGE_IDS}"
    return ARCH_MAGE_IDS[:2]


@case("1 形状契约：两个检索函数都返回 list[KnowledgeChunk]（不能有元组混进来）")
async def case_shape():
    doc_ids = two_doc_ids()
    multi = retrieve_chunks("HEXA", doc_ids)
    single = retrieve_chunks_in_doc(["HEXA"], doc_ids[0])

    for label, chunks in (("retrieve_chunks", multi), ("retrieve_chunks_in_doc", single)):
        assert chunks, f"{label} 应有命中"
        bad = [type(c).__name__ for c in chunks if not isinstance(c, KnowledgeChunk)]
        assert not bad, f"{label} 返回了非 KnowledgeChunk 元素：{bad}"

    # 两个函数都要能被注入文本的构造函数直接消费（这里就是当初炸掉的那一步）
    assert build_knowledge_block(multi), "retrieve_chunks 的结果喂不进 build_knowledge_block"
    assert build_knowledge_block(single)


@case("2 零命中兜底：第一遍的'每文档至少一段'不因该段无分数而 KeyError")
async def case_zero_score_fallback():
    doc_ids = two_doc_ids()
    # 这个词元在索引里根本不存在 → 所有片段分数为 0，
    # 而第一遍仍会把每个文档的首片段兜进来（正是旧代码 [cid] 取分的位置）
    chunks = retrieve_chunks("zzz_no_such_token_zzz", doc_ids)

    # 零命中时两遍都会选到"分数为 0"的片段：第一遍保底每文档一段，
    # 第二遍把剩余预算也填满 —— 所以这里断言的是"每篇都被覆盖"，不是"只有 N 段"。
    assert len(chunks) >= len(doc_ids), f"至少要覆盖每个传入文档：{len(chunks)} 段"
    covered = {c.doc_id for c in chunks}
    assert covered == set(doc_ids), f"兜底片段应覆盖每个传入文档：{covered}"
    assert build_knowledge_block(chunks), "兜底片段也要能拼成注入文本"


@case("3 真实参数回归：d175e0da 里失败的 6 次调用现在都要成功")
async def case_real_world_queries():
    doc_ids = two_doc_ids()
    for query in REAL_QUERIES:
        chunks = retrieve_chunks(query, doc_ids)
        assert chunks, f"{query!r} 应有命中"
        block = build_knowledge_block(chunks)
        assert block and "片段 1/" in block, f"{query!r} 的注入文本不对"

        out = retrieval_among_document(doc_ids=doc_ids, retrieval_text=query)
        assert out.text and not out.text.startswith("未能找到"), \
            f"{query!r} 工具层仍失败：{out.text[:120]}"
        assert out.structured.get("chunk_ids"), f"{query!r} 没回传 chunk_ids（评测的召回算不出来）"


@case("4 工具层契约：两个检索工具都返回 text + structured.chunk_ids")
async def case_tool_contract():
    doc_ids = two_doc_ids()

    multi = retrieval_among_document(doc_ids=doc_ids, retrieval_text="HEXA")
    assert multi.structured["chunk_ids"], multi
    assert all("#" in cid for cid in multi.structured["chunk_ids"]), multi.structured

    single = retrieval_in_document(doc_id=doc_ids[0], retrieval_texts=["HEXA", "Boost"])
    assert single.structured["chunk_ids"], single
    assert all(cid.startswith(doc_ids[0]) for cid in single.structured["chunk_ids"]), \
        "retrieval_in_document 只许返回该文档的片段"

    # chunk_ids 只给观测/评测用，不许混进给模型看的正文
    assert "#0" not in multi.text.split("【")[0], "正文里混进了 chunk_id"


@case("5 预算与幻觉 id：不超预算、不抛异常")
async def case_budget_and_hallucination():
    doc_ids = two_doc_ids()

    chunks = retrieve_chunks("HEXA", doc_ids, max_chars=3000)
    assert chunks, "预算缩小后仍应有命中"
    total = sum(len(c.text) for c in chunks)
    assert total <= 3000, f"超出字符预算：{total}"

    # 幻觉 id：全部无效时返回空、部分无效时静默丢掉 —— 两种都不许抛异常
    assert retrieve_chunks("HEXA", ["no-such-doc-id"]) == []
    assert retrieve_chunks("HEXA", ["no-such-doc-id", doc_ids[0]]), "有效 id 仍应能检索"
    # 只断言"不炸"，**不**钉死文案：下面"建议改进"里那条（把被忽略的无效 id
    # 明确告诉模型）会改动这句话，钉死文案会让改进本身把测试弄红。
    # 见文件末尾「已知遗留」。
    out = retrieval_among_document(doc_ids=["no-such-doc-id"], retrieval_text="HEXA")
    assert isinstance(out.text, str), out


async def main() -> int:
    saved = settings.knowledge_max_chars
    failures: list[tuple[str, BaseException]] = []
    try:
        for desc, fn in CASES:
            try:
                await fn()
            except BaseException as e:  # noqa: BLE001 - 逐用例收集，跑完再汇总
                failures.append((desc, e))
                print(f"[FAIL] {desc}\n       {type(e).__name__}: {e}")
                if "--trace" in sys.argv:
                    traceback.print_exc()
            else:
                print(f"[ ok ] {desc}")
    finally:
        settings.knowledge_max_chars = saved

    print()
    if failures:
        print(f"smoke FAILED — {len(failures)}/{len(CASES)} 个用例没过：")
        for desc, e in failures:
            print(f"  - {desc}  ({type(e).__name__})")
        return 1
    print(f"smoke ok — {len(CASES)}/{len(CASES)} 个用例全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))


# ---------------------------------------------------------------------------
# 已知遗留（这次没改，需要你决定）
# ---------------------------------------------------------------------------
# doc_ids 里的**无效 id 被静默丢弃**：`retrieve_chunks` 里
# `valid = [d for d in doc_ids if d in _CHUNK_OF_DOC]` 不做任何回报。
# 模型若传 2 个 id、其中 1 个写错（比如漏了结尾的 `-`），它会拿到只含 1 篇的片段，
# 却以为两篇都查过了 —— 跨文档比较题会据此给出错误结论，而且录制里只看到
# "调用成功"。建议在返回文本里显式写一句"以下 doc_id 不存在，已忽略：…"，
# 让模型有机会纠正 id（这属于工具契约变更，故未擅自实施）。
