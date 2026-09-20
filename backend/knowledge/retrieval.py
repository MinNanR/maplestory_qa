"""
知识库本地检索（backend/knowledge/retrieval.py）

零外部依赖的 BM25-风格检索，服务于两处上下文瘦身：
1. retrieve_documents()：意图解析前，从目录条目（title/description/keyword）
   中筛出少量候选 knowledge_id，替代把整份 build_catalog() 塞进提示词；
2. retrieve_chunks()：回答前，在指定文档内按问题截取相关片段，
   替代把文档全文注入上下文；并保证「池内每个文档至少返回其最高分片段」，
   从而让多轮对话中历史轮次解析出的知识（如塞伦机制）在后续追问中仍然在场。

索引为模块级缓存，以知识库文件 mtime 指纹判断变更后自动重建
（新增/修改知识文档后无需重启，下一次调用即生效）。

用法：
    python -m backend.knowledge.retrieval --query "虎影 Bravado 技能效果"
    python -m backend.knowledge.retrieval --query "卡洛斯 Chaos 入场条件" --id kalos

对外 API：
    tokenize(text)
    retrieve_documents(query, top_n=None, min_score=0.0)
    retrieve_chunks(query, doc_ids, top_k=None, max_chars=None)
    filter_valid_ids(ids)
    doc_chars(doc_id)
    format_candidates(ids, max_chars=3000)
    build_knowledge_block(chunks)
"""

from __future__ import annotations

import argparse
import math
import re

from backend.config import settings
from backend.knowledge.chunker import KnowledgeChunk, split_document
from backend.knowledge.knowledge import DEFAULT_KNOWLEDGE_DIR, KNOWLEDGE_CATALOG, get_knowledge

# 词元化：ASCII 字母数字整段为词元；CJK 单字为词元（配合 meta 关键词/同义词使用）
_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """极简词元化：小写；连续 ASCII 字母/数字为一个词元；每个 CJK 单字一个词元。"""
    text = (text or "").lower()
    tokens = _WORD_RE.findall(text)
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            tokens.append(ch)
    return tokens


class Bm25Index:
    """极简 BM25（k1≈1.5, b≈0.75），基于倒排表实现，纯标准库。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._postings: dict[str, dict[str, int]] = {}   # 词元 -> {doc_id: tf}
        self._doc_len: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self.n_docs = 0
        self.avgdl = 0.0

    def add_doc(self, doc_id: str, tokens: list[str]) -> None:
        if not tokens:
            return
        counts: dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        self._doc_len[doc_id] = len(tokens)
        for t, tf in counts.items():
            self._postings.setdefault(t, {})[doc_id] = tf
        self.n_docs += 1

    def build(self) -> None:
        n = self.n_docs
        self.avgdl = (sum(self._doc_len.values()) / n) if n else 0.0
        self._idf = {}
        for t, post in self._postings.items():
            df = len(post)
            self._idf[t] = math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: list[str]) -> list[tuple[str, float]]:
        """按分数降序返回 [(doc_id, score), ...]。"""
        if not query_tokens or self.n_docs == 0:
            return []
        acc: dict[str, float] = {}
        seen: set[str] = set()
        k1, b, avgdl = self.k1, self.b, self.avgdl
        for t in query_tokens:
            if t in seen or t not in self._postings:
                continue
            seen.add(t)
            idf = self._idf.get(t, 0.0)
            for doc_id, tf in self._postings[t].items():
                dl = self._doc_len[doc_id]
                denom = tf + k1 * (1 - b + b * dl / avgdl) if avgdl else (tf + k1)
                acc[doc_id] = acc.get(doc_id, 0.0) + idf * (tf * (k1 + 1)) / denom
        return sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))


# ---------------------------------------------------------------------------
# 模块级索引缓存（文档索引 + 片段索引），按知识库 mtime 指纹懒重建
# ---------------------------------------------------------------------------

_DOC_INDEX: Bm25Index | None = None
_CHUNK_INDEX: Bm25Index | None = None
_CHUNKS: dict[str, KnowledgeChunk] = {}       # chunk_id -> chunk
_CHUNK_OF_DOC: dict[str, list[str]] = {}      # doc_id -> [按 seq 排序的 chunk_id]
_DOC_CHARS: dict[str, int] = {}               # doc_id -> 文档全部分块字符数
_FINGERPRINT: tuple[int, int] | None = None


def _fingerprint() -> tuple[int, int]:
    """知识库文件指纹：(文件数, 最新 mtime_ns)。"""
    kb_dir = DEFAULT_KNOWLEDGE_DIR
    if not kb_dir.is_dir():
        return (0, 0)
    count = 0
    last = 0
    for p in kb_dir.rglob("*.md"):
        count += 1
        try:
            last = max(last, p.stat().st_mtime_ns)
        except OSError:
            pass
    return (count, last)


def _ensure_indexes() -> None:
    global _DOC_INDEX, _CHUNK_INDEX, _CHUNKS, _CHUNK_OF_DOC, _DOC_CHARS, _FINGERPRINT
    fp = _fingerprint()
    if fp == _FINGERPRINT and _DOC_INDEX is not None:
        return

    doc_index = Bm25Index()
    chunk_index = Bm25Index()
    chunks: dict[str, KnowledgeChunk] = {}
    chunk_of_doc: dict[str, list[str]] = {}
    doc_chars: dict[str, int] = {}

    for doc_id in sorted(KNOWLEDGE_CATALOG):
        info = KNOWLEDGE_CATALOG[doc_id]
        title = str(info.get("title") or doc_id)
        meta_parts = [
            str(info.get("description") or ""),
            str(info.get("keyword") or ""),
            " ".join(map(str, info.get("tags") or [])),
            str(info.get("folder") or ""),
        ]
        doc_index.add_doc(doc_id, tokenize(f"{title} {' '.join(meta_parts)}"))

        text = get_knowledge(doc_id)
        if not text:
            continue
        doc_chunks = split_document(
            text,
            doc_id,
            title,
            min_chars=settings.chunk_min_chars,
            max_chars=settings.chunk_max_chars,
        )
        ids: list[str] = []
        total = 0
        for c in doc_chunks:
            chunks[c.chunk_id] = c
            ids.append(c.chunk_id)
            total += len(c.text)
            chunk_index.add_doc(c.chunk_id, tokenize(c.text))
        chunk_of_doc[doc_id] = ids
        doc_chars[doc_id] = total

    doc_index.build()
    chunk_index.build()

    _DOC_INDEX = doc_index
    _CHUNK_INDEX = chunk_index
    _CHUNKS = chunks
    _CHUNK_OF_DOC = chunk_of_doc
    _DOC_CHARS = doc_chars
    _FINGERPRINT = fp


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


def filter_valid_ids(ids: list[str]) -> list[str]:
    """过滤掉目录中不存在的 id（LLM 可能输出幻觉 id），保序去重。"""
    seen: set[str] = set()
    valid: list[str] = []
    for i in ids or []:
        if i in KNOWLEDGE_CATALOG and i not in seen:
            seen.add(i)
            valid.append(i)
    return valid


def doc_chars(doc_id: str) -> int:
    """文档全部分块的总字符数（用于知识池预算/淘汰）。"""
    _ensure_indexes()
    return _DOC_CHARS.get(doc_id, 0)


def retrieve_documents(
    query: str,
    top_n: int | None = None,
    min_score: float = 0.0,
) -> list[tuple[str, float]]:
    """按问题从目录条目中检索候选知识条目，返回 [(knowledge_id, score), ...]。"""
    _ensure_indexes()
    if top_n is None:
        top_n = settings.retrieval_top_docs
    out: list[tuple[str, float]] = []
    for doc_id, score in _DOC_INDEX.score(tokenize(query)):
        if score < min_score:
            continue
        out.append((doc_id, score))
        if len(out) >= top_n:
            break
    return out


def retrieve_chunks(
    query: str,
    doc_ids: list[str],
    top_k: int | None = None,
    max_chars: int | None = None,
) -> list[KnowledgeChunk]:
    """在给定文档内按问题截取片段，总量受 max_chars 预算约束。

    选择策略（关键：保证多轮知识在场）：
    1. 第一遍：每个传入文档至少返回其最高分片段（预算允许时），
       使历史轮次解析出的文档在后续追问中仍被注入；
    2. 第二遍：按相关分降序填充剩余预算（每文档最多 top_k 个片段）。
    """
    _ensure_indexes()
    if top_k is None:
        top_k = settings.chunk_top_k
    if max_chars is None:
        max_chars = settings.knowledge_max_chars

    valid = [d for d in doc_ids if d in _CHUNK_OF_DOC]
    if not valid:
        return []

    q_tokens = tokenize(query)
    doc_score = dict(_DOC_INDEX.score(q_tokens))
    chunk_score = dict(_CHUNK_INDEX.score(q_tokens))

    def rank_ids(doc_id: str) -> list[str]:
        return sorted(
            _CHUNK_OF_DOC[doc_id],
            key=lambda cid: (-chunk_score.get(cid, 0.0), _CHUNKS[cid].seq),
        )

    # 相关度高的文档在前；同分时更新鲜（列表靠后）的优先，保证新知识不输给旧知识
    recency = {doc_id: i for i, doc_id in enumerate(valid)}
    doc_order = sorted(valid, key=lambda d: (-doc_score.get(d, 0.0), -recency[d]))

    selected: list[str] = []
    used_chars = 0

    def fits(chunk_id: str) -> bool:
        return used_chars + len(_CHUNKS[chunk_id].text) <= max_chars

    # 第一遍：每个文档至少一段
    for doc_id in doc_order:
        ids = rank_ids(doc_id)
        if ids and fits(ids[0]):
            selected.append(ids[0])
            used_chars += len(_CHUNKS[ids[0]].text)

    # 第二遍：填充剩余预算
    for doc_id in doc_order:
        for chunk_id in rank_ids(doc_id)[1:top_k]:
            if chunk_id in selected:
                continue
            if not fits(chunk_id):
                break
            selected.append(chunk_id)
            used_chars += len(_CHUNKS[chunk_id].text)

    return [_CHUNKS[cid] for cid in selected]


def format_candidates(ids: list[str], max_chars: int = 3000) -> str:
    """把候选知识条目压缩成一行一条的紧凑文本（供意图解析 LLM 选择）。"""
    lines: list[str] = []
    for doc_id in ids:
        info = KNOWLEDGE_CATALOG.get(doc_id)
        if not info:
            continue
        title = str(info.get("title") or doc_id)
        keyword = str(info.get("keyword") or "")
        description = str(info.get("description") or "")
        if len(keyword) > 90:
            keyword = keyword[:90] + "…"
        if len(description) > 60:
            description = description[:60] + "…"
        line = f"- id: {doc_id} | title: {title}"
        if keyword:
            line += f" | keyword: {keyword}"
        if description:
            line += f" | description: {description}"
        lines.append(line)
        if sum(len(l) + 1 for l in lines) > max_chars:
            lines = lines[: max(1, len(lines) - 1)]
            break
    return "\n".join(lines)


def build_knowledge_block(chunks: list[KnowledgeChunk]) -> str | None:
    """把检索片段拼成注入用的 system 知识消息；无片段时返回 None。"""
    if not chunks:
        return None
    head = (
        "以下是按你的问题从知识库检索到的相关片段"
        "（片段可能不完整；若信息不足，请如实说明，不要编造）："
    )
    parts = [head]
    for i, c in enumerate(chunks, 1):
        section = c.heading_path or c.title
        parts.append(
            f"【片段 {i}/{len(chunks)}｜文档：{c.title}｜章节：{section}】\n{c.text}"
        )
    return "\n\n----\n\n".join(parts)


def get_chunk_text(chunk_id: str) ->str | None:
    """按 chunk_id 取片段文本；未收录时返回 None（不要抛异常）。

    评测侧会对 LLM 可能产出的幻觉 id 逐个调用，抛异常会连累整轮的采集。
    """
    _ensure_indexes()
    chunk = _CHUNKS.get(chunk_id)
    return chunk.text if chunk else None


def main() -> int:
    parser = argparse.ArgumentParser(description="知识库本地检索演示")
    parser.add_argument("--query", default="虎影 Bravado 技能效果", help="检索问题")
    parser.add_argument("--top", type=int, default=None, help="候选条目数")
    parser.add_argument(
        "--id", action="append", default=None, help="限定检索的 knowledge_id（可多次）"
    )
    args = parser.parse_args()

    docs = retrieve_documents(args.query, top_n=args.top)
    print(f"问题：{args.query}")
    print(f"候选知识条目（{len(docs)} 个）：")
    for doc_id, score in docs:
        print(f"  - {doc_id} (score={score:.3f})")

    doc_ids = args.id if args.id else [doc_id for doc_id, _ in docs[:3]]
    chunks = retrieve_chunks(args.query, doc_ids)
    print(f"\n命中片段（{len(chunks)} 个，预算 {settings.knowledge_max_chars} 字符）：")
    for c in chunks:
        print(f"  - {c.chunk_id} | 章节：{c.heading_path or '无'} | {len(c.text)} 字符")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
