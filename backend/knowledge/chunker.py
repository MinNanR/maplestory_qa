"""
知识文档分块（backend/knowledge/chunker.py）

把单个知识文档（Markdown）按标题结构切成有边界的片段（chunk），
供检索与回答时「根据用户问题截取有用片段」注入提示词，避免整篇文档进上下文。

分块策略（自适应，纯标准库）：
1. 剥离 front-matter；把正文解析成标题树（heading 行 -> 节点，普通行归属最近标题）；
2. 递归产出片段：节点片段文本 = 该节点的标题路径（含祖先标题行，便于按章节检索）
   + 节点自身正文；
3. 片段文本超过 chunk_max_chars 时：
   - 节点含子标题 -> 由递归天然按子标题切细；
   - 纯正文节点（无子标题，如大表格）-> 按空行段落硬切为 <= chunk_max_chars 的窗口；
4. 相邻过小片段（< chunk_min_chars）在不超上限的前提下向后合并。

对外 API：
    KnowledgeChunk                     # 片段数据类
    split_document(raw_text, doc_id, title,
                   min_chars, max_chars) -> list[KnowledgeChunk]

覆盖的知识文档结构：
- job_skill：## 转职阶段 / ### 技能（天然按技能切块，heading 带英文技能名，利于检索）；
- boss：## 章节 / ### 小节；
- system：## / ### + 大表格（纯正文节点按段落/行硬切兜底）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 与 backend/knowledge/knowledge.py 保持一致的 front-matter 解析
FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


@dataclass
class KnowledgeChunk:
    """知识文档中的一个片段。"""

    doc_id: str            # 所属知识文档的 knowledge_id
    title: str             # 文档标题（用于展示）
    heading_path: str      # 章节路径展示文本，如 "5 转技能 / Burning Soul Blade"
    seq: int               # 文档内序号（0 起）
    text: str              # 片段文本（含标题路径行）

    @property
    def chunk_id(self) -> str:
        """稳定片段 id：doc_id#seq，用于跨轮次去重。"""
        return f"{self.doc_id}#{self.seq}"


@dataclass
class _HNode:
    """标题树节点。"""

    level: int                          # 标题级别（1~6），根节点为 0
    heading: str                        # 标题文本（不含 #），根节点为空
    own_lines: list[str] = field(default_factory=list)   # 该节点下、子标题前的正文行
    children: list["_HNode"] = field(default_factory=list)


def _parse_tree(body: str) -> _HNode:
    """把 markdown 正文解析成标题树，根节点 level=0、heading=''。"""
    root = _HNode(0, "")
    stack: list[_HNode] = [root]
    for line in body.splitlines():
        m = HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            heading = m.group(2).strip()
            while stack and stack[-1].level >= level:
                stack.pop()
            node = _HNode(level, heading)
            stack[-1].children.append(node)
            stack.append(node)
        else:
            stack[-1].own_lines.append(line)
    return root


def _hard_split(text: str, max_chars: int) -> list[str]:
    """把超长纯文本按空行段落切成 <= max_chars 的窗口；单段超长时按行切。"""
    pieces: list[str] = []
    current = ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if not current or len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                pieces.append(current)
            if len(para) <= max_chars:
                current = para
            else:
                # 单个段落仍超长：按行累积窗口
                current = ""
                for ln in para.splitlines():
                    if current and len(current) + len(ln) + 1 > max_chars:
                        pieces.append(current)
                        current = ln
                    else:
                        current = f"{current}\n{ln}" if current else ln
    if current:
        pieces.append(current)
    return [p for p in pieces if p.strip()]


def _emit(
    node: _HNode,
    ancestors: list[tuple[int, str]],
    out: list[tuple[str, str]],
    max_chars: int,
) -> None:
    """深度优先产出 (片段文本, 章节展示路径)，保持文档顺序。

    ancestors 为祖先标题的 (level, heading) 列表。
    """
    path = ancestors + ([(node.level, node.heading)] if node.heading else [])
    prefix = "\n".join(f"{'#' * level} {heading}" for level, heading in path)
    display = " / ".join(heading for _, heading in path)

    own = "\n".join(node.own_lines).strip()
    if own:
        txt = f"{prefix}\n{own}" if prefix else own
        if len(txt) <= max_chars:
            out.append((txt, display))
        else:
            for piece in _hard_split(txt, max_chars):
                out.append((piece, display))

    for child in node.children:
        _emit(child, path, out, max_chars)


def _merge_small(
    pieces: list[tuple[str, str]], min_chars: int, max_chars: int
) -> list[tuple[str, str]]:
    """把过小的相邻片段向后合并（合并后仍不超过 max_chars）。"""
    merged: list[tuple[str, str]] = []
    for text, display in pieces:
        if (
            merged
            and len(merged[-1][0]) < min_chars
            and len(merged[-1][0]) + len(text) <= max_chars
        ):
            prev_text, prev_display = merged[-1]
            merged[-1] = (f"{prev_text}\n\n{text}", prev_display or display)
        else:
            merged.append((text, display))
    # 尾部过小片段并入前一块
    if len(merged) >= 2 and len(merged[-1][0]) < min_chars:
        tail_text, _ = merged.pop()
        prev_text, prev_display = merged[-1]
        if len(prev_text) + len(tail_text) <= max_chars:
            merged[-1] = (f"{prev_text}\n\n{tail_text}", prev_display)
        else:
            merged.append((tail_text, ""))
    return merged


def split_document(
    raw_text: str,
    doc_id: str,
    title: str,
    min_chars: int = 300,
    max_chars: int = 2500,
) -> list[KnowledgeChunk]:
    """把单个知识文档切成带章节路径的片段列表。"""
    body = raw_text
    m = FRONT_MATTER_RE.match(raw_text)
    if m:
        body = m.group(2)

    root = _parse_tree(body)

    pieces: list[tuple[str, str]] = []
    root_own = "\n".join(root.own_lines).strip()
    if root_own:
        pieces.append((root_own, ""))
    for child in root.children:
        _emit(child, [], pieces, max_chars)

    pieces = _merge_small(pieces, min_chars, max_chars)

    chunks: list[KnowledgeChunk] = []
    for seq, (text, display) in enumerate(pieces):
        chunks.append(
            KnowledgeChunk(
                doc_id=doc_id,
                title=title,
                heading_path=display,
                seq=seq,
                text=text,
            )
        )
    return chunks
