"""
知识库（knowledge/）meta.md 解析、目录概览与知识文件索引。

本模块提供三块能力：
1. build_catalog()：聚合各分类文件夹的 meta.md → 一份 LLM 可用的
   「知识库目录概览」文本（供 QueryAnalyzer 展示给分析 LLM，输出 knowledge_id 列表）；
2. scan_knowledge() / KNOWLEDGE_CATALOG：扫描知识库文件夹下的知识文件，
   构建 knowledge_id -> 文件索引 的路径缓存，供 get_knowledge() 使用；
3. get_knowledge(knowledge_id)：按 knowledge_id 读取对应知识文档的全文。

meta.md 格式约定（列表式）：

    ---
    id: boss-meta
    title: BOSS 攻略与数据
    type: folder_meta
    ---

    # BOSS 攻略与数据

    <文件夹主题描述正文……>

    ## 文件索引

    - id: kalos
      title: 卡洛斯（Kalos the Guardian）
      description: <文档内容描述……>
      keyword: 卡洛斯, Kalos, BOSS, 攻略, ...

说明：
- 文件夹主题 = 一级标题（#）之后、第一个二级标题（##）之前的正文；
- 文件索引 = 「## 文件索引」下的条目列表，每个条目以 "- id: xxx" 开头，
  后续以 2 空格缩进的 title / description / keyword 字段；值为单行文本。

knowledge_id 的解析顺序（scan_knowledge）：
1. 知识文件 front-matter 的 id（如 kalos / seren / starforce）；
2. 无 id 时，与同目录 meta.md「文件索引」条目匹配（按 front-matter title
   相等、规范化文件名与条目标题的前缀关系），取条目的 id，并附带 description / keyword；
3. 仍未匹配的文件回退用规范化文件名（去扩展名）作为 knowledge_id。

用法：
    python backend/knowledge/knowledge.py                       # 输出目录概览 + 文件索引
    python backend/knowledge/knowledge.py --index-only          # 只输出 id -> 文件 索引
    python backend/knowledge/knowledge.py --dir path/to/knowledge
    # 作为模块复用：
    from backend.knowledge.knowledge import build_catalog, get_knowledge, scan_knowledge
    catalog = build_catalog(Path("knowledge"))     # str，可直接拼进 system prompt
    index   = scan_knowledge()                     # {knowledge_id: 文件索引}
    content = get_knowledge("kalos")               # 知识文档全文（str）或 None
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# 与 app/knowledge.py 保持一致的 front-matter 解析方式
FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.MULTILINE)

# 文件索引条目：以 "- id: xxx" 开头
ENTRY_START_RE = re.compile(r"^-\s*id:\s*(?P<value>.+?)\s*$", re.MULTILINE)
# 条目字段：2 空格及以上缩进的 title / description / keyword
ENTRY_FIELD_RE = re.compile(r"^\s{2,}(?P<key>title|description|keyword):\s*(?P<value>.*?)\s*$", re.MULTILINE)

# 模块位于 <仓库根>/backend/knowledge/knowledge.py，parents[2] 即仓库根目录
_REPO_ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
DEFAULT_KNOWLEDGE_DIR = _REPO_ROOT / "knowledge"

# 不视为知识文件的文件名
_NON_DOC_NAMES = {"meta.md", "readme.md"}


@dataclass
class MetaEntry:
    """meta.md「文件索引」中的一条知识文档索引。"""
    id: str
    title: str
    description: str
    keyword: str


@dataclass
class FolderMeta:
    """一个分类文件夹（含 meta.md）的解析结果。"""
    folder: str                        # 相对 knowledge 的目录名，如 "job_skill"
    meta_path: str                     # meta.md 的相对路径
    title: str                         # front-matter.title，缺省用目录名
    summary: str                       # 文件夹主题描述正文
    entries: list[MetaEntry] = field(default_factory=list)   # 「文件索引」条目
    doc_count: int = 0                 # 该文件夹下除 meta.md 外的文档数（衡量规模）


def _section(heading: str, body: str) -> str:
    """抽取指定二级标题（##）下的正文，到下一个 ## 为止。"""
    lines, out, capture = body.splitlines(), [], False
    for line in lines:
        hm = HEADING_RE.match(line)
        if hm:
            if hm.group(1) == "##" and hm.group(2).strip() == heading:
                capture = True
                continue
            if capture:
                break
        if capture:
            out.append(line)
    return "\n".join(out).strip()


def extract_theme(body: str) -> str:
    """提取一级标题（#）之后、第一个二级标题（##）之前的正文，作为文件夹主题描述。"""
    lines, out, capture = body.splitlines(), [], False
    for line in lines:
        hm = HEADING_RE.match(line)
        if hm:
            if hm.group(1) == "#":
                capture = True
                continue
            if capture:
                break
        if capture:
            out.append(line)
    return "\n".join(out).strip()


def parse_entries(section_text: str) -> list[MetaEntry]:
    """解析「文件索引」章节中的条目列表。

    每个条目形如：
        - id: xxx
          title: xxx
          description: xxx
          keyword: xxx
    """
    entries: list[MetaEntry] = []
    current: dict[str, str] | None = None

    for line in section_text.splitlines():
        start = ENTRY_START_RE.match(line)
        if start:
            if current:
                entries.append(MetaEntry(**current))
            current = {"id": start.group("value"), "title": "", "description": "", "keyword": ""}
            continue
        if current is not None:
            fm = ENTRY_FIELD_RE.match(line)
            if fm:
                current[fm.group("key")] = fm.group("value")

    if current:
        entries.append(MetaEntry(**current))
    return entries


def parse_meta_file(path: Path) -> FolderMeta:
    raw = path.read_text(encoding="utf-8")

    # 1) front-matter（YAML）+ 正文分离
    m = FRONT_MATTER_RE.match(raw)
    if m:
        meta = yaml.safe_load(m.group(1)) or {}
        body = m.group(2)
    else:
        meta, body = {}, raw

    folder = path.parent.name
    title = str(meta.get("title") or folder)

    # 2) 文件夹主题描述 + 文件索引条目
    summary = extract_theme(body)
    entries = parse_entries(_section("文件索引", body))

    # 3) 同目录下知识文档数量（meta.md 自身不计）
    doc_count = sum(
        1 for p in path.parent.glob("*.md") if p.name.lower() != "meta.md"
    )

    # meta_path：优先给出相对仓库根的路径；目录不在仓库根下时退化为原样路径
    try:
        meta_path = path.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        meta_path = path.as_posix()

    return FolderMeta(
        folder=folder,
        meta_path=meta_path,
        title=title,
        summary=summary,
        entries=entries,
        doc_count=doc_count,
    )


def build_catalog(knowledge_dir: Path) -> str:
    """遍历 knowledge_dir 下所有含 meta.md 的文件夹，返回聚合文本。"""
    knowledge_dir = Path(knowledge_dir).resolve()
    if not knowledge_dir.is_dir():
        raise FileNotFoundError(f"知识库目录不存在：{knowledge_dir}")

    metas: list[FolderMeta] = []
    for meta_path in sorted(knowledge_dir.rglob("meta.md")):
        metas.append(parse_meta_file(meta_path))

    # 汇总统计
    total_docs = sum(m.doc_count for m in metas)
    missing = sorted(
        p.relative_to(knowledge_dir).as_posix()
        for p in knowledge_dir.iterdir()
        if p.is_dir() and not (p / "meta.md").exists()
    )

    lines = [
        "# 知识库目录概览（MapleStory Knowledge Base）",
        "",
        f"共 {len(metas)} 个分类，{total_docs} 篇知识文档。",
        "",
        "使用说明：请根据用户问题判断属于哪个分类，再前往该分类的文件索引定位对应知识文档；",
        "条目中的 id 与知识文档 front-matter 的 id 对应，标题/描述用于判断内容，关键词用于联想与匹配。",
        "",
    ]

    for m in metas:
        lines.append(f"## [{m.folder}] {m.title}")
        lines.append(f"- 内容主题：{' '.join(m.summary.split())}" if m.summary else "- 内容主题：无")
        if m.entries:
            lines.append("- 文件索引：")
            for e in m.entries:
                lines.append(f"  - id: {e.id}")
                lines.append(f"    title: {e.title}")
                lines.append(f"    description: {' '.join(e.description.split())}")
                lines.append(f"    keyword: {e.keyword}")
        lines.append("")

    if missing:
        lines.append(f"⚠ 以下文件夹缺少 meta.md：{', '.join(missing)}")
        lines.append("")

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# 知识文件索引：扫描知识库文件夹 -> KNOWLEDGE_CATALOG（id -> 文件信息），
# 供 get_knowledge() 读取文档全文。
# ---------------------------------------------------------------------------


def _read_front_matter(path: Path) -> tuple[dict, str]:
    """读取 md 文件的 front-matter（dict）与正文；解析失败视为无 front-matter。"""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}, ""
    m = FRONT_MATTER_RE.match(raw)
    if not m:
        return {}, raw
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        fm = {}
    if not isinstance(fm, dict):
        fm = {}
    return fm, m.group(2)


def _norm(text: str) -> str:
    """规范化字符串用于宽松比较：小写、去掉空白/下划线/连字符/括号。"""
    return re.sub(r"[\s_\-（）()\[\]]+", "", text).lower()


def _sanitize_id(text: str) -> str:
    """无 front-matter id 时的回退 id：小写、非字母数字字符转 '-。"""
    return re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", text.strip().lower()).strip("-")


def scan_knowledge(knowledge_dir: Path | None = None) -> dict[str, dict]:
    """扫描知识库文件夹下的知识文件，构建 knowledge_id -> 文件索引。

    knowledge_id 的解析顺序：
      1. 知识文件 front-matter 的 id（如 kalos / seren / starforce）；
      2. 无 id 时，用同目录 meta.md「文件索引」条目匹配（front-matter title 相等，
         或规范化文件名与条目标题成前缀/包含关系），取条目的 id，
         并附带条目的 description / keyword；
      3. 仍未匹配到的文件回退用规范化文件名（去扩展名）作为 knowledge_id。

    返回值形如：
        {knowledge_id: {
            "path":       相对仓库根目录的 POSIX 路径（get_knowledge 据此读取文件），
            "folder":     所在分类文件夹名，
            "title":      文档标题（取自 meta 条目或 front-matter 或文件名），
            "type":       front-matter 的 type（缺省 "note"），
            "tags":       front-matter 的 tags 列表，
            "description": meta 条目的描述（若由 meta 收录），
            "keyword":    meta 条目的关键词（若由 meta 收录）}}
    """
    knowledge_dir = Path(knowledge_dir).resolve() if knowledge_dir is not None else DEFAULT_KNOWLEDGE_DIR
    if not knowledge_dir.is_dir():
        return {}

    catalog: dict[str, dict] = {}
    claimed: set[Path] = set()

    def register(doc_id: str, path: Path, fm: dict, entry: MetaEntry | None) -> None:
        # path：优先相对仓库根，目录不在仓库根下时退化为原样路径
        try:
            rel = path.resolve().relative_to(_REPO_ROOT).as_posix()
        except ValueError:
            rel = path.as_posix()
        if entry is not None:
            title = entry.title
        else:
            title = str(fm.get("title") or path.stem)
        info: dict = {
            "path": rel,
            "folder": path.parent.name,
            "title": title,
            "type": str(fm.get("type") or "note"),
            "tags": list(fm.get("tags") or []),
        }
        if entry is not None:
            info["description"] = entry.description
            info["keyword"] = entry.keyword
        catalog[doc_id] = info

    # 1) 有 meta.md 的分类文件夹：按 meta「文件索引」条目与文件夹内文件一一对应
    for folder in sorted(p for p in knowledge_dir.iterdir() if p.is_dir()):
        meta_path = folder / "meta.md"
        if not meta_path.is_file():
            continue
        folder_meta = parse_meta_file(meta_path)

        docs = [
            (p, _read_front_matter(p)[0])
            for p in sorted(folder.glob("*.md"))
            if p.name.lower() not in _NON_DOC_NAMES
        ]
        remaining: list[tuple[Path, dict]] = list(docs)

        for entry in folder_meta.entries:
            hit: tuple[Path, dict] | None = None

            # a) front-matter id 精确匹配
            hit = next(
                ((p, fm) for p, fm in remaining if str(fm.get("id", "")).strip() == entry.id),
                None,
            )
            # b) front-matter title 规范化相等
            if hit is None:
                title_n = _norm(entry.title)
                hit = next(
                    ((p, fm) for p, fm in remaining
                     if fm.get("title") and _norm(str(fm["title"])) == title_n),
                    None,
                )
            # c) 规范化文件名（去扩展名）与条目标题成前缀/包含关系
            if hit is None:
                title_n = _norm(entry.title)
                hit = next(
                    ((p, fm) for p, fm in remaining
                     if (stem_n := _norm(p.stem)) and (title_n.startswith(stem_n) or stem_n.startswith(title_n))),
                    None,
                )
            # d) 文件夹中唯一剩余且无 front-matter id 的文件（兜底）
            if hit is None:
                leftovers = [(p, fm) for p, fm in remaining if not str(fm.get("id", "")).strip()]
                if len(leftovers) == 1:
                    hit = leftovers[0]

            if hit is not None:
                p, fm = hit
                register(entry.id, p, fm, entry)
                claimed.add(p)
                remaining = [x for x in remaining if x[0] != p]

        # 2) 未被 meta 收录的剩余文件：按 front-matter id 或规范化文件名收录
        for p, fm in remaining:
            doc_id = str(fm.get("id", "")).strip() or _sanitize_id(p.stem)
            if doc_id and doc_id not in catalog:
                register(doc_id, p, fm, None)
                claimed.add(p)

    # 3) 兜底：任何仍未被收录的知识文件（例如尚无 meta.md 的分类文件夹）
    for path in sorted(knowledge_dir.rglob("*.md")):
        if path in claimed or path.name.lower() in _NON_DOC_NAMES:
            continue
        fm = _read_front_matter(path)[0]
        doc_id = str(fm.get("id", "")).strip() or _sanitize_id(path.stem)
        if doc_id and doc_id not in catalog:
            register(doc_id, path, fm, None)

    return catalog


# 知识库路径缓存：knowledge_id -> 文件索引（模块加载时构建，可用 reload_knowledge_index 刷新）
KNOWLEDGE_CATALOG: dict[str, dict] = scan_knowledge()


def reload_knowledge_index(knowledge_dir: Path | None = None) -> dict[str, dict]:
    """重新扫描知识库并刷新全局 KNOWLEDGE_CATALOG，返回新的索引。"""
    global KNOWLEDGE_CATALOG
    KNOWLEDGE_CATALOG = scan_knowledge(knowledge_dir)
    return KNOWLEDGE_CATALOG


def get_knowledge(knowledge_id: str) -> str | None:
    """根据 knowledge_id 返回知识文档全文（str）；未收录或文件缺失时返回 None。"""
    info = KNOWLEDGE_CATALOG.get(knowledge_id)
    if not info:
        return None
    path = _REPO_ROOT / info["path"]
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="扫描知识库并输出目录概览与文件索引")
    parser.add_argument("--dir", default=str(DEFAULT_KNOWLEDGE_DIR), help="知识库根目录")
    parser.add_argument("--index-only", action="store_true", help="只输出 id -> 文件 索引")
    args = parser.parse_args()
    root = Path(args.dir)
    try:
        if not root.is_dir():
            raise FileNotFoundError(f"知识库目录不存在：{root}")
        index = scan_knowledge(root)
        if args.index_only:
            for doc_id in sorted(index):
                print(f"{doc_id}\t{index[doc_id]['path']}")
            return 0
        print(build_catalog(root))
        print()
        print("# 文件索引（knowledge_id -> 文件）")
        for doc_id in sorted(index):
            info = index[doc_id]
            print(f"- {doc_id}: {info['path']}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
