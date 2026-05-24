from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3

import yaml


WORD_RE = re.compile(r"[A-Za-z0-9_+\-]+|[\u4e00-\u9fff]+")
HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)
FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


@dataclass
class Chunk:
    chunk_id: str
    entry_id: str
    title: str
    entry_type: str
    source: str
    section: str
    tags: list[str]
    text: str
    tokens: list[str]
    path: str


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in WORD_RE.findall(text):
        lowered = token.lower()
        tokens.append(lowered)
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            chars = list(token)
            tokens.extend(chars)
            tokens.extend("".join(chars[index:index + 2]) for index in range(len(chars) - 1))
    return tokens


def split_markdown_sections(body: str) -> list[tuple[str, str]]:
    matches = list(HEADING_RE.finditer(body))
    if not matches:
        return [("正文", body.strip())] if body.strip() else []

    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        section_title = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        content = body[start:end].strip()
        if content:
            sections.append((section_title, content))
    return sections


def parse_markdown_file(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    match = FRONT_MATTER_RE.match(raw)
    if match:
        meta = yaml.safe_load(match.group(1)) or {}
        body = match.group(2)
    else:
        meta = {}
        body = raw

    entry_id = meta.get("id", path.stem)
    title = meta.get("title", path.stem)
    entry_type = meta.get("type", "note")
    source = meta.get("source", path.name)
    tags = meta.get("tags", [])

    chunks: list[Chunk] = []
    sections = split_markdown_sections(body)
    if not sections and body.strip():
        sections = [("正文", body.strip())]

    for index, (section, text) in enumerate(sections, start=1):
        combined = f"{title}\n{section}\n{text}".strip()
        chunks.append(
            Chunk(
                chunk_id=f"{entry_id}:{index}",
                entry_id=entry_id,
                title=title,
                entry_type=entry_type,
                source=source,
                section=section,
                tags=tags,
                text=text,
                tokens=tokenize(combined),
                path=str(path),
            )
        )
    return chunks


def sanitize_entry_id(text: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", text.strip().lower())
    return sanitized.strip("-") or "entry"


def build_boss_chunks(row: sqlite3.Row, db_path: Path) -> list[Chunk]:
    boss_name = row["boss_name"]
    entry_id = f"boss-db-{row['id']}-{sanitize_entry_id(boss_name)}"
    source = "robot.db:boss_info"
    path = f"{db_path}#boss_info:{row['id']}"
    tags = ["BOSS", "数据库", "boss_info"]

    sections = [
        (
            "基础信息",
            (
                f"名称：{boss_name}\n"
                f"等级：{row['level'] or '未知'}\n"
                f"血量：{row['hp'] or '未知'}\n"
                f"奖励：{row['reward'] if row['reward'] is not None else '未知'}"
            ),
        ),
        (
            "战斗属性",
            (
                f"物理防御：{row['physical_defense'] if row['physical_defense'] is not None else '未知'}\n"
                f"魔法防御：{row['magical_defense'] if row['magical_defense'] is not None else '未知'}\n"
                f"属性减伤：{row['element_reduction'] if row['element_reduction'] is not None else '未知'}\n"
                f"ARC 要求：{row['arc'] or '无'}\n"
                f"AUT 要求：{row['aut'] or '无'}"
            ),
        ),
        (
            "挑战限制",
            (
                f"重新进入间隔：{row['reenter_interval'] or '未知'}\n"
                f"领取次数限制：{row['claim_limit'] or '未知'}"
            ),
        ),
    ]

    chunks: list[Chunk] = []
    for index, (section, text) in enumerate(sections, start=1):
        combined = f"{boss_name}\n{section}\n{text}"
        chunks.append(
            Chunk(
                chunk_id=f"{entry_id}:{index}",
                entry_id=entry_id,
                title=boss_name,
                entry_type="boss",
                source=source,
                section=section,
                tags=tags,
                text=text,
                tokens=tokenize(combined),
                path=path,
            )
        )
    return chunks


def load_boss_info_chunks(db_path: Path) -> list[Chunk]:
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                id,
                boss_name,
                hp,
                level,
                physical_defense,
                magical_defense,
                element_reduction,
                arc,
                aut,
                reenter_interval,
                claim_limit,
                reward
            FROM boss_info
            ORDER BY id
            """
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    chunks: list[Chunk] = []
    for row in rows:
        chunks.extend(build_boss_chunks(row, db_path))
    return chunks


class KnowledgeBase:
    def __init__(self, knowledge_dir: Path, db_path: Path | None = None):
        self.knowledge_dir = knowledge_dir
        self.db_path = db_path
        self.chunks: list[Chunk] = []

    def load(self) -> None:
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        chunks: list[Chunk] = []
        for path in sorted(self.knowledge_dir.rglob("*.md")):
            chunks.extend(parse_markdown_file(path))
        if self.db_path is not None:
            chunks.extend(load_boss_info_chunks(self.db_path))
        self.chunks = chunks

    def list_entries(self) -> list[dict]:
        grouped: dict[str, dict] = {}
        for chunk in self.chunks:
            grouped.setdefault(
                chunk.entry_id,
                {
                    "entry_id": chunk.entry_id,
                    "title": chunk.title,
                    "type": chunk.entry_type,
                    "source": chunk.source,
                    "tags": chunk.tags,
                    "path": chunk.path,
                    "sections": [],
                },
            )
            grouped[chunk.entry_id]["sections"].append(chunk.section)
        return sorted(grouped.values(), key=lambda item: (item["type"], item["title"]))

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_set = set(query_tokens)
        scored: list[tuple[float, Chunk]] = []
        for chunk in self.chunks:
            token_set = set(chunk.tokens)
            overlap = len(query_set & token_set)
            if overlap == 0:
                continue

            coverage = overlap / len(query_set)
            density = overlap / max(len(token_set), 1)
            title_boost = 0.5 if any(token in chunk.title.lower() for token in query_set) else 0.0
            tag_boost = 0.3 if any(token in " ".join(chunk.tags).lower() for token in query_set) else 0.0
            score = overlap + coverage + density + title_boost + tag_boost
            scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, chunk in scored[:top_k]:
            results.append(
                {
                    "score": round(score, 3),
                    "chunk_id": chunk.chunk_id,
                    "entry_id": chunk.entry_id,
                    "title": chunk.title,
                    "type": chunk.entry_type,
                    "section": chunk.section,
                    "source": chunk.source,
                    "tags": chunk.tags,
                    "text": chunk.text,
                    "path": chunk.path,
                }
            )
        return results
