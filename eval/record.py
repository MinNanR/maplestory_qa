"""录制契约：runner 与 metrics 之间唯一的数据形状。

设计约定
--------
1. **录制自包含**：CaseRecord.case 是当时的用例快照（含逐轮期望）。metrics 只读录制
   就能算分，不需要重新加载用例集 —— 否则用例集一改，历史 run 的分数就会漂移。
2. **失败也要有记录**：CaseRecord.status / TurnRecord.error 让 metrics 能区分
   "没跑" 和 "跑失败"。
3. **meta 是环境指纹**：回答"两次 run 之间分数为什么变"。
   注意白名单取值，绝不 dump 全量 settings（会写入 llm_api_key）。
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TurnRecord:
    turn_idx: int
    request_id: str
    query: str
    answer: str
    error: str | None = None
    injected_chunk_ids: list[str] = field(default_factory=list)
    injected_doc_ids: list[str] = field(default_factory=list)
    # 注入片段的**文本**（不是指标结果）：录制自带证据，改锚点/加指标时无需重跑 LLM。
    injected_chunk_texts: list[str] = field(default_factory=list)
    did_retrieve: bool = False
    latency_ms: float = 0.0
    ttft_ms: float | None = None

    # token / 成本：来自 usage 事件。None = 未知（不要与 0 混）
    llm_calls: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    usage_estimated: bool = False          # 有 span 的 token 是估算值
    spans: list[dict] = field(default_factory=list)   # 每 span 摘要（name/kind/tokens/duration）


@dataclass
class CaseRecord:
    case: dict                    # 用例快照（asdict(Case)，含逐轮期望）
    turns: list[TurnRecord]
    status: str = "ok"            # "ok" | "error"


@dataclass
class RunRecord:
    run_id: str
    started_at: str
    meta: dict
    cases: list[CaseRecord]


def record_run(run_record: RunRecord, save_dir: str) -> Path:
    p = Path(f"{save_dir}/{run_record.run_id}.json")
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p.resolve(), "w", encoding="utf-8") as f:
        json.dump(asdict(run_record), f, ensure_ascii=False, indent=2)

    return p


def load_run(path: str | Path) -> RunRecord:
    """读回录制。metrics 的两个入口（内存直传 / 读文件）都走同一形状。"""
    p = Path(path)
    if p.is_dir():
        found = sorted(p.glob("*.json"))
        if len(found) != 1:
            raise ValueError(f"{p} 下应有且仅有一个 run json，实际 {len(found)} 个")
        p = found[0]

    raw = json.loads(p.read_text(encoding="utf-8"))
    return RunRecord(
        run_id=raw["run_id"],
        started_at=raw["started_at"],
        meta=raw.get("meta") or {},
        cases=[
            CaseRecord(
                case=c["case"],
                status=c.get("status", "ok"),
                turns=[
                    TurnRecord(
                        turn_idx=t["turn_idx"],
                        request_id=t.get("request_id", ""),
                        query=t.get("query", ""),
                        answer=t.get("answer", ""),
                        error=t.get("error"),
                        injected_chunk_ids=t.get("injected_chunk_ids") or [],
                        injected_doc_ids=t.get("injected_doc_ids") or [],
                        injected_chunk_texts=t.get("injected_chunk_texts")
                        or t.get("context_recall")      # 兼容旧录制里那个名字
                        or [],
                        did_retrieve=bool(t.get("did_retrieve")),
                        latency_ms=float(t.get("latency_ms") or 0.0),
                        ttft_ms=t.get("ttft_ms"),
                        llm_calls=t.get("llm_calls"),
                        input_tokens=t.get("input_tokens"),
                        output_tokens=t.get("output_tokens"),
                        usage_estimated=bool(t.get("usage_estimated")),
                        spans=list(t.get("spans") or []),
                    )
                    for t in c.get("turns", [])
                ],
            )
            for c in raw.get("cases", [])
        ],
    )


# ---------------------------------------------------------------------------
# 环境指纹
# ---------------------------------------------------------------------------

# 影响行为的配置项白名单。刻意不用 settings.model_dump()：
# 那会把 llm_api_key 写进录制文件里。
SNAPSHOT_SETTINGS = (
    "llm_provider",
    "llm_model",
    "llm_base_url",
    "retrieval_top_docs",
    "chunk_top_k",
    "chunk_min_chars",
    "chunk_max_chars",
    "knowledge_max_chars",
    "knowledge_pool_max_docs",
    "llm_stream_include_usage",
)

# 价目快照：只存 token、不把金额写进 span，所以单价必须随 run 一起存下来，
# 否则历史 run 的成本无法按当时的价目复算。
SNAPSHOT_PRICING = (
    "llm_price_input_per_mtok",
    "llm_price_output_per_mtok",
)


def _git(*args: str) -> str | None:
    """跑一条 git 命令；git 不可用/不在仓库里/命令失败都返回 None（不抛异常）。"""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def prompt_hash() -> str | None:
    """当前 prompt 的指纹。改了 prompt 之后，历史 run 的分数才解释得清。"""
    try:
        from backend.analysis.query_analyzer import SYSTEM_PROMPT
        from backend.prompts.system import get_system_message

        text = get_system_message().content + "\x00" + SYSTEM_PROMPT
    except Exception as e:  # noqa: BLE001 - 指纹采集失败不该影响评测
        print(f"[WARN] 采集 prompt_hash 失败：{e}")
        return None
    return _sha(text)


def code_hash() -> str | None:
    """backend/ 与 eval/ 下 .py 文件内容的指纹（不依赖 git）。

    为什么还需要它：git_status_hash 只对 `git status --porcelain` 的输出取 hash，
    而那份输出仅含"路径 + 状态"，**不含内容** —— 同一个文件改两次，两次的
    git_status_hash 是一样的。本函数直接对文件内容取 hash，因此
    「没提交也能精确区分两次 run 的代码是否相同」。
    """
    parts: list[str] = []
    for sub in ("backend", "eval"):
        root = REPO_ROOT / sub
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except OSError:
                continue
            parts.append(f"{p.relative_to(REPO_ROOT).as_posix()}\x00{text}")
    if not parts:
        return None
    return _sha("\x01".join(parts))


def git_meta() -> dict:
    """git 指纹。

    只记 commit 是不够的：本仓库当前 HEAD 只有一个 "Initial commit"，而几乎所有
    代码都在工作区未提交 —— 两次 run 之间 commit 完全不变，光看它区分不出任何变化。
    所以额外记：
      - git_dirty：工作区是否有改动
      - git_status_hash：改动清单的指纹（"没提交也能区分两次 run 的代码是否一样"）
    """
    commit = _git("rev-parse", "--short", "HEAD")
    porcelain = _git("status", "--porcelain")
    return {
        "git_commit": commit,
        "git_commit_count": _git("rev-list", "--count", "HEAD"),
        "git_dirty": bool(porcelain) if porcelain is not None else None,
        "git_status_hash": _sha(porcelain) if porcelain else None,
    }


def build_meta(cases: list, extra: dict | None = None) -> dict:
    from backend.config import settings

    meta: dict = {
        **git_meta(),
        "cases_count": len(cases),
        "cases_hash": _cases_hash(cases),
        "prompt_hash": prompt_hash(),
        "code_hash": code_hash(),
        "settings": {k: getattr(settings, k, None) for k in SNAPSHOT_SETTINGS},
        "pricing": {k: getattr(settings, k, None) for k in SNAPSHOT_PRICING},
        "python": platform.python_version(),
        "platform": sys.platform,
    }
    if extra:
        meta.update(extra)
    return meta


def _cases_hash(cases: list) -> str:
    from eval.case import cases_hash

    return cases_hash(cases)
