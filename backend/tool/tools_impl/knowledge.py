from backend.tool.dispatcher import dispatcher, ToolOutput
from backend.knowledge.knowledge import (
    FolderMeta,
    parse_meta_file,
    # ,
    DEFAULT_KNOWLEDGE_DIR,
)
from backend.knowledge.retrieval import (
    retrieve_chunks_in_doc,
    retrieve_chunks,
    build_knowledge_block,
    build_sub_catalog
)
import json

# 会"检索知识库"的工具名集合。
# 刻意放在工具层而不是评测层：改名或新增检索工具时只改这一处 ——
# 否则 eval 里会残留一份过期的工具名表，指标无声失效。
RETRIEVAL_TOOL_NAMES = frozenset(
    {"retrieval_in_document", "retrieval_among_document"}
)


@dispatcher.register
def read_knowledge_catalog() -> str:
    """查看知识库的全部分类目录（每个分类的标题与内容概述）。

    何时用：不确定某个主题属于哪个分类，或需要确认知识库里收录了哪些内容时。
    何时不用：已经知道确切的 doc_id（直接调用检索工具更快）；
    与游戏无关的闲聊、写代码类问题（不要调用任何工具）。

    返回：分类列表（title / summary / folder）。**它不返回具体知识条目的 id** ——
    要拿到条目 id，请把这里的 folder 原样传给 read_knowledge_sub_catalog。
    """
    metas: list[FolderMeta] = []
    for meta_path in sorted(DEFAULT_KNOWLEDGE_DIR.rglob("meta.md")):
        metas.append(parse_meta_file(meta_path))

    catalog = [
        {"title": meta.title, "summary": meta.summary, "folder": meta.folder}
        for meta in metas
    ]

    return json.dumps(catalog, ensure_ascii=False)


@dispatcher.register
def read_knowledge_sub_catalog(folder: str) -> str:
    """查看某个分类下的知识条目清单（条目 id、标题、描述、关键词）。

    何时用：已经知道大致分类，但不确定该读哪一篇具体文档时。
    何时不用：已经拿到确切的 doc_id（直接调用检索工具）；还不确定分类（先调用 read_knowledge_catalog）。

    返回：该分类下所有知识条目的 id 与说明。**返回的 id 就是后续检索工具要的
    doc_id**（retrieval_in_document 的 doc_id，或 retrieval_among_document 的 doc_ids 元素），
    请原样使用，不要改写。

    Args:
        folder (string): 分类文件夹名，必须原样取自 read_knowledge_catalog 返回的 folder 字段，不要自己拼写或猜测
    """
    dir = DEFAULT_KNOWLEDGE_DIR / folder / "meta.md"
    return build_sub_catalog(dir)


@dispatcher.register
def retrieval_in_document(doc_id: str, retrieval_texts: list[str]) -> ToolOutput:
    """在**一个**指定文档内部，按关键词检回相关原文片段。

    何时用：已经拿到确切的 doc_id，需要该文档中与问题相关的原文
    （技能数值、机制说明、规则条文等）。
    何时不用：还不确定 doc_id（先用目录工具定位）；需要在多篇文档之间横向比较
    （改用 retrieval_among_document）。

    能力边界：**它不能跨文档搜索** —— 一次调用只覆盖 doc_id 指定的那一篇。

    Args:
        doc_id (string): 目标文档 id，必须原样取自目录工具返回的条目 id，不要自己拼写、猜测或改写
        retrieval_texts (list[string]): 1~3 个具体关键词（技能英文名、系统名、数值项），多个关键词会合并打分；不要把用户的整句问题当关键词传入
    """
    chunks = retrieve_chunks_in_doc(retrieval_texts, doc_id)

    if chunks:
        # chunk_ids 走 structured：只给观测/评测用，不进 text
        # （片段头里只有文档 title、没有 id，正文反解析不出来）
        return ToolOutput(
            text=build_knowledge_block(chunks=chunks),
            structured={"chunk_ids": [c.chunk_id for c in chunks]},
        )
    return ToolOutput(text="未能找到文档片段")


@dispatcher.register
def retrieval_among_document(doc_ids: list[str], retrieval_text: str) -> ToolOutput:
    """在多个指定文档中，按**一个**关键词检回相关原文片段（用于跨文档横向比较）。

    何时用：需要在 2 篇以上文档之间对比同一主题时，例如比较两个职业的同类技能、
    或核对某条规则在多篇文档中的说法。
    何时不用：只查一篇文档（改用 retrieval_in_document，它支持多个关键词）。

    注意：本工具的参数形状与 retrieval_in_document **不同** ——
    这里是 doc_ids 数组 + retrieval_text **单个**关键词字符串。
    需要多个关键词时，请对本工具发起多次调用，每次传一个关键词。

    Args:
        doc_ids (list[string]): 目标文档 id 列表（2 篇以上），每个 id 都必须原样取自目录工具返回的条目 id
        retrieval_text (string): **一个**关键词字符串（技能英文名、系统名），不要传整句问题，也不要传数组
    """
    chunks = retrieve_chunks(retrieval_text, doc_ids)

    if chunks:
        return ToolOutput(
            text=build_knowledge_block(chunks=chunks),
            structured={"chunk_ids": [c.chunk_id for c in chunks]},
        )
    return ToolOutput(text="未能找到文档片段")
