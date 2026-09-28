from backend.models import ToolCall, ToolFunctionDefinition, FunctionParameter
from docstring_parser import Docstring, DocstringParam, parse
from dataclasses import dataclass, field
from typing import Any
import json


@dataclass
class ToolCallRequest:
    tool_call_id: str = ""
    name: str = ""
    params: str = "{}"
    max_output_chars: int = 4000


@dataclass
class ToolOutput:
    """工具的文本 + 结构化结果。

    text       → 回灌给模型的正文
    structured → 给观测/评测/程序消费，**不进模型上下文**

    dispatcher 只透传 structured，**不解释里面的键** —— 键名由各工具自己声明：
    知识检索用 {"chunk_ids": [...]}，将来的外部搜索可以用 {"urls": [...]}，
    数据库检索可以用 {"row_count": n, "table": "..."}。加新工具族时 dispatcher 零改动。

    约束：structured 只放摘要级原始类型（id 列表、URL、计数），**不放正文** ——
    它会被带进事件流（tool_result 事件）并落进 trace 文件，正文只该走 trace 的
    tool_output 字段。

    text 与 structured 刻意分开：把 id 列表混进给模型看的正文会白烧 token，
    还会诱导模型在答案里复述 id。
    """

    text: str = ""
    structured: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallResult:
    tool_call_id: str = ""
    name: str = ""
    succeeded: bool = False  # 是否调用成功
    content: str = ""  # 工具调用结果（回灌给模型的文本）
    error_msg: str = ""  # 工具调用失败时的响应消息
    deduped: bool = False
    # 工具自报的结构化结果，dispatcher 原样透传。空 = 该工具没有结构化产出。
    # 评测侧的 doc_recall / ctx_recall 就靠这里的 chunk_ids —— 片段头里只有文档
    # title、没有 id，从正文反解析是反推不出来的。
    structured: dict[str, Any] = field(default_factory=dict)


def build_tool_call_request(tool_call: ToolCall) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call_id=tool_call.id, name=tool_call.name, params=tool_call.arguments
    )


class ToolDispatcher:
    def __init__(self):
        self._tools = {}

    def register(self, func=None, *, name=None):
        if func is None:

            def decorator(f):
                self._add_tool(f, name)
                return f

            return decorator
        self._add_tool(func, name)
        return func

    def _add_tool(self, func, name: str | None) -> None:
        """注册的唯一入口：校验通过才登记（装饰器写法与直接调用都走这里）。"""
        func_name = name or func.__name__
        _validate_tool_definition(func, func_name)
        self._tools[func_name] = func

    def dispatch(self, request: ToolCallRequest) -> ToolCallResult:
        tool_call_id = request.tool_call_id
        func_name = request.name
        if func_name not in self._tools:
            return ToolCallResult(
                tool_call_id=tool_call_id,
                name=func_name,
                succeeded=False,
                error_msg=f"未能找到tool:{func_name}",
            )
        func = self._tools.get(func_name)
        try:
            parameters_json = json.loads(request.params)
        except Exception:
            parameters_json = {}

        func_definition: ToolFunctionDefinition = _parse_function_doc(func=func)
        properties = func_definition.parameters.properties
        required_params = func_definition.parameters.required

        required_set = set(required_params)
        input_set = set(parameters_json.keys())
        if not required_set <= input_set:
            missing_set = required_set - input_set
            return ToolCallResult(
                tool_call_id=tool_call_id,
                name=func_name,
                succeeded=False,
                error_msg=f"缺少必传参数:{','.join(missing_set)}",
            )

        parameters_input = {
            p: parameters_json[p] for p in properties.keys() if p in parameters_json
        }
        raw = func(**parameters_input)
        # 三种返回形状归一到一个契约上：
        #   ToolOutput → text + structured
        #   str        → text，structured 为空
        #   其他/None  → 转成字符串，绝不让 None 流进 content
        #     （content 声明是 str，None 会在后续 len()/切片处炸成无关的 TypeError）
        if isinstance(raw, ToolOutput):
            content = raw.text
            structured = dict(raw.structured)
        elif isinstance(raw, str):
            content = raw
            structured = {}
        else:
            content = "" if raw is None else str(raw)
            structured = {}
        return ToolCallResult(
            tool_call_id=tool_call_id,
            name=func_name,
            content=content,
            succeeded=True,
            structured=structured,
        )

    def build_schema(self) -> list[dict]:
        tools_definitions = [
            {
                "type": "function",
                "function": _parse_function_doc(
                    func, function_name=func_name
                ).model_dump(),
            }
            for func_name, func in self._tools.items()
        ]
        return tools_definitions


def _validate_tool_definition(func, func_name: str) -> None:
    """注册期校验：宁可在 import 时炸，也不要等到模型调用它时才失败。

    为什么"没有 description"必须拦在注册期：工具 description 是模型选择工具的
    唯一依据（`tools=` 里描述"何时用它"的就只有这一个字段）。空 description 的工具
    在 schema 里是一张白纸，模型不可能可靠地选中它 —— 这是**配置错误**，
    不是运行期偶发故障，所以要在注册那一刻就暴露，而不是每次调用都返回一句
    "工具执行失败"，让模型和人都以为是自己参数写错了。

    判定用 `doc.description` 而不是 `func.__doc__`，两个原因：
      1. docstring_parser 会把第一个段落之后的 `Args:` / `Returns:` 段从
         description 里剥掉 —— 只有 `Returns:` 段的 docstring，description 是 None，
         它同样进不了 schema，属于同一类错误；
      2. 纯空白的 docstring 会解析出 `'  '`（真值判定为真），必须 strip 后再比。
    """
    doc: Docstring = parse(func.__doc__ or "")
    if (doc.description or "").strip():
        return

    raw = (func.__doc__ or "").strip()
    shown = repr(raw[:60]) if raw else "<没有 docstring>"
    raise ValueError(
        f"工具 {func_name!r} 缺少 docstring 描述，拒绝注册。\n"
        f"  要求：函数 docstring 的第一段写明「做什么 / 何时用 / 何时不用」，"
        f"它会原样成为模型看到的 description。\n"
        f"  实际：{shown}"
    )


def _parse_function_doc(func, function_name: str | None = "") -> ToolFunctionDefinition:
    doc: Docstring = parse(func.__doc__ or "")
    doc_params_map = {p.arg_name: _parse_function_parameter(p) for p in doc.params}

    function_parameter = FunctionParameter(
        properties=doc_params_map,
        required=[p.arg_name for p in doc.params if not p.is_optional],
    )

    return ToolFunctionDefinition(
        name=(function_name or func.__name__),
        description=doc.description,
        parameters=function_parameter,
    )


def _parse_function_parameter(param: DocstringParam) -> dict:
    import re

    LIST_PATTERN = re.compile(r"list\[(.*)\]")

    type_name = param.type_name
    m = LIST_PATTERN.match(type_name)
    if m:
        type_name = m.group(1)
        return {
            "type": "array",
            "description": param.description,
            "items": {"type": type_name, "description": ""},
        }
    else:
        return {
            "type": type_name,
            "description": param.description,
        }


dispatcher = ToolDispatcher()


if __name__ == "__main__":
    schema = dispatcher.build_schema()
    print(schema)
