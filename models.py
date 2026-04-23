# -*- coding: utf-8 -*-
"""任务与返回结构定义、返回结果验证"""

import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any, Type, TypeVar, Dict, Union, Optional, Tuple

T = TypeVar("T")

@dataclass
class TaskItem:
    """单条任务：用于批次内传递"""
    task_id: str
    system_prompt: str
    query_prompt: str
    raw_item: dict  # 原始 list 元素，便于日志或重试时使用


@dataclass
class TaskResult_R1:
    case_opinion: str
    filter_reason: str
    question_design: str

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass
class TaskResult_R2:
    final_answer: str
    quote_source: Dict[str, str]

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})


def validate_string_result(raw: str) -> str:
    """校验返回为字符串：非空且为文本。"""
    if not isinstance(raw, str):
        raise ValueError(f"期望 str，得到 {type(raw).__name__}")
    s = raw.strip()
    if not s:
        raise ValueError("返回内容为空")
    return s


def _strip_markdown_code_fence(text: str) -> str:
    s = text.strip()
    if not (s.startswith("```") and s.endswith("```")):
        return s
    inner = s[3:-3].strip()
    if "\n" in inner:
        first_line, rest = inner.split("\n", 1)
        if first_line.strip().lower() in {"json", "javascript", "js"}:
            return rest.strip()
    return inner.strip()


def _find_first_json_span(text: str) -> Optional[Tuple[int, int]]:
    """
    在任意文本中，寻找第一段完整的 JSON 对象/数组片段的 [start, end)。
    通过括号栈 + 字符串状态机，避免被引号内的 '{' '}' 干扰。
    """
    s = text
    start = -1
    stack: list[str] = []
    in_str = False
    escape = False

    for i, ch in enumerate(s):
        if start == -1:
            if ch in "{[":
                start = i
                stack.append(ch)
            continue

        if in_str:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch in "{[":
            stack.append(ch)
            continue

        if ch in "}]":
            if not stack:
                return None
            open_ch = stack.pop()
            if (open_ch == "{" and ch != "}") or (open_ch == "[" and ch != "]"):
                return None
            if not stack:
                return (start, i + 1)

    return None


def _extract_json_text(raw: str) -> str:
    s = _strip_markdown_code_fence(raw)
    span = _find_first_json_span(s)
    if span is None:
        return s.strip()
    a, b = span
    return s[a:b].strip()


def validate_dataclass_result(raw: Union[str, dict], result_class: Type[T]) -> T:
    """校验返回可解析为指定 dataclass 并字段合法。"""
    if not is_dataclass(result_class):
        raise TypeError(f"{result_class} 不是 dataclass")

    data: Any
    if isinstance(raw, dict):
        data = raw
    else:
        if not isinstance(raw, str):
            raise ValueError(f"期望 str 或 dict，得到 {type(raw).__name__}")
        candidate = _extract_json_text(raw)
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"返回不是合法 JSON: {e} |提取后：{candidate} |模型返回：{raw}"
            )

    if not isinstance(data, dict):
        raise ValueError(f"期望 JSON 对象(dict)，得到 {type(data).__name__}")
    try:
        return result_class.from_dict(data)
    except Exception as e:
        raise ValueError(f"无法构造 {result_class.__name__}: {e}")
