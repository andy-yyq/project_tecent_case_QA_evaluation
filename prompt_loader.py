# -*- coding: utf-8 -*-
"""从 prompts 目录读取母版，并用 dict 指定字段替换占位符"""

import re
from pathlib import Path
from typing import Dict, Any

def load_prompt_template(system_prompt_path: Path, query_prompt_path: Path) -> tuple[str,str]:
    """读取 .txt 模板内容。"""
    if not system_prompt_path.exists():
        raise FileNotFoundError(f"模板不存在: {system_prompt_path}")
    if not query_prompt_path.exists():
        raise FileNotFoundError(f"模板不存在: {query_prompt_path}")
    system_prompt_template = system_prompt_path.read_text(encoding="utf-8").strip()
    query_prompt_template = query_prompt_path.read_text(encoding="utf-8").strip()
    return system_prompt_template, query_prompt_template


def get_placeholder_values(item: Dict[str, Any], prompt_placeholder_relation: Dict) -> Dict[str, Any]:
    """
    从 list 元素的 dict 中按 prompt_placeholder_relation 提取值。
    占位符名为 key，取 item[prompt_placeholder_relation[key]]，缺失则用空字符串。
    """
    values = {}
    for placeholder_name, field_key in prompt_placeholder_relation.items():
        values[placeholder_name] = str(item.get(field_key, ""))
    return values


def replace_placeholders(template: str, values: Dict[str, Any]) -> str:
    """将模板中 {{name}} 替换为 values.get("name", "")。"""
    def repl(m: re.Match) -> str:
        return values.get(m.group(1), "")
    return re.sub(r"\{\{(\w+)\}\}", repl, template)


def build_prompt(prompt_template:str, item: Dict[str, Any], prompt_placeholder_relation:Dict) -> str:
    """根据当前 item 替换指定掉标记位，生成最终 prompt。"""
    values = get_placeholder_values(item, prompt_placeholder_relation)
    return replace_placeholders(prompt_template, values)
