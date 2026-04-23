# -*- coding: utf-8 -*-
"""根据配置的批次大小，将 list[dict] 组装为任务批次组"""

from typing import List, Dict, Any

from settings import BATCH_SIZE
from prompt_loader import build_prompt
from models import TaskItem
from utilities import short_id


def build_task_id(index: int, item: Dict[str, Any]) -> str:
    """生成任务 ID，可改为从 item 中取唯一键。"""
    return f"taskID_{short_id(length=10)}"


def item_to_task(index: int, item: Dict[str, Any],
                 system_prompt_template:str, query_prompt_template:str, prompt_placeholder_relation:Dict) -> TaskItem:
    """将 list 的一个元素转为 TaskItem。"""
    task_id = build_task_id(index, item)
    system_prompt = build_prompt(system_prompt_template, item, prompt_placeholder_relation)
    query_prompt = build_prompt(query_prompt_template, item, prompt_placeholder_relation)
    return TaskItem(
        task_id=task_id,
        system_prompt=system_prompt,
        query_prompt=query_prompt,
        raw_item=item,
    )


def build_batches(data: List[Dict[str, Any]], system_prompt_template:str, query_prompt_template:str,
                  prompt_placeholder_relation:Dict) -> List[List[TaskItem]]:
    """
    按 BATCH_SIZE 将 data 转为任务列表，并切分为批次。
    每个批次为 List[TaskItem]。
    """
    tasks = [item_to_task(i, item, system_prompt_template, query_prompt_template, prompt_placeholder_relation)
             for i, item in enumerate(data)]
    batches = [
        tasks[i : i + BATCH_SIZE]
        for i in range(0, len(tasks), BATCH_SIZE)
    ]
    return batches
