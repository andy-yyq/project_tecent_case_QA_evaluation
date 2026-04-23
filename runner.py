# -*- coding: utf-8 -*-
"""异步调用大模型、结果校验、批次内重试与统计"""

import asyncio
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from loguru._logger import Logger
from openai import AsyncOpenAI

from settings import MAX_RETRIES
from models import TaskItem, validate_string_result, validate_dataclass_result

from task_builder import build_batches

def get_client(base_url:str, api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )


async def call_one_task(
    client: AsyncOpenAI,
    model_name: str,
    task: TaskItem,
    result_type: str,
    result_dataclass: Optional[type],
) -> Tuple[str, Any, int, Optional[str]]:
    """
    执行单条任务。返回 (task_id, 校验后的结果, usage_total_tokens, error_msg)。
    若成功 error_msg 为 None；失败时结果为 None，error_msg 为原因。
    """
    try:
        resp = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": task.system_prompt},
                {"role": "user", "content": task.query_prompt},
            ],
        )
    except Exception as e:
        return task.task_id, None, 0, f"大模型调用失败: {e}"

    choice = resp.choices[0] if resp.choices else None
    if not choice or not getattr(choice, "message", None):
        return task.task_id, None, 0, "大模型返回无有效 message"

    raw = getattr(choice.message, "content", None) or ""
    tokens = 0
    if getattr(resp, "usage", None):
        tokens = getattr(resp.usage, "total_tokens", 0) or 0

    try:
        if result_type == "string":
            value = validate_string_result(raw)
        elif result_type == "dataclass" and result_dataclass is not None:
            value = validate_dataclass_result(raw, result_dataclass)
            value = asdict(value) if is_dataclass(value) else value
        else:
            value = validate_string_result(raw)
    except Exception as e:
        return task.task_id, None, tokens, f"返回内容验证失败: {e}"

    return task.task_id, value, tokens, None


async def run_batch(
    client: AsyncOpenAI,
    model_name: str,
    batch: List[TaskItem],
    result_type: str,
    result_dataclass: Optional[type],
    logger: Logger
) -> Tuple[List[Tuple[str, Any]], int, int, int, int]:
    """
    运行一个批次，失败则重试（最多 MAX_RETRIES 次）。
    返回 (成功列表 [(task_id, value), ...], 成功数, 调用失败数, 验证失败数, 总 token)。
    若某任务超过最大重试仍失败，抛出 RuntimeError，由上层退出程序。
    """
    completed: List[Tuple[str, Any]] = []
    api_fail_count = 0
    validation_fail_count = 0
    total_tokens = 0
    pending = list(batch)

    for attempt in range(MAX_RETRIES + 1):
        if not pending:
            break
        coros = [
            call_one_task(client, model_name, t, result_type, result_dataclass)
            for t in pending
        ]
        results = await asyncio.gather(*coros, return_exceptions=False)
        next_pending: List[TaskItem] = []
        for task, (task_id, value, tokens, err) in zip(pending, results):
            total_tokens += tokens
            if err is None:
                completed.append((task_id, value))
            else:
                if "大模型调用失败" in err or "返回无有效" in err:
                    api_fail_count += 1
                else:
                    validation_fail_count += 1
                logger.warning(f"任务 {task_id} 失败 (尝试 {attempt + 1}/{MAX_RETRIES + 1}): {err}")
                next_pending.append(task)
        pending = next_pending
        if pending and attempt < MAX_RETRIES:
            await asyncio.sleep(1)

    if pending:
        raise RuntimeError(
            f"以下任务在 {MAX_RETRIES + 1} 次重试后仍失败: {[t.task_id for t in pending]}"
        )

    return completed, len(completed), api_fail_count, validation_fail_count, total_tokens


async def one_round(task_list: List, base_url:str, api_key:str, model_name: str,
                    system_prompt_template:str, query_prompt_template:str, prompt_placeholder_relation:Dict,
                    result_type:str, result_dataclass:Optional[type], logger: Logger) -> Dict:
    """按批次执行全部任务，结果写入全局 Tasks_Results。"""
    Tasks_Results = {}
    Prompt_Simple = True
    Result_Simple = True

    total = len(task_list)
    start_time = datetime.now()
    logger.info(f"任务开始时间: {start_time.isoformat()}")
    logger.info(f"任务总数量: {total}")

    batches = build_batches(task_list, system_prompt_template, query_prompt_template, prompt_placeholder_relation)
    client = get_client(base_url, api_key)

    done_count = 0
    total_tokens = 0
    total_api_fails = 0
    total_validation_fails = 0

    for batch_idx, batch in enumerate(batches):
        if Prompt_Simple:
            TaskItem_sample = batch[0]
            print(f"\n ======================================")
            print(f"            ROUND PROMPT SAMPLE        ")
            print(f" ======================================\n")
            print(f"System Prompt\n{TaskItem_sample.system_prompt}")
            print(f"\n******************************\n")
            print(f"Query Prompt\n{TaskItem_sample.query_prompt}")
            Prompt_Simple = False
        batch_start = datetime.now()
        raw_by_id = {t.task_id: t.raw_item for t in batch}
        try:
            completed, success_count, api_fails, validation_fails, tokens = await run_batch(
                client, model_name, batch, result_type, result_dataclass, logger
            )
        except RuntimeError as e:
            logger.error(str(e))
            raise SystemExit(1)
        for task_id, value in completed:
            original = raw_by_id.get(task_id, {})
            if isinstance(original, dict):
                Tasks_Results[task_id] = {**original, "llm_result": value}
            else:
                Tasks_Results[task_id] = {"task_id": task_id, "llm_result": value}

            if Result_Simple:
                print(f"\n ========== ROUND RESULT SAMPLE ============\n")
                print(f"Task ID: {task_id}")
                print(f"Task Result:")
                for k, v in Tasks_Results[task_id].items():
                    if isinstance(v, str):
                        print(f" * Key:{k} => {v[:50]}")
                    if isinstance(v, dict):
                        print(f" * Key:{k}:")
                        for sub_k, sub_v in v.items():
                            print(f"   ** Sub Key:{sub_k} => {sub_v}")
                print(f"\n ===========================================\n")
                Result_Simple = False

        done_count += success_count
        total_tokens += tokens
        total_api_fails += api_fails
        total_validation_fails += validation_fails
        batch_elapsed = (datetime.now() - batch_start).total_seconds()
        logger.info(
            f"批次 {batch_idx + 1}/{len(batches)} 完成 | "
            f"已完成数量: {done_count}/{total} | "
            f"本批次大模型调用失败次数: {api_fails} | "
            f"本批次返回内容验证失败次数: {validation_fails} | "
            f"本批次 token 消耗: {tokens} | "
            f"本批次耗时: {batch_elapsed:.2f}s"
        )

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(
        f"全部完成 | 任务总数: {len(Tasks_Results)} | "
        f"总耗时: {elapsed:.2f}s | "
        f"总 token: {total_tokens} | "
        f"总调用失败次数: {total_api_fails} | "
        f"总验证失败次数: {total_validation_fails}"
    )



    return Tasks_Results

def asyncio_run_one_round(task_list, base_url, api_key, model_name,
                          system_prompt_template, query_prompt_template, prompt_placeholder_relation,
                          result_type, result_dataclass, logger) -> Dict:
    Tasks_Results = asyncio.run(one_round(task_list, base_url, api_key, model_name,
                                          system_prompt_template, query_prompt_template, prompt_placeholder_relation,
                                          result_type, result_dataclass, logger))
    return Tasks_Results

