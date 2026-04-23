# -*- coding: utf-8 -*-
"""
入口：配置 loguru、加载数据、按批次异步调用大模型，
结果校验与重试，结果汇总到 Tasks_Results。
"""
import os
from dotenv import load_dotenv
from pre_handle import r1_task_prepare_handle, r2_task_prepare_handle
import settings
from runner import asyncio_run_one_round
from post_handle import r1_task_result_handle, r2_task_result_handle
from prompt_loader import load_prompt_template
from utilities import setup_logging
from loguru import logger


if __name__ == "__main__":
    log_path = setup_logging()
    logger.info(f"日志文件: {log_path}")

    load_dotenv()
    # # --- Round One ---
    r1_base_url = os.getenv("LLM_ALI_BASE_URL")
    r1_api_key = os.getenv("LLM_ALI_API_KEY")
    r1_model_name = "qwen3-max"
    r1_system_prompt_path = getattr(settings, "R1_SYSTEM_PROMPT_FILE", None)
    r1_query_prompt_path = getattr(settings, "R1_QUERY_PROMPT_FILE", None)
    r1_system_prompt_template, r1_query_prompt_template = load_prompt_template(
        r1_system_prompt_path, r1_query_prompt_path)
    r1_result_type = getattr(settings, "R1_RESULT_TYPE", "string")
    r1_result_dataclass = getattr(settings, "R1_RESULT_DATACLASS", None)
    r1_prompt_placeholder_relation = getattr(settings, "R1_PROMPT_PLACEHOLDER_RELATIONSHIP", None)

    if not (r1_api_key or r1_base_url or r1_model_name):
        logger.error(f"未设置模型配置, 请在 .env 中配置")
        raise SystemExit(1)

    r1_task_list = r1_task_prepare_handle()
    r1_task_result = asyncio_run_one_round(
        r1_task_list, r1_base_url, r1_api_key, r1_model_name,
        r1_system_prompt_template, r1_query_prompt_template, r1_prompt_placeholder_relation,
        r1_result_type, r1_result_dataclass, logger)
    r1_task_result_handle(r1_task_result)

    # --- Round Two ---
    r2_base_url = os.getenv("LLM_ALI_BASE_URL")
    r2_api_key = os.getenv("LLM_ALI_API_KEY")
    r2_model_name = "qwen3-max"
    r2_system_prompt_path = getattr(settings, "R2_SYSTEM_PROMPT_FILE", None)
    r2_query_prompt_path = getattr(settings, "R2_QUERY_PROMPT_FILE", None)
    r2_system_prompt_template, r2_query_prompt_template = load_prompt_template(
        r2_system_prompt_path, r2_query_prompt_path)
    r2_result_type = getattr(settings, "R2_RESULT_TYPE", "string")
    r2_result_dataclass = getattr(settings, "R2_RESULT_DATACLASS", None)
    r2_prompt_placeholder_relation = getattr(settings, "R2_PROMPT_PLACEHOLDER_RELATIONSHIP", None)

    if not (r2_api_key or r2_base_url or r2_model_name):
        logger.error(f"未设置模型配置, 请在 .env 中配置")
        raise SystemExit(1)

    r2_task_list = r2_task_prepare_handle()
    r2_task_result = asyncio_run_one_round(
        r2_task_list, r2_base_url, r2_api_key, r2_model_name,
        r2_system_prompt_template, r2_query_prompt_template, r2_prompt_placeholder_relation,
        r2_result_type, r2_result_dataclass, logger)
    r2_task_result_handle(r2_task_result)