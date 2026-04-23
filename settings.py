# -*- coding: utf-8 -*-
"""配置文件：批次大小、重试、模型、占位符与返回类型等"""

import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from models import TaskResult_R1, TaskResult_R2

load_dotenv()
# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# 批次与重试
BATCH_SIZE = 4
MAX_RETRIES = 2

# Prompts 目录
PROMPTS_DIR = PROJECT_ROOT / "prompts"
# ======== Round One ===================
R1_SYSTEM_PROMPT_FILE = PROMPTS_DIR / "r1_system_prompt.txt"
R1_QUERY_PROMPT_FILE = PROMPTS_DIR / "r1_query_prompt.txt"

# prompt 占位符替换规则
# 键：为占位符名，在prompt中与{{field_name}} 对应
# 值：为单个任务（dict）中重传递的 key 名称（对应到value）
R1_PROMPT_PLACEHOLDER_RELATIONSHIP = {
    "case_content": "case_content"
}

# 返回类型： "string" | "dataclass"
R1_RESULT_TYPE: Literal["string", "dataclass"] = "dataclass"
R1_RESULT_DATACLASS = TaskResult_R1

# ======== Round Two ===================
R2_SYSTEM_PROMPT_FILE = PROMPTS_DIR / "r2_system_prompt.txt"
R2_QUERY_PROMPT_FILE = PROMPTS_DIR / "r2_query_prompt.txt"

# prompt 占位符替换规则
# 键：为占位符名，在prompt中与{{field_name}} 对应
# 值：为单个任务（dict）中重传递的 key 名称（对应到value）
R2_PROMPT_PLACEHOLDER_RELATIONSHIP = {
    "question": "question",
    "case_ori_content": "case_ori_content",
    "case_refs_content": "case_refs_content"
}

# 返回类型： "string" | "dataclass"
R2_RESULT_TYPE: Literal["string", "dataclass"] = "dataclass"
R2_RESULT_DATACLASS = TaskResult_R2

# ==========================================================
# 日志目录
LOG_DIR = PROJECT_ROOT / "logs"

# 任务抽样数量（<=0 表示不抽样，使用全量）
TASK_SAMPLE_SIZE = 120

# MySQL 数据库配置（敏感信息从 .env 读取，禁止硬编码）
MYSQL_PROD_CONFIG = {
    "host": os.getenv("MYSQL_PROD_HOST", ""),
    "port": int(os.getenv("MYSQL_PROD_PORT", "3306")),
    "user": os.getenv("MYSQL_PROD_USER", ""),
    "password": os.getenv("MYSQL_PROD_PASSWORD", ""),
    "database": os.getenv("MYSQL_PROD_DATABASE", ""),
    "charset": os.getenv("MYSQL_PROD_CHARSET", "utf8mb4"),
}

MYSQL_DEV_CONFIG = {
    "host": os.getenv("MYSQL_DEV_HOST", ""),
    "port": int(os.getenv("MYSQL_DEV_PORT", "3306")),
    "user": os.getenv("MYSQL_DEV_USER", ""),
    "password": os.getenv("MYSQL_DEV_PASSWORD", ""),
    "database": os.getenv("MYSQL_DEV_DATABASE", ""),
    "charset": os.getenv("MYSQL_DEV_CHARSET", "utf8mb4"),
}

QDRANT_CONFIG = {
    "url": os.getenv("QDRANT_URL", ""),
    "grpc_port": os.getenv("QDRANT_GRPC_PORT", ""),
    "api_key": os.getenv("QDRANT_API_KEY", ""),
    "timeout": 30
}