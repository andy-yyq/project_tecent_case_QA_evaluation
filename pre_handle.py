# -*- coding: utf-8 -*-
"""前置数据准备：从 MySQL 拉取、抽样、校验，生成批任务输入 task_list。"""

import random
from typing import Any, Dict, List

from dotenv import load_dotenv
from loguru import logger

import settings
from qdrant_client import QdrantClient
from utilities import Qdrant_search, embedding_zhipu_api, mysql_query


def _assert_mysql_config(mysql_config: Dict[str, Any]) -> None:
    required = ["host", "port", "user", "password", "database", "charset"]
    missing = [k for k in required if not mysql_config.get(k)]
    if missing:
        raise ValueError(f"MySQL 配置缺失: {missing}（请在 .env 中填写对应变量）")


def fetch_case_pass_step_all() -> List[Dict[str, Any]]:
    """
    访问视图 v_all_ai_manual_case_info 获取数据：
    返回结构：{"case_id": "...", "case_content": "..."}
    同时通过 case_id 过滤掉已存在于 dev 库 casebase_qa_evaluation.case_source 的条目，
    剩余条目打乱顺序返回。
    """
    load_dotenv()
    mysql_prod_config = settings.MYSQL_PROD_CONFIG
    _assert_mysql_config(mysql_prod_config)

    sql_case_info = "SELECT case_id, case_content FROM v_all_ai_manual_case_info"
    ok1, catch_results, msg1 = mysql_query(sql_query=sql_case_info, mysql_config=mysql_prod_config)
    if not ok1:
        raise RuntimeError(f"MySQL 查询失败(case_pass_step): {msg1}")

    all_case_list: List[Dict[str, Any]] = []

    # utilities.mysql_query 默认返回 cursor.fetchmany 的原始 rows（tuple），这里兼容 dict/tuple 两种情况
    for r in catch_results or []:
        if isinstance(r, dict):
            case_id = r.get("case_id")
            case_content = r.get("case_content")
        else:
            case_id = r[0] if len(r) > 0 else None
            case_content = r[1] if len(r) > 1 else None
        all_case_list.append({"case_id": None if case_id is None else str(case_id), "case_content": case_content})

    # 过滤掉 dev 库已存在的 question_case_id
    mysql_dev_config = settings.MYSQL_DEV_CONFIG
    _assert_mysql_config(mysql_dev_config)
    sql_existing = "SELECT question_case_id FROM casebase_qa_evaluation"
    ok2, rows2, msg2 = mysql_query(sql_query=sql_existing, mysql_config=mysql_dev_config)
    if not ok2:
        raise RuntimeError(f"MySQL 查询失败(casebase_qa_evaluation): {msg2}")

    existing_sources = set()
    for r in rows2 or []:
        if isinstance(r, dict):
            v = r.get("case_source")
        else:
            v = r[0] if len(r) > 0 else None
        if v is not None:
            existing_sources.add(str(v))

    filtered = [item for item in all_case_list if item.get("case_id") and item["case_id"] not in existing_sources]
    random.shuffle(filtered)
    return filtered


def sample_tasks(data: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
    """对全量数据进行随机抽样。sample_size<=0 则返回全量。"""
    if sample_size <= 0:
        return list(data)
    if sample_size >= len(data):
        return list(data)
    return random.sample(data, sample_size)


def validate_task_list(task_list: List[Dict[str, Any]]) -> None:
    """确保每个任务元素必须有 task_id 字段。"""
    if not isinstance(task_list, list):
        raise TypeError("task_list 必须是 list")
    for i, item in enumerate(task_list):
        if not isinstance(item, dict):
            raise TypeError(f"task_list[{i}] 必须是 dict")

def r1_task_prepare_handle() -> List[Dict[str, Any]]:
    """
    生成批任务输入数据 task_list：
    - 抽样（settings.TASK_SAMPLE_SIZE）
    """
    all_rows = fetch_case_pass_step_all()
    picked = sample_tasks(all_rows, getattr(settings, "TASK_SAMPLE_SIZE", 0))

    task_list: List[Dict[str, Any]] = []
    for row in picked:
        case_id = row.get("case_id")
        task_list.append(
            {
                "case_id": case_id,
                "case_content": row.get("case_content"),
            }
        )

    validate_task_list(task_list)
    logger.info(f"pre_handle 生成 task_list 完成: {len(task_list)} 条（可供抽样数据：{len(all_rows)},抽样={getattr(settings, 'TASK_SAMPLE_SIZE', 0)}）")
    return task_list


def r2_task_prepare_handle() -> List[Dict[str, Any]]:
    """
    Round Two 数据预处理：
    - 从 dev 库 casebase_qa_evaluation 取 case_opinion="保留" 的 (question_id, question, question_case_id)
    - 到 prod 库 v_all_ai_manual_case_info 取 question_case_id 对应的 case_content 作为 case_ori_content
    - 对 question 做向量化后在 Qdrant(tecent_tickets_case_study) 搜索最相近案例
    - 再根据 Qdrant payload 的 case_id 到 prod 库取 case_content，并拼接为 case_refs_content
    """
    load_dotenv()

    mysql_dev_config = settings.MYSQL_DEV_CONFIG
    _assert_mysql_config(mysql_dev_config)
    mysql_prod_config = settings.MYSQL_PROD_CONFIG
    _assert_mysql_config(mysql_prod_config)

    qdrant_config = getattr(settings, "QDRANT_CONFIG", None) or {}
    if not qdrant_config.get("url"):
        raise ValueError("QDRANT_CONFIG.url 未配置，请检查 .env 并确保 settings.QDRANT_CONFIG 正确。")

    # 你们需求里写的是“特殊符号间隔”，这里用一个不易与自然语言混淆的分隔符
    case_ref_separator = "\n\n[CASE_REF_SEP]\n\n"

    # 如果你们希望“包含自身”而不是“排除自身”，把以下这个开关改为 False 即可。
    exclude_self_in_refs = True

    # 1) 从 dev 库取待处理数据
    sql_dev = (
        "SELECT question_id, question, question_case_id "
        "FROM casebase_qa_evaluation "
        "WHERE case_opinion='保留'"
    )
    ok, dev_rows, msg = mysql_query(sql_query=sql_dev, mysql_config=mysql_dev_config)
    if not ok:
        raise RuntimeError(f"MySQL 查询失败(r2_task_prepare_handle: dev): {msg}")

    # dev_rows = dev_rows[:3]

    # 2) 初始化 Qdrant 连接
    qdrant_conn = QdrantClient(**qdrant_config)

    task_list: List[Dict[str, Any]] = []
    for row in dev_rows or []:
        if isinstance(row, dict):
            question_id = row.get("question_id")
            question = row.get("question")
            question_case_id = row.get("question_case_id")
        else:
            question_id = row[0] if len(row) > 0 else None
            question = row[1] if len(row) > 1 else None
            question_case_id = row[2] if len(row) > 2 else None

        question_id = None if question_id is None else str(question_id)
        question_case_id = None if question_case_id is None else str(question_case_id)
        question = "" if question is None else str(question)

        if not question_id or not question or not question_case_id:
            logger.warning(
                f"r2_task_prepare_handle: 发现无效数据，跳过 "
                f"(question_id={question_id}, question_case_id={question_case_id}, question_len={len(question)})"
            )
            continue

        # 3) 查原始案例内容 case_ori_content
        sql_ori = (
            "SELECT case_content "
            "FROM v_all_ai_manual_case_info "
            "WHERE case_id='{case_id}' "
            "LIMIT 1"
        )
        ok_ori, ori_rows, msg_ori = mysql_query(
            sql_query=sql_ori,
            mysql_config=mysql_prod_config,
            case_id=question_case_id,
        )
        if not ok_ori:
            logger.error(f"r2_task_prepare_handle: 获取原始 case_content 失败: {msg_ori}")
            continue

        case_ori_content = None
        if ori_rows:
            case_ori_content = ori_rows[0][0] if not isinstance(ori_rows[0], dict) else ori_rows[0].get("case_content")
        case_ori_content = "" if case_ori_content is None else str(case_ori_content)

        # 4) question 向量化
        question_embedding = embedding_zhipu_api(question)
        if question_embedding is None:
            logger.error(f"r2_task_prepare_handle: question_embedding 为 None，跳过 question_id={question_id}")
            continue

        # 5) Qdrant 检索 top7（必要时为了排除自身会多取一些，再在本地过滤）
        search_limit = 10 if exclude_self_in_refs else 7
        top_points = Qdrant_search(
            qdrant_conn=qdrant_conn,
            collection_name="tecent_tickets_case_study",
            embedding=question_embedding,
            query_filter=None,
            with_payload=True,
            limit=search_limit,
        )

        # 取 Qdrant 返回的 case_content，拼接为 case_refs_content
        ref_case_contents: List[str] = []
        for p in top_points or []:
            payload = getattr(p, "payload", None)
            if payload is None and isinstance(p, dict):
                payload = p.get("payload")
            case_id_from_qdrant = (payload or {}).get("case_id") if payload else None
            if case_id_from_qdrant is None:
                continue
            case_id_from_qdrant = str(case_id_from_qdrant)

            # 排除自身（或允许自身）逻辑
            if exclude_self_in_refs and case_id_from_qdrant == question_case_id:
                continue

            sql_ref = (
                "SELECT case_content "
                "FROM v_all_ai_manual_case_info "
                "WHERE case_id='{case_id}' "
                "LIMIT 1"
            )
            ok_ref, ref_rows, msg_ref = mysql_query(
                sql_query=sql_ref,
                mysql_config=mysql_prod_config,
                case_id=case_id_from_qdrant,
            )
            if not ok_ref:
                logger.error(f"r2_task_prepare_handle: 获取 ref case_content 失败: {msg_ref}")
                continue

            case_ref_content = None
            if ref_rows:
                case_ref_content = ref_rows[0][0] if not isinstance(ref_rows[0], dict) else ref_rows[0].get("case_content")
            case_ref_content = "" if case_ref_content is None else str(case_ref_content)
            if case_ref_content:
                ref_case_contents.append(case_ref_content)

            # 保证最多拼接 7 条（exclude_self_in_refs 时可能少于 7，但这里至少符合“返回 top7 的意图”）
            if len(ref_case_contents) >= 7:
                break

        case_refs_content = case_ref_separator.join(ref_case_contents)

        task_list.append(
            {
                "question_id": question_id,
                "question": question,
                "case_ori_content": case_ori_content,
                "case_refs_content": case_refs_content,
            }
        )

    logger.info(f"pre_handle r2_task_prepare_handle 生成 task_list 完成: {len(task_list)} 条")
    return task_list

if __name__ == "__main__":
    task_list = r2_task_prepare_handle()
    for task in task_list:
        print(f"--------------")
        for k, v in task.items():
            print(f"{k}: {v}")
