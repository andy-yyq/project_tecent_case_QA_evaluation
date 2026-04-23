import json
from typing import Any, Dict, List, Tuple

from loguru import logger

import settings
from utilities import get_current_time, mysql_insert, mysql_query, mysql_update, short_id


def _escape_sql_string_literal(value: str) -> str:
    """用于 mysql_query 的 {占位符} 拼接时，对字符串字面量做最小转义。"""
    return (value or "").replace("\\", "\\\\").replace("'", "''")


def _build_quote_info(quote_source: Any) -> Tuple[str, int]:
    """将 quote_source 格式化为 quote_info 文本，并返回条目数量。"""
    if not quote_source or not isinstance(quote_source, dict):
        return "", 0
    blocks = []
    for case_id, basis in quote_source.items():
        blocks.append(f"参考案例：{case_id}\n参考依据：{basis}")
    text = "\n\n".join(blocks)
    return text, len(quote_source)


def r1_task_result_handle(Tasks_Results: Dict[str, Dict[str, Any]]) -> None:
    """
    将 r1 产物写入 dev 库表 casebase_qa_evaluation。

    Tasks_Results: dict，key 为 task_id，value 结构：
    {
        "case_id": "...",
        "case_content": "...",
        "llm_result": {
            "case_opinion": "...",
            "filter_reason": "...",
            "question_design": "...",
        }
    }

    字段映射：
    - llm_result.question_design -> question
    - case_id -> question_case_id
    - llm_result.case_opinion -> case_opinion
    - llm_result.filter_reason -> filter_reason
    """
    if not Tasks_Results:
        logger.info("post_handle r1_task_result_handle: Tasks_Results 为空，跳过写库")
        return

    # 先从 prod 库 case_data 反查 queue_name（尽量批量查询，异常仅记录日志不影响后续写入）
    queue_name_by_case_id: Dict[str, Any] = {}
    try:
        case_ids: List[str] = []
        for _task_id, _case_info in (Tasks_Results or {}).items():
            _cid = _case_info.get("case_id")
            if _cid:
                case_ids.append(str(_cid))
        unique_case_ids = list(dict.fromkeys(case_ids))

        if unique_case_ids:
            in_list = ",".join(f"'{_escape_sql_string_literal(c)}'" for c in unique_case_ids)
            q_sql = f"SELECT case_number, queue_name FROM case_data WHERE case_number IN ({in_list})"
            ok_q, rows_q, msg_q = mysql_query(sql_query=q_sql, mysql_config=settings.MYSQL_PROD_CONFIG)
            if not ok_q:
                logger.error(f"prod 库查询 case_data.queue_name 失败: {msg_q}")
            else:
                counts: Dict[str, int] = {}
                values: Dict[str, Any] = {}
                for r in (rows_q or []):
                    # 期望 (case_id, queue_name)
                    cid = str(r[0])
                    qn = r[1] if len(r) > 1 else None
                    counts[cid] = counts.get(cid, 0) + 1
                    # 先记录一份，重复的情况后面会整体置空并打日志
                    if cid not in values:
                        values[cid] = qn

                for cid in unique_case_ids:
                    cnt = counts.get(cid, 0)
                    if cnt == 1:
                        queue_name_by_case_id[cid] = values.get(cid)
                    elif cnt == 0:
                        logger.error(f"prod 库 case_data 未找到 case_id={cid} 的 queue_name")
                        queue_name_by_case_id[cid] = None
                    else:
                        logger.error(f"prod 库 case_data case_id={cid} 命中 {cnt} 条，期望 1 条")
                        queue_name_by_case_id[cid] = None
    except Exception as e:
        logger.exception(f"反查 prod 库 queue_name 异常（将跳过 queue_name 写入）: {e}")

    insert_sql = (
        "INSERT INTO casebase_qa_evaluation "
        "(question_id, question, question_case_id, case_opinion, filter_reason, queue_name, update_datetime) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)"
    )

    now = get_current_time()
    insert_data = []
    for task_id, case_info in (Tasks_Results or {}).items():
        case_id = case_info.get("case_id")
        llm_result = case_info.get("llm_result") or {}
        question_id = short_id(length=10)
        question = llm_result.get("question_design")
        case_opinion = llm_result.get("case_opinion")
        filter_reason = llm_result.get("filter_reason")

        # 保底：无 case_id 的数据不入库（无法追溯来源）
        if not case_id:
            logger.warning(f"task_id={task_id} 缺少 case_id，跳过写入")
            continue

        case_id_s = str(case_id)
        queue_name = queue_name_by_case_id.get(case_id_s)
        insert_data.append((question_id, question, case_id_s, case_opinion, filter_reason, queue_name, now))

    if not insert_data:
        logger.info("post_handle r1_task_result_handle: 无有效数据写入（均被跳过）")
        return

    ok, msg = mysql_insert(insert_sql=insert_sql, insert_data=insert_data, mysql_config=settings.MYSQL_DEV_CONFIG)
    if not ok:
        raise RuntimeError(f"MySQL 写入失败(casebase_qa_evaluation): {msg}")

    logger.info(f"post_handle r1_task_result_handle: 写入 casebase_qa_evaluation 成功，共 {len(insert_data)} 条")


def r2_task_result_handle(Tasks_Results: Dict[str, Dict[str, Any]]) -> None:
    """
    将步骤 2（r2）LLM 产物按 question_id 回写 dev 库表 casebase_qa_evaluation。

    Tasks_Results: dict，key 为 task_id，value 含 question_id、question、case_ori_content、
    case_refs_content、llm_result（含 final_answer、quote_source）等。
    """
    if not Tasks_Results:
        logger.info("post_handle r2_task_result_handle: Tasks_Results 为空，跳过写库")
        return

    question_ids_ordered: List[str] = []
    for task_id, row in Tasks_Results.items():
        qid = row.get("question_id")
        if not qid:
            raise RuntimeError(f"task_id={task_id} 缺少 question_id，终止")
        question_ids_ordered.append(str(qid))

    unique_qids = list(dict.fromkeys(question_ids_ordered))
    in_list = ",".join(f"'{_escape_sql_string_literal(q)}'" for q in unique_qids)
    check_sql = f"SELECT question_id FROM casebase_qa_evaluation WHERE question_id IN ({in_list})"
    ok, rows, msg = mysql_query(sql_query=check_sql, mysql_config=settings.MYSQL_DEV_CONFIG)
    if not ok:
        raise RuntimeError(f"MySQL 查询失败(casebase_qa_evaluation): {msg}")

    found = {r[0] for r in (rows or [])}
    missing = [q for q in unique_qids if q not in found]
    if missing:
        raise RuntimeError(
            f"casebase_qa_evaluation 中未找到 question_id（共 {len(missing)} 条），终止: {missing[:20]}"
            + (" ..." if len(missing) > 20 else "")
        )

    update_sql = (
        "UPDATE casebase_qa_evaluation SET "
        "ori_answer=%s, ground_truth=%s, quote_info=%s, quote_num=%s, update_datetime=%s "
        "WHERE question_id=%s"
    )

    now = get_current_time()
    update_data = []
    for task_id, row in Tasks_Results.items():
        question_id = str(row["question_id"])
        llm_result = row.get("llm_result")
        if llm_result is None:
            llm_result = {}
        elif not isinstance(llm_result, dict):
            raise RuntimeError(f"task_id={task_id} llm_result 不是 dict，终止")

        try:
            ori_answer = json.dumps(llm_result, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"task_id={task_id} llm_result 无法序列化为 JSON: {e}") from e

        ground_truth = llm_result.get("final_answer")
        quote_info, quote_num = _build_quote_info(llm_result.get("quote_source"))
        update_data.append((ori_answer, ground_truth, quote_info, quote_num, now, question_id))

    ok_u, msg_u = mysql_update(
        update_sql=update_sql, update_data=update_data, mysql_config=settings.MYSQL_DEV_CONFIG
    )
    if not ok_u:
        raise RuntimeError(f"MySQL 更新失败(casebase_qa_evaluation): {msg_u}")

    logger.info(f"post_handle r2_task_result_handle: 更新 casebase_qa_evaluation 成功，共 {len(update_data)} 条")
