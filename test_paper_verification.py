# -*- coding: utf-8 -*-
"""
试卷 / 问答审核页：从 casebase_qa_evaluation 浏览与人工审核，写回 final_check。
运行：streamlit run test_paper_verification.py
"""

from __future__ import annotations

import json
import random
import hashlib
import os
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

import settings
from utilities import get_current_time, mysql_query, mysql_update


def _escape_sql_literal(value: str) -> str:
    return (value or "").replace("\\", "\\\\").replace("'", "''")

def _sha256_hex(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def _load_auth_users() -> Dict[str, str]:
    """
    读取本地账号配置：
    - 优先读取 auth_users.json（不要提交到仓库）
    - 若不存在，则读取 auth_users.json（示例）

    支持两种格式：
    1) {"users": [{"username": "...", "password": "..."}]}
    2) {"users": [{"username": "...", "password_sha256": "..."}]}
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "auth_users.json"),
        os.path.join(base_dir, "auth_users.json"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if not path:
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    users = data.get("users") if isinstance(data, dict) else None
    if not isinstance(users, list):
        return {}

    mapping: Dict[str, str] = {}
    for u in users:
        if not isinstance(u, dict):
            continue
        username = str(u.get("username") or "").strip()
        if not username:
            continue
        if u.get("password_sha256"):
            mapping[username] = str(u.get("password_sha256") or "").strip().lower()
        elif u.get("password") is not None:
            mapping[username] = _sha256_hex(str(u.get("password") or ""))
    return mapping


def _require_login() -> None:
    if "tpv_authed" not in st.session_state:
        st.session_state.tpv_authed = False
    if "tpv_authed_user" not in st.session_state:
        st.session_state.tpv_authed_user = None

    if st.session_state.tpv_authed:
        return

    st.title("评估集审核")
    st.info("请先登录后再使用。")

    users = _load_auth_users()
    if not users:
        st.error("未找到可用的账号配置，请先创建 auth_users.json（可参考 auth_users.json）。")
        st.stop()

    with st.form("tpv_login_form", clear_on_submit=False):
        username = st.text_input("账号", value="", placeholder="请输入账号")
        password = st.text_input("密码", value="", placeholder="请输入密码", type="password")
        submit = st.form_submit_button("登录", type="primary")
        if submit:
            u = (username or "").strip()
            p = password or ""
            if not u:
                st.warning("请输入账号")
                st.stop()
            if u not in users:
                st.error("账号或密码错误")
                st.stop()
            if _sha256_hex(p) != users[u]:
                st.error("账号或密码错误")
                st.stop()
            st.session_state.tpv_authed = True
            st.session_state.tpv_authed_user = u
            st.rerun()

    st.stop()


def fetch_all_evaluation_rows() -> Tuple[bool, List[Dict[str, Any]], str]:
    sql = (
        "SELECT id, question_id, question, question_case_id, ground_truth, quote_info, ori_answer, final_check, queue_name "
        "FROM casebase_qa_evaluation WHERE case_opinion = '保留' AND final_check is NULL ORDER BY id"
    )
    ok, rows, msg = mysql_query(sql_query=sql, mysql_config=settings.MYSQL_DEV_CONFIG)
    if not ok:
        return False, [], msg or "查询失败"
    cols = ["id", "question_id", "question", "question_case_id", "ground_truth", "quote_info", "ori_answer", "final_check", "queue_name"]
    records = [dict(zip(cols, r)) for r in (rows or [])]
    return True, records, ""

def fetch_pass_summary_by_queue() -> Tuple[bool, List[Dict[str, Any]], str]:
    sql = (
        "SELECT "
        "  COALESCE(queue_name, '（空）') AS queue_name, "
        "  SUM(CASE WHEN TRIM(COALESCE(final_check, '')) = '通过' THEN 1 ELSE 0 END) AS pass_cnt, "
        "  SUM(CASE WHEN TRIM(COALESCE(final_check, '')) = '不通过' THEN 1 ELSE 0 END) AS fail_cnt, "
        "  SUM(CASE WHEN TRIM(COALESCE(final_check, '')) = '' THEN 1 ELSE 0 END) AS pending_cnt, "
        "  COUNT(1) AS total_cnt "
        "FROM casebase_qa_evaluation "
        "WHERE case_opinion = '保留'"
        "GROUP BY COALESCE(queue_name, '（空）') "
        "ORDER BY total_cnt DESC, pass_cnt DESC, fail_cnt DESC, queue_name ASC"
    )
    ok, rows, msg = mysql_query(sql_query=sql, mysql_config=settings.MYSQL_DEV_CONFIG)
    if not ok:
        return False, [], msg or "查询失败"
    records = [
        {
            "queue_name": r[0],
            "pass_cnt": int(r[1] or 0),
            "fail_cnt": int(r[2] or 0),
            "pending_cnt": int(r[3] or 0),
            "total_cnt": int(r[4] or 0),
        }
        for r in (rows or [])
    ]
    return True, records, ""


def extract_quote_case_ids(ori_answer: Optional[str]) -> List[str]:
    if not ori_answer or not str(ori_answer).strip():
        return []
    try:
        data = json.loads(ori_answer)
    except (json.JSONDecodeError, TypeError):
        return []
    qs = data.get("quote_source")
    if not isinstance(qs, dict):
        return []
    return [str(k) for k in qs.keys()]


def fetch_case_content(case_id: str) -> Tuple[bool, Optional[str], str]:
    cid = _escape_sql_literal(str(case_id))
    sql = f"SELECT case_content FROM v_all_ai_manual_case_info WHERE case_id = '{cid}' LIMIT 1"
    ok, rows, msg = mysql_query(sql_query=sql, mysql_config=settings.MYSQL_PROD_CONFIG)
    if not ok:
        return False, None, msg or "生产库查询失败"
    if not rows:
        return False, None, f"未在 v_all_ai_manual_case_info 中找到 case_id = {case_id!r}"
    return True, rows[0][0], ""


def submit_final_check(record_id: int, conclusion: str, reason: str) -> Tuple[bool, str]:
    now = get_current_time()
    update_sql = (
        "UPDATE casebase_qa_evaluation "
        "SET final_check=%s, final_check_reason=%s, update_datetime=%s "
        "WHERE id=%s"
    )
    ok, msg = mysql_update(
        update_sql=update_sql,
        update_data=[(conclusion, reason, now, record_id)],
        mysql_config=settings.MYSQL_DEV_CONFIG,
    )
    if not ok:
        return False, msg or "更新失败"
    return True, ""


def _ensure_state() -> None:
    if "tpv_records" not in st.session_state:
        st.session_state.tpv_records = []
    if "tpv_index" not in st.session_state:
        st.session_state.tpv_index = 0
    if "tpv_load_error" not in st.session_state:
        st.session_state.tpv_load_error = None
    if "tpv_summary_error" not in st.session_state:
        st.session_state.tpv_summary_error = None
    if "tpv_summary_rows" not in st.session_state:
        st.session_state.tpv_summary_rows = []


def main() -> None:
    st.set_page_config(page_title="试题审核", layout="wide")
    _ensure_state()
    _require_login()

    st.title("评估集审核")
    # st.caption("开发库读取题目与答案；源案例正文从生产库 v_all_ai_manual_case_info 拉取。")

    tab_check, tab_summary = st.tabs(["Test Check", "Test Summary"])

    with tab_check:
        def _reload_check_records() -> None:
            ok, records, err = fetch_all_evaluation_rows()
            if not ok:
                st.session_state.tpv_load_error = err
                st.session_state.tpv_records = []
                st.session_state.tpv_index = 0
            else:
                st.session_state.tpv_load_error = None
                st.session_state.tpv_records = records
                st.session_state.tpv_index = 0

        col_a, col_b, col_c, col_d, col_e, col_f = st.columns([1, 1, 1.2, 1, 1, 1])
        with col_a:
            if st.button("数据加载", type="primary"):
                _reload_check_records()
                st.rerun()
        with col_b:
            if st.button("随机 1 条") and st.session_state.tpv_records:
                st.session_state.tpv_index = random.randrange(len(st.session_state.tpv_records))
                st.rerun()
        with col_c:
            qid_input = st.text_input("指定问题 ID", key="tpv_qid_input", placeholder="输入后点右侧跳转")
        with col_d:
            if st.button("跳转") and st.session_state.tpv_records:
                target = (qid_input or "").strip()
                if not target:
                    st.warning("请输入 question_id")
                else:
                    found = -1
                    for i, r in enumerate(st.session_state.tpv_records):
                        if str(r.get("question_id") or "") == target:
                            found = i
                            break
                    if found < 0:
                        st.error(f"当前列表中不存在 question_id = {target!r}，请先刷新列表或检查 ID。")
                    else:
                        st.session_state.tpv_index = found
                        st.rerun()
        with col_e:
            if st.button("Last") and st.session_state.tpv_records:
                n = len(st.session_state.tpv_records)
                st.session_state.tpv_index = (st.session_state.tpv_index - 1) % n
                st.rerun()
        with col_f:
            if st.button("Next") and st.session_state.tpv_records:
                n = len(st.session_state.tpv_records)
                st.session_state.tpv_index = (st.session_state.tpv_index + 1) % n
                st.rerun()

        if st.session_state.tpv_load_error:
            st.error(st.session_state.tpv_load_error)

        records: List[Dict[str, Any]] = st.session_state.tpv_records
        if not records:
            st.info('请点击「数据加载」拉取数据。')
            return

        idx = int(st.session_state.tpv_index) % len(records)
        st.session_state.tpv_index = idx
        row = records[idx]

        st.divider()
        meta1, meta2, meta3, meta4, meta5 = st.columns(5)
        with meta1:
            st.metric("当前序号", f"{idx + 1} / {len(records)}")
        with meta2:
            st.metric("队列名称", row.get("queue_name") or "（空）")
        with meta3:
            st.metric("案例来源", row.get("question_case_id") or "—")
        with meta4:
            st.metric("AI审核意见", row.get("final_check") or "（空）")
        with meta5:
            st.metric("问题ID", row.get("question_id") or "—")

        left, right = st.columns([1, 1])
        with left:
            st.subheader("问题描述")
            st.text_area("question", value=row.get("question") or "", height=100, disabled=True, label_visibility="collapsed")
            st.subheader("参考答案")
            st.text_area("ground_truth", value=row.get("ground_truth") or "", height=150, disabled=True, label_visibility="collapsed")
            st.subheader("参考依据")
            st.text_area("quote_info", value=row.get("quote_info") or "", height=400, disabled=True, label_visibility="collapsed")

        with right:
            st.subheader("源案例内容")
            case_ids = extract_quote_case_ids(row.get("ori_answer"))
            if not case_ids:
                st.warning("当前记录的 ori_answer 中无法解析出 quote_source 下的案例 ID（可能为空或 JSON 格式异常）。")
                selected: Optional[str] = None
            else:
                selected = st.selectbox("选择案例 ID", case_ids, key=f"tpv_case_{row['id']}")

            if selected:
                ok_c, content, err_c = fetch_case_content(selected)
                if ok_c:
                    st.text_area("案例详情", value=content or "", height=800, disabled=True)
                else:
                    st.error(err_c)

        st.divider()
        st.subheader("最终审核结论")
        conclusion = st.radio("审核结果", ("通过", "不通过"), horizontal=True, key=f"tpv_audit_{row['id']}")
        reason = st.text_area("不通过驳回理由（精简描述即可）", value="", max_chars=200, key=f"tpv_reason_{row['id']}")
        if st.button("提交审核结果", type="primary"):
            ok_u, err_u = submit_final_check(int(row["id"]), conclusion, (reason or "").strip())
            if not ok_u:
                st.error(err_u)
            else:
                st.success(f"已写入审核结果：{conclusion}（id = {row['id']}）")
                # 提交成功后：自动重载待审核列表（本列表只展示 final_check is NULL 的数据）
                _reload_check_records()
                # 同步清空 Summary 缓存，确保后续刷新/切换能拿到最新统计
                st.session_state.tpv_summary_rows = []
                st.session_state.tpv_summary_error = None
                st.rerun()

    with tab_summary:
        st.subheader("按队列统计")
        refresh = st.button("刷新统计", type="primary")
        if refresh or not st.session_state.tpv_summary_rows:
            ok_s, rows_s, err_s = fetch_pass_summary_by_queue()
            if not ok_s:
                st.session_state.tpv_summary_error = err_s
                st.session_state.tpv_summary_rows = []
            else:
                st.session_state.tpv_summary_error = None
                st.session_state.tpv_summary_rows = rows_s

        if st.session_state.tpv_summary_error:
            st.error(st.session_state.tpv_summary_error)
        else:
            rows_show: List[Dict[str, Any]] = st.session_state.tpv_summary_rows or []
            if not rows_show:
                st.info("当前表内无数据，暂无可统计的队列。")
            else:
                # 用 dataframe 展示，避免 text_input 的 key 缓存导致刷新后仍显示旧值
                st.dataframe(
                    rows_show,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "queue_name": st.column_config.TextColumn("队列名称"),
                        "pass_cnt": st.column_config.NumberColumn("通过数量"),
                        "fail_cnt": st.column_config.NumberColumn("不通过数量"),
                        "pending_cnt": st.column_config.NumberColumn("未处理数量"),
                        "total_cnt": st.column_config.NumberColumn("总数量"),
                    },
                )


if __name__ == "__main__":
    main()
