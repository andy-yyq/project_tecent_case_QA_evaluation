import time

from utilities import (get_specific_evaluation_data,
                       run_async_evaluation)
from settings import MYSQL_DEV_CONFIG
from utilities import mysql_update, get_current_time


def save_results_to_db(evaluate_items_result, table_name, test_mark):
    """
    将评估结果批量更新回数据库
    :param
    evaluate_items_result: 包含测试结果的列表
    test_mark:本次测试标记
    :return: None
    """
    if not evaluate_items_result:
        print("没有数据需要更新。")
        return

    update_datetime = get_current_time()

    # 1. 准备SQL更新语句
    # 根据 question_key 匹配，更新 test_answer 和 test_ref 字段
    update_sql = f"""
        UPDATE {table_name} 
        SET test_answer = %s, test_ref = %s, test_mark = %s, update_datetime = %s 
        WHERE question_id = %s
    """

    # 2. 准备更新数据列表
    # executemany 要求的数据格式为列表套元组：[(test_answer, test_ref, test_mark, update_datetime, question_key), ...]
    update_data = []

    for item in evaluate_items_result:
        q_key = item.get("question_key")
        t_answer = item.get("test_answer", "")
        t_ref = item.get("test_ref", "")

        # 确保必要字段存在
        if q_key:
            update_data.append((t_answer, t_ref, test_mark, update_datetime, q_key))

    if not update_data:
        print("解析后无有效更新数据。")
        return

    print(f"准备更新数据库，共 {len(update_data)} 条数据...")

    # 3. 调用 mysql_update 方法执行批量更新
    status, msg = mysql_update(
        update_sql=update_sql,
        update_data=update_data,
        mysql_config=MYSQL_DEV_CONFIG
    )

    # 4. 打印结果
    if status:
        print("-" * 30)
        print("【数据库更新成功】")
        print(f"成功更新记录数: {len(update_data)}")
        print("-" * 30)
    else:
        print("-" * 30)
        print("【数据库更新失败】")
        print(f"错误信息: {msg}")
        print("-" * 30)

if __name__ == "__main__":
    table_name = "casebase_qa_evaluation_deepseek"
    test_mark = "deepseek-v3.2"

    sql = (f"SELECT question_id, question, quote_num FROM {table_name} "
           "where final_check = '通过'")
    evaluate_items = get_specific_evaluation_data(sql)
    print(f"共涉及数量 {len(evaluate_items)} 条")
    # 打印前3条数据验证
    for item in evaluate_items[:3]:
        print(item)
    print(f"\n-- 任务10秒后开始 -------------------------------------\n")

    time.sleep(10)

    batch_size = 3
    # 执行异步批量请求
    evaluate_items_result = run_async_evaluation(evaluate_items, batch_size)

    # 打印结果验证
    print("\n" + "=" * 60)
    print("处理结果示例:")
    print("=" * 60)
    for item in evaluate_items_result[:3]:
        print(f"\nquestion_key: {item['question_key']}")
        print(f"question: {item['question']}")
        print(f"test_answer: {item.get('test_answer', 'N/A')[:30]}...")
        print(f"test_ref: {item.get('test_ref', 'N/A')[:30]}...")

    save_results_to_db(evaluate_items_result, table_name, test_mark)