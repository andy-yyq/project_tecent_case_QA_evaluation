import re, sys
from datetime import datetime
import pymysql
import string
import random
from settings import MYSQL_DEV_CONFIG, QDRANT_CONFIG
import asyncio
import aiohttp
import copy
import time
from typing import List, Dict
from loguru import logger
from loguru._logger import Logger
from pathlib import Path
import settings
from qdrant_client import QdrantClient, models
from zhipuai import ZhipuAI
import os
from dotenv import load_dotenv

load_dotenv()

def log_error(**kwargs):
    for key, value in kwargs.items():
        logger.error(f"{key}, {value}")

qdrant_refs_conn = QdrantClient(**QDRANT_CONFIG)

def Qdrant_search(qdrant_conn, collection_name, embedding, query_filter=None, with_payload=None, limit=None):
    return qdrant_conn.search(
        collection_name=collection_name,
        query_filter=query_filter,
        with_payload=with_payload,
        query_vector=embedding,
        limit=limit
    )

def embedding_zhipu_api(content, embedding_model="embedding-3"):
    engine = ZhipuAI(api_key=os.getenv("LLM_ZHIPU_API_KEY"))
    embedding_vector = None
    try:
        embedding_vector = engine.embeddings.create(model=embedding_model, dimensions = 1024,
                                                    input=content).data[0].embedding
    except Exception as e:
        log_error(msg=f"【data_embedding】智谱embedding接口报错(第1次):{e}，向量化内容：{content}")
        time.sleep(1)
        try:
            embedding_vector = engine.embeddings.create(model=embedding_model, dimensions = 1024,
                                                        input=content).data[0].embedding
        except Exception as e:
            log_error(msg=f"【data_embedding】智谱embedding接口报错(第2次):{e}，向量化内容：{content}")

    return embedding_vector


def setup_logging(logger_instance: Logger = logger) -> Path:
    """创建 log_yyyy-mm-dd_hh-mm-ss.log 并绑定 loguru。"""
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_name = datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.log")
    log_path = settings.LOG_DIR / log_name

    if logger_instance is logger:
        logger_instance.remove()
        logger_instance.add(sys.stderr, level="INFO")

    logger_instance.add(log_path, encoding="utf-8", level="DEBUG")
    return log_path

class CustomError(Exception):
    def __init__(self, ErrorInfo):
        self.ErrorInfo = ErrorInfo

    def __str__(self):
        return self.ErrorInfo

def mysql_query(**kwargs):
    '''
    通过指定sql查询语句（及动态参数，可以没有）查询结果
    :param sql_query: 查询语句
    :param kwarg: 动态参数
    :return:
    '''
    assert kwargs.get("sql_query"), f"入参缺失：sql_query "
    assert kwargs.get("mysql_config"), f"入参缺失：mysql_config "
    sql_query = kwargs["sql_query"]
    mysql_config = kwargs["mysql_config"]

    result = list()
    pattern = re.compile(r"\{.*?\}")
    while True:
        if pattern.search(sql_query):
            sign = pattern.search(sql_query).group()
            param = sign[1:-1]
            if kwargs.get(param):
                sql_query = re.sub(sign, kwargs[param], sql_query)
            else:
                raise CustomError(f"Mysql查询参数缺失或错误:{param}")
        else:
            break

    try:
        conn = pymysql.connect(**mysql_config)
        cursor = conn.cursor()
        cursor.execute(sql_query)

        size = 500
        while True:
            rows = cursor.fetchmany(size)
            if not rows:
                break
            result.extend(rows)
        cursor.close()
        conn.close()
    except Exception as e:
        msg = f"Error, mysql数据库查询失败a, {e}"
        return False, None, msg
    else:
        return True, result, ""

def mysql_insert(**kwargs):
    assert kwargs.get("insert_sql"), f"入参缺失：insert_sql"
    assert kwargs.get("insert_data"), f"入参缺失：insert_data "
    assert kwargs.get("mysql_config"), f"入参缺失：mysql_config "
    insert_sql = kwargs["insert_sql"]
    insert_data = kwargs["insert_data"]
    mysql_config = kwargs["mysql_config"]

    try:
        conn = pymysql.connect(**mysql_config)
        with conn.cursor() as cursor:
            cursor.executemany(insert_sql, insert_data)
            conn.commit()
        conn.close()
    except Exception as e:
        msg = f"Error, mysql数据库写入失败 {e}"
        return False, msg

    return True, None

def mysql_update(**kwargs):
    assert kwargs.get("update_sql"), "入参缺失：update_sql"
    assert kwargs.get("update_data"), "入参缺失：update_data"
    assert kwargs.get("mysql_config"), f"入参缺失：mysql_config "
    update_sql = kwargs["update_sql"]
    update_data = kwargs["update_data"]
    mysql_config = kwargs["mysql_config"]

    try:
        conn = pymysql.connect(**mysql_config)
        with conn.cursor() as cursor:
            cursor.executemany(update_sql, update_data)
            conn.commit()
        conn.close()
    except Exception as e:
        msg = f"Error, mysql数据库更新失败 {e}"
        return False, msg

    return True, None

def short_id(length=8):  # 增加总长度
    # 使用微秒级时间戳 + 更长的随机串
    num = length//2
    timestamp = str(int(time.time() * 1000))[-num:]  # 取最后6位
    rand_str = ''.join(random.choices(string.ascii_letters + string.digits, k=length-num))
    return timestamp + rand_str

def get_specific_evaluation_data(sql):
    """
    获取特定查询的casebase_qa_evaluation表数据
    """
    status, result, msg = mysql_query(
        sql_query=sql,
        mysql_config=MYSQL_DEV_CONFIG
    )

    if not status:
        print(f"数据库查询失败: {msg}")
        return []

    evaluate_items = list()
    for row in result:
        # row 结构: (question_key, question, quote_num)
        item = {
            "question_key": row[0],
            "question": row[1],
            "quote_num": row[2]
        }
        evaluate_items.append(item)

    return evaluate_items


async def send_single_request(session: aiohttp.ClientSession, item: Dict, url: str) -> Dict:
    """
    发送单个异步请求
    :param session: aiohttp会话
    :param item: evaluate_items中的单条数据
    :param url: 请求地址
    :return: 包含请求结果的数据项
    """
    # 复制原始数据，避免修改原数据
    result_item = copy.deepcopy(item)

    # 构造请求数据
    data = {
        "user_id": "yyq",
        "session_id": "LHL_001",
        "history_dialogs": [],
        "msg_followUp": item["question"],
        "search_method": "localsearch",
        "websearch_mode": 0,
        "stream_mode": 0,
        "scenario_mode": 1
    }

    try:
        async with session.post(url, json=data, timeout=aiohttp.ClientTimeout(total=60)) as response:
            if response.status == 200:
                json_result = await response.json()

                # 检查返回结构
                if json_result.get("code") == 200 and json_result.get("data"):
                    data_obj = json_result["data"]
                    result_item["test_answer"] = data_obj.get("response", "")
                    result_item["test_ref"] = data_obj.get("data_retrieved_list", "")
                else:
                    print(
                        f"[请求失败] question_key: {item['question_key']}, 错误: {json_result.get('message', '未知错误')}")
                    result_item["test_answer"] = ""
                    result_item["test_ref"] = ""
            else:
                print(f"[HTTP错误] question_key: {item['question_key']}, 状态码: {response.status}")
                result_item["test_answer"] = ""
                result_item["test_ref"] = ""

    except asyncio.TimeoutError:
        print(f"[超时] question_key: {item['question_key']}")
        result_item["test_answer"] = ""
        result_item["test_ref"] = ""
    except Exception as e:
        print(f"[异常] question_key: {item['question_key']}, 错误: {str(e)}")
        result_item["test_answer"] = ""
        result_item["test_ref"] = ""

    return result_item


async def process_batch(session: aiohttp.ClientSession, batch_items: List[Dict], url: str, batch_num: int) -> List[
    Dict]:
    """
    处理单个批次的请求
    :param session: aiohttp会话
    :param batch_items: 当前批次的数据列表
    :param url: 请求地址
    :param batch_num: 批次号
    :return: 当前批次的处理结果列表
    """
    print(f"\n========== 开始处理第 {batch_num} 批次，共 {len(batch_items)} 条 ==========")
    start_time = time.time()

    # 创建当前批次的所有异步任务
    tasks = [send_single_request(session, item, url) for item in batch_items]

    # 并发执行所有任务
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time
    print(f"第 {batch_num} 批次完成，耗时: {elapsed:.2f} 秒")

    return list(results)


async def async_batch_request(evaluate_items: List[Dict], batch_size: int = 5) -> List[Dict]:
    """
    异步批量请求主函数
    :param evaluate_items: 待处理的数据列表
    :param batch_size: 每批次并发数量，默认5
    :return: 包含请求结果的数据列表
    """
    # url = "http://47.116.178.225:8507/search"
    url = "http://localhost:8000/search"

    # 复制原始列表到新变量
    evaluate_items_result = []

    total_count = len(evaluate_items)
    print(f"总数据量: {total_count}，每批次: {batch_size}，预计批次: {(total_count + batch_size - 1) // batch_size}")

    # 创建aiohttp会话
    connector = aiohttp.TCPConnector(limit=batch_size, force_close=True)

    async with aiohttp.ClientSession(connector=connector) as session:
        batch_num = 0

        # 分批处理
        for i in range(0, total_count, batch_size):
            batch_num += 1
            batch_items = evaluate_items[i:i + batch_size]

            # 处理当前批次
            batch_results = await process_batch(session, batch_items, url, batch_num)

            # 将结果追加到结果列表
            evaluate_items_result.extend(batch_results)

            # 可选：批次间短暂休息，避免请求过快
            if i + batch_size < total_count:
                await asyncio.sleep(0.5)

    print(f"\n所有批次处理完成，共处理 {len(evaluate_items_result)} 条数据")
    return evaluate_items_result


def run_async_evaluation(evaluate_items: List[Dict], batch_size: int = 5) -> List[Dict]:
    """
    同步入口函数，用于启动异步请求
    :param evaluate_items: 待处理的数据列表
    :param batch_size: 每批次并发数量
    :return: 包含请求结果的数据列表
    """
    return asyncio.run(async_batch_request(evaluate_items, batch_size))

def get_current_time():
    # 获取当前时间
    current_time = datetime.now()
    # 格式化时间
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_time