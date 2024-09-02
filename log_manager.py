import sys
import redis
import json
from loguru import logger
from asgi_correlation_id import correlation_id
from itertools import tee
from config_manager import ConfigLoader

# Redis 설정 로드
config_loader = ConfigLoader()
redis_config = config_loader.get_config().redis

# Redis 클라이언트 설정
redis_client = redis.Redis(
    host=redis_config.host,
    port=redis_config.port,
    db=redis_config.db,
    password=redis_config.password
)

def configure_logging(logging_config):
    """
    로깅 설정을 초기화하고 구성합니다.

    Args:
        logging_config (Box): 로깅 설정을 포함하는 Box 객체
    """
    # Remove default logger
    logger.remove()

    # Define a filter function to include correlation_id in logs
    def correlation_id_filter(record):
        record["correlation_id"] = correlation_id.get() or "N/A"
        return record

    # Add file logger
    logger.add(
        logging_config.file,
        format=logging_config.format,
        level=logging_config.level,
        rotation="1 MB",
        compression="zip",
        filter=correlation_id_filter,
    )

    # Add console logger
    logger.add(
        sys.stdout,
        format=logging_config.format,
        level=logging_config.level,
        filter=correlation_id_filter,
    )

def log_request_and_store(chat_id, query_text, response, source_nodes):
    """
    요청과 응답을 로깅하고 Redis에 대화 기록을 저장합니다.

    Args:
        chat_id (str): 대화 ID
        query_text (str): 사용자의 쿼리 문자열
        response: 생성된 응답
        source_nodes: 응답을 생성하는 데 사용된 소스 노드들
    
    Returns:
        response: 로그 기록 후 반환된 응답
        chat_history: 저장된 대화 히스토리
    """
    request_id = correlation_id.get()

    # 요청 로그
    logger.info(f"[{request_id}] Received query: {query_text}")

    # 응답 로그
    response, response_copy = tee(response)  # 스트림 복제
    response_text = ""  # 전체 응답을 저장할 변수

    for token in response_copy:
        # 만약 token이 문자열이라면, 그대로 사용
        if isinstance(token, dict):
            text_part = token.get("text", "")
        else:
            text_part = token
        logger.info(f"[{request_id}] Response part: {text_part}")
        response_text += text_part

    # Redis에 대화 기록 저장
    previous_history = redis_client.get(f"chat:{chat_id}")
    if previous_history:
        chat_history = json.loads(previous_history)
    else:
        chat_history = []

    chat_history.append({"role": "user", "content": query_text})
    chat_history.append({"role": "assistant", "content": response_text})

    redis_client.set(f"chat:{chat_id}", json.dumps(chat_history))
    logger.info(f"[{request_id}] Response stored in Redis with chat_id: {chat_id}")

    return response, chat_history

def get_chat_history(chat_id):
    """
    Redis에서 대화 히스토리를 가져옵니다.

    Args:
        chat_id (str): 대화 ID
    
    Returns:
        list: 대화 히스토리 리스트 (없으면 빈 리스트 반환)
    """
    previous_history = redis_client.get(f"chat:{chat_id}")
    if previous_history:
        return json.loads(previous_history)
    return []
