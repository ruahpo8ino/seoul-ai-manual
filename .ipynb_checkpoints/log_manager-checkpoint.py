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

def log_and_store_conversation(query_text, response_text, parent_id=None):
    """
    요청과 응답을 로깅하고 Redis에 개별 대화 기록을 저장합니다.

    Args:
        query_text (str): 사용자의 쿼리 문자열
        response: 생성된 응답
        source_nodes: 응답을 생성하는 데 사용된 소스 노드들
        parent_id (str, optional): 이전 쿼리의 ID (대화 흐름 유지용)
    
    Returns:
        response: 로그 기록 후 반환된 응답
    """
    request_id = correlation_id.get()
    logger.info(f"Generated Request ID: {request_id}")

    # 요청 로그
    logger.info(f"Received query: {query_text}")
    # 결과 로그
    logger.info(f"Result: {response_text}")
    # Redis에 개별 대화 기록 저장
    redis_client.hset(request_id, mapping={
        "user": query_text,
        "assistant": response_text,
        "parent_id": parent_id or ""
    })
    logger.info(f"Stored conversation in Redis: {request_id}")

def get_chat_history(chat_id, max_turns=3):
    """
    Redis에서 대화 히스토리를 최대 max_turns 수만큼 가져옵니다.
    주어진 chat_id에서 시작하여 parent_id가 없을 때까지 추적하되,
    최대 max_turns개의 대화만 가져옵니다.

    Args:
        chat_id (str): 대화 ID
        max_turns (int): 가져올 대화 기록의 최대 수 (기본값은 3)
    
    Returns:
        list: 최신 순서로 반환된 대화 히스토리 리스트 (최대 max_turns개의 기록)
    """
    history = []
    current_id = chat_id
    turn_count = 0

    while current_id and turn_count < max_turns:
        # Redis에서 현재 ID에 해당하는 대화 기록 가져오기
        chat_data = redis_client.hgetall(current_id)
        if not chat_data:
            break

        # 사용자 및 어시스턴트의 대화 기록 추가
        history.append({
            "role": "user",
            "content": chat_data.get(b"user", "").decode('utf-8')
        })
        history.append({
            "role": "assistant",
            "content": chat_data.get(b"assistant", "").decode('utf-8')
        })

        # 대화 기록을 추가할 때마다 turn_count 증가
        turn_count += 1

        # 다음 부모 ID로 이동
        current_id = chat_data.get(b"parent_id", "").decode('utf-8')

    return history