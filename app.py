import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from query_engine import QueryEngineSetup, get_response
from log_manager import configure_logging, log_request_and_store, get_chat_history
from asgi_correlation_id import CorrelationIdMiddleware
from asgi_correlation_id import correlation_id
from loguru import logger
from contextlib import asynccontextmanager

# FastAPI 앱 초기화
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 애플리케이션의 수명 주기 관리.

    서버의 시작과 종료 시 수행될 작업들을 정의합니다.
    """
    from config_manager import ConfigLoader

    config_loader = ConfigLoader()
    logging_config = config_loader.get_config().logging
    configure_logging(logging_config)
    logger.info("Starting up the application...")

    query_engine_setup = QueryEngineSetup()
    app.state.query_engine_setup = query_engine_setup

    yield

    logger.info("Shutting down the application...")

app = FastAPI(lifespan=lifespan)
# Correlation ID 미들웨어 추가
app.add_middleware(CorrelationIdMiddleware)

class Chat(BaseModel):
    """
    대화 요청을 나타내는 모델입니다.
    
    Attributes:
        parent_id (str, optional): 이전 대화의 ID. 없을 경우 새로운 대화로 간주됩니다.
        query (str): 사용자의 쿼리 문자열.
    """
    parent_id: str = None
    query: str

@app.post("/chat")
async def chat_manual(chat: Chat, request: Request):
    """
    사용자의 쿼리에 응답하는 API 엔드포인트입니다.
    
    Args:
        chat (Chat): 사용자의 대화 요청 모델.
        request (Request): FastAPI 요청 객체.
    
    Returns:
        StreamingResponse: 생성된 응답을 스트리밍 방식으로 반환합니다.
    """
    query_text = chat.query
    chat_id = chat.parent_id or correlation_id.get()

    # request.app.state에서 query_engine_setup을 가져옵니다.
    query_engine_setup = request.app.state.query_engine_setup

    # 이전 대화 히스토리 가져오기
    chat_history = get_chat_history(chat_id)
    if chat_history:
        logger.info(f"[{chat_id}] Found previous history")
    else:
        logger.info(f"[{chat_id}] Starting new conversation")

    # get_response 함수 호출
    response, source_nodes = get_response(query_text, query_engine_setup, chat_history)

    # 요청과 응답을 로그로 기록하고 Redis에 저장
    response, chat_history = log_request_and_store(chat_id, query_text, response, source_nodes)

    # 실제 클라이언트로 스트리밍할 생성기 함수
    async def response_generator():
        for token in response:
            yield json.dumps({"text": token, "metadata": [node.metadata for node in source_nodes]}).encode() + "\\u0000".encode() + "\n".encode()
    
    return StreamingResponse(response_generator(), media_type="application/json")

if __name__ == "__main__":
    import uvicorn

    # Uvicorn 서버 실행
    uvicorn.run(app, host='0.0.0.0', port=2223)
