import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from query_engine import QueryEngineSetup, get_response, get_condensed_question
from log_manager import configure_logging, log_and_store_conversation, get_chat_history
from llama_index.core.schema import QueryBundle, MetadataMode
from llama_index.core.llms import ChatMessage
from asgi_correlation_id import CorrelationIdMiddleware
from asgi_correlation_id import correlation_id
from loguru import logger
from contextlib import asynccontextmanager
from more_itertools import peekable
from typing import List
import re
import json

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

    collections = config_loader.get_config().engine.collections
    app.state.query_engines = [QueryEngineSetup(collection) for collection in collections]

    yield

    logger.info("Shutting down the application...")

app = FastAPI(lifespan=lifespan)
# Correlation ID 미들웨어 추가
app.add_middleware(CorrelationIdMiddleware)

class Chat(BaseModel):
    """
    대화 요청을 나타내는 모델입니다.
    
    Attributes:
        category (int): 0 ~ 17까지의 카테고리 넘버(카테고리 순서는 이름순).
        parent_id (str, optional): 이전 대화의 ID. 없을 경우 새로운 대화로 간주됩니다.
        query (str): 사용자의 쿼리 문자열.
    """
    category: int
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
    category = chat.category
    query_text = chat.query
    parent_id = chat.parent_id
    chat_id = correlation_id.get()

    # request.app.state에서 query_engine_setup을 가져옵니다.
    query_engine_setup = request.app.state.query_engines[category]

    # 이전 대화 히스토리 가져오기
    chat_history = get_chat_history(parent_id, 3)
    if chat_history:
        logger.info(f"Found previous history")
        # 간결한 질문 생성
        condensed_question = get_condensed_question(chat_history, query_text, query_engine_setup)
        logger.info(f"Condensed Question: {condensed_question}")
    else:
        logger.info(f"Starting new conversation")
        condensed_question = query_text  # 새로운 대화일 경우 현재 쿼리를 그대로 사용
    
    # 응답 생성 및 로그 저장을 한 번에 처리
    response, source_nodes = get_response(condensed_question, query_engine_setup, chat_history)
    #print(response)

    
    
    
    def response_generator(response, source_nodes):
        metadata = [{
            "source": node.text.split("\n")[0],
            "score": node.score,
            "filename": node.metadata["file_name"].split(".pdf")[0],
            "page_number": node.metadata["page_number"]
        } for node in source_nodes]
        gen = peekable(response)
        for token in gen:
            if gen.peek(None) is None:
                log_and_store_conversation(condensed_question, result, parent_id)
                print(metadata)
            result = str(token).split("assistant:")[-1].strip()
            #result = str(token)
            # yield json.dumps({"text": result, "metadata": metadata}).encode() + "\\u0000".encode() + "\n".encode()
            # res = json.dumps({"metadata": metadata, "text": result, "id": chat_id})
            res = json.dumps({"metadata": metadata, "text": result})
            yield f"data: {res}\n\n".encode()
    
    return StreamingResponse(response_generator(response, source_nodes))


if __name__ == "__main__":
    import uvicorn

    # Uvicorn 서버 실행
    uvicorn.run("app:app", host='0.0.0.0', port=2222, reload=True)