# seoul-ai-manual
서울시 업무 생산성 향상을 위한 AI 기반 매뉴얼 챗봇입니다.

## installation

필수 라이브러리 설치

`pip install -r requirements.txt`

flash infer 설치

`pip install flashinfer==0.1.6 -i https://flashinfer.ai/whl/cu124/torch2.4`

FlagEmbedding 설치

`pip install git+https://github.com/FlagOpen/FlagEmbedding.git`

## Quickstart

vllm 서버 실행

`run_vllm.sh`

embedding/reranker 서버 실행

`python embedding_service/embedding.py`

api 서버 실행

`python app.py`

## Sample
```
curl -X 'POST' \
  'http://localhost:2222/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "category": 0,
    "parent_id": "",
    "query": "카카오톡 설치 어떻게 해?"
}'
```
