import os
import qdrant_client
from box import Box
from transformers import AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.vllm import VllmServer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core import (
    PromptTemplate,
    get_response_synthesizer,
    Settings,
    VectorStoreIndex,
)
from config_manager import ConfigLoader
from loguru import logger

# 환경 변수 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["HF_HOME"] = "/workspace/backup/model/"

class QueryEngineSetup:
    """
    쿼리 엔진을 초기화하고 설정하는 클래스입니다.

    이 클래스는 구성 파일을 로드하여 쿼리 엔진의 각 구성 요소를 초기화하고,
    쿼리 엔진 객체를 생성하는 역할을 합니다.
    """

    def __init__(self):
        # Load config using ConfigLoader
        config_loader = ConfigLoader()
        self.config = config_loader.get_config()
        
        self.model = self.config.model
        self.logging = self.config.logging
        self.qdrant = self.config.qdrant
        self.engine = self.config.engine

        # Initialize components
        self._initialize_embedding_model()
        self._initialize_llm()
        self._initialize_vector_store()
        self._initialize_query_engine()

    def _initialize_embedding_model(self):
        """HuggingFace 임베딩 모델을 초기화합니다."""
        self.embed_model = HuggingFaceEmbedding(model_name=self.model.embed.model_name)
        Settings.embed_model = self.embed_model

    def _initialize_llm(self):
        """LLM 서버와 토크나이저를 초기화합니다."""
        self.llm = VllmServer(
            api_url=self.model.llm.api,
            max_new_tokens=self.model.llm.max_new_tokens,
            temperature=self.model.llm.temperature
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.llm.dir)
        Settings.llm = self.llm

    def _initialize_vector_store(self):
        """Qdrant 벡터 스토어를 초기화합니다."""
        self.client = qdrant_client.QdrantClient(url=self.qdrant.host, port=self.qdrant.port)
        self.vector_store = QdrantVectorStore(client=self.client, collection_name=self.engine.collection)
        self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)

    def _initialize_query_engine(self):
        """쿼리 엔진과 관련된 컴포넌트들을 초기화하고 설정합니다."""
        self.retriever = self.index.as_retriever(similarity_top_k=self.engine.sim_top_k)
        self.rerank = FlagEmbeddingReranker(model=self.model.rerank.model_name, top_n=self.model.rerank.sim_top_k)

        qa_prompt_tmpl = PromptTemplate(self.engine.template.context)

        self.synth = get_response_synthesizer(
            text_qa_template=qa_prompt_tmpl,
            streaming=True
        )

        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=self.retriever,
            llm=self.llm,
            temperature=0,
            response_synthesizer=self.synth,
            node_postprocessors=[self.rerank],
        )

    def get_query_engine(self):
        """
        초기화된 쿼리 엔진 객체를 반환합니다.

        Returns:
            RetrieverQueryEngine: 초기화된 쿼리 엔진 객체
        """
        return self.query_engine

def get_response(query_text, engine_setup, chat_history):
    """
    쿼리를 처리하고 응답을 생성합니다.

    Args:
        query_text (str): 사용자의 쿼리 문자열
        engine_setup (QueryEngineSetup): 쿼리 엔진 설정 객체
        chat_history (list): 대화 히스토리
    
    Returns:
        response: 생성된 응답
        source_nodes: 응답을 생성하는 데 사용된 소스 노드들
    """
    query_engine = engine_setup.get_query_engine()
    
    # 멀티턴 대화로 처리
    if chat_history:
        prompt = "\n".join([f"{entry['role']}: {entry['content']}" for entry in chat_history])
        prompt += f"\nuser: {query_text}\nassistant:"
    else:
        prompt = query_text
    
    answer = query_engine.query(prompt)
    
    if not answer.source_nodes:
        logger.warning("This is an empty response")
        response = engine_setup.llm.stream_complete(engine_setup.engine.template.no_context.format(prompt=query_text))
    else:
        response = answer.response_gen

    return response, answer.source_nodes
