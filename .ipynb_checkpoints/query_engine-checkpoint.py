import os
import qdrant_client
from box import Box
from transformers import AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.vllm import VllmServer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import QueryBundle, MetadataMode
from llama_index.core.postprocessor import SimilarityPostprocessor
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.llm.dir)

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
        Settings.llm = self.llm

    def _initialize_vector_store(self):
        """Qdrant 벡터 스토어를 초기화합니다."""
        self.client = qdrant_client.QdrantClient(url=self.qdrant.host, port=self.qdrant.port)
        self.vector_store = QdrantVectorStore(client=self.client, collection_name=self.engine.collection)
        self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)

    def _initialize_query_engine(self):
        """쿼리 엔진과 관련된 컴포넌트들을 초기화하고 설정합니다."""
        self.retriever = self.index.as_retriever(similarity_top_k=self.engine.sim_top_k)
        reranker = FlagEmbeddingReranker(model=self.model.rerank.model_name, top_n=self.model.rerank.sim_top_k)
        sim_processor = SimilarityPostprocessor(similarity_cutoff=self.engine.sim_cutoff)
        postprocessors = [sim_processor, reranker]

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
            node_postprocessors=postprocessors,
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
    retrieved_nodes = query_engine.retriever.retrieve(query_text)
    for postprocessor in query_engine._node_postprocessors:
        retrieved_nodes = postprocessor.postprocess_nodes(
            retrieved_nodes, query_bundle=QueryBundle(query_text)
        )
    context_str = "\n\n".join(
        [n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in retrieved_nodes]
    )
    
    if context_str:
        query_template = {"role": "user", "content": engine_setup.engine.template.context.format(context_str=context_str, query_str=query_text)}
    else:
        query_template = {"role": "user", "content": engine_setup.engine.template.no_context.format(query_str=query_text)}
    # 멀티턴 대화로 처리 
    if chat_history:
        chat_history.append(query_template)
        prompt = chat_history
    else:
        prompt = [query_template]
    prompt = engine_setup.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    
    response = engine_setup.llm.stream_complete(prompt)

    return response, retrieved_nodes

def get_condensed_question(chat_history, current_query, engine_setup):
    """
    사용자의 이전 대화와 현재 쿼리를 기반으로 간결한 질문을 생성합니다.
    
    Args:
        chat_history (list): 이전 대화 히스토리 (최신 대화가 마지막에 있음).
        current_query (str): 사용자의 현재 쿼리.
        engine_setup (QueryEngineSetup): 쿼리 엔진 설정 객체.
    
    Returns:
        str: 간결한 질문 문자열.
    """
    
    user_query = chat_history[-2]["content"]
    assistant_response = chat_history[-1]["content"]
    
    condense_prompt = engine_setup.config.engine.template.condense.format(
        user_query=user_query,
        assistant_response=assistant_response,
        current_query=current_query
    )
    
    # condense_prompt를 기반으로 모델에게 condensed question 요청
    condensed_question = engine_setup.llm.complete(condense_prompt).text.split("condensed_question:")[-1].strip()
    
    return condensed_question

