from typing import Any, List, Optional
import requests
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle

dispatcher = get_dispatcher(__name__)


class FlagEmbeddingReranker(BaseNodePostprocessor):
    """Flag Embedding Reranker."""

    model: str = Field(description="BAAI Reranker model name or URL.")
    top_n: int = Field(description="Number of nodes to return sorted by score.")

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_model(self.model, True)

    def _initialize_model(self, model: str, use_fp16: bool) -> None:
        """Initialize the Flag Embedding model."""
        if not model.startswith("http"):
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(model, use_fp16=use_fp16)
        else:
            self._model = model

    def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None) -> List[NodeWithScore]:
        """Re-rank the given nodes based on the query."""
        if query_bundle is None:
            raise ValueError("Query bundle is required.")
        if not nodes:
            return []

        # Compute scores and re-rank nodes
        scores = self._compute_scores(nodes, query_bundle.query_str)
        for node, score in zip(nodes, scores):
            node.score = score
        return sorted(nodes, key=lambda x: -x.score)[:self.top_n]

    def _compute_scores(self, nodes: List[NodeWithScore], query: str) -> List[float]:
        """Compute scores for the given nodes."""
        query_and_nodes = [(query, node.node.get_content()) for node in nodes]
        if self._model.startswith("http"):
            response = requests.post(self._model, json={"query": query_and_nodes})
            response.raise_for_status()  # Raise an error for bad HTTP responses
            return response.json()
        return self._model.compute_score(query_and_nodes)