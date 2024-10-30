# embedding.py
from typing import List, Union
from fastapi import FastAPI
import torch
import typer
import uvicorn
from pydantic import BaseModel
from typing_extensions import Annotated
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from FlagEmbedding import BGEM3FlagModel
from FlagEmbedding.flag_reranker import FlagReranker
from logger import config_logger

app = FastAPI(debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

config_logger('embedding')

class Node:
    """
    A class representing a model node for either embedding or reranking tasks.

    Args:
        idx (int): The index of the node for identification.
        model_path (str): The path to the model.
        device (str): The GPU device to use (e.g., 'cuda:0').
        model_type (str, optional): The type of the model, 'embedding' or 'reranker'. Defaults to 'embedding'.
    """

    def __init__(self, idx: int, model_path: str, device: str, model_type: str = "embedding"):
        self.idx = idx
        self.len_queue = 0
        device_str = f"cuda:{device}"
        assert model_type in ["embedding", "reranker"], "Invalid model type specified."
        self.model_type = model_type
        if model_type == "embedding":
            self.model = BGEM3FlagModel(model_path, device=device_str)
        else:
            self.model = FlagReranker(model_path, use_fp16=False, device=device_str)

    def _embedding(self, query: list) -> torch.Tensor:
        """Generate embeddings for each query in the list using the embedding model."""
        assert isinstance(self.model, BGEM3FlagModel), "Embedding called on a reranker node."
        return self.model.encode(query)['dense_vecs']

    def _rerank(self, data: list) -> list:
        """Rerank the list of queries using the reranker model."""
        return self.model.compute_score(data)

    def __call__(self, query: list) -> torch.Tensor | list:
        """Process the query using the appropriate model based on the node type."""
        self.len_queue += 1
        try:
            if self.model_type == 'embedding':
                result = self._embedding(query)
            else:
                result = self._rerank(query)
        finally:
            self.len_queue -= 1
        return result

class ModelCluster:
    """
    A class for managing a cluster of model nodes for distributed inference.

    Args:
        model_path (str): The path to the model.
        devices (List[str]): List of GPU devices for the models.
        model_type (str): The type of model, either 'embedding' or 'reranker'.
    """

    def __init__(self, model_path: str, devices: List[str], model_type: str):
        assert model_type in ['embedding', 'reranker'], "Model type must be 'embedding' or 'reranker'."
        self.nodes = [Node(idx, model_path, device, model_type) for idx, device in enumerate(devices)]

    def __call__(self, query: list) -> torch.Tensor | list:
        """
        Process the query by finding the least loaded node and using it for inference.

        Args:
            query (List): The input query to process.

        Returns:
            torch.Tensor | List: The result from the model, either embeddings or reranking scores.
        """
        # Sort nodes by queue length (ascending)
        order = sorted(self.nodes, key=lambda node: node.len_queue)
        for node in order:
            try:
                return node(query)
            except RuntimeError as e:
                logger.error(f"Request failed to node {node.idx}. Trying next node. Error: {e}")
                continue
        raise RuntimeError("All nodes failed to process the request.")

def main(
    embedding_model_path: Annotated[str, typer.Option(help="Path to embedding model")] = "/workspace/backup/local_model/bge-m3",
    reranker_model_path: Annotated[str, typer.Option(help="Path to reranker model")] = "/workspace/backup/local_model/bge-reranker-v2-m3",
    port: Annotated[int, typer.Option(help="Port for API")] = 2202,
    embedding_device: Annotated[str, typer.Option(help="Devices for embedding model (comma-separated)")] = "0",
    reranker_device: Annotated[str, typer.Option(help="Devices for reranker model (comma-separated)")] = "0",
):

    # Initialize model clusters
    embedding_nodes = ModelCluster(embedding_model_path, embedding_device.split(','), "embedding")
    reranker_nodes = ModelCluster(reranker_model_path, reranker_device.split(','), "reranker")

    def make_status_json():
        """Generate JSON representing the load status of all nodes."""
        embedding_load = [{"id": f"Embedding{node.idx}", "count": node.len_queue} for node in embedding_nodes.nodes]
        reranker_load = [{"id": f"Reranker{node.idx}", "count": node.len_queue} for node in reranker_nodes.nodes]
        return {
            "nodes": embedding_load + reranker_load
        }

    class Payload(BaseModel):
        """Schema for API requests."""
        query: list
        origin: str = None

    @app.post("/embedding")
    def encode(query: Payload):
        """Endpoint to get embeddings for a given query."""
        origin = query.origin if query.origin else None
        logger.info(f"Embedding request received from {origin}")
        generated = embedding_nodes(query.query)
        res = [i.tolist() for i in generated]
        return res

    @app.post("/rerank")
    def rerank(query: Payload):
        """Endpoint to get reranking scores for a given query."""
        origin = query.origin if query.origin else "Unknown"
        logger.info(f"Rerank request received from {origin}")
        return reranker_nodes(query.query)

    @app.get("/status")
    async def get_load_status():
        """Endpoint to get the current load status of the server."""
        return make_status_json()

    uvicorn.run(app, host='0.0.0.0', port=int(port))

if __name__ == '__main__':
    typer.run(main)
