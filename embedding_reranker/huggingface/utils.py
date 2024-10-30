# huggingface/utils.py
from typing import Optional
import requests

DEFAULT_HUGGINGFACE_EMBEDDING_MODEL = "BAAI/bge-small-en"
DEFAULT_QUERY_INSTRUCTION = "Represent this question for retrieving relevant documents:"
DEFAULT_EMBED_INSTRUCTION = "Represent the document for retrieval:"
BGE_MODELS = {"BAAI/bge-small-en", "BAAI/bge-base-en", "BAAI/bge-large-en"}

def get_query_instruct_for_model_name(model_name: Optional[str]) -> str:
    """Return the query instruction for the given model."""
    if model_name in BGE_MODELS:
        return DEFAULT_QUERY_INSTRUCTION
    return ""

def format_query(query: str, model_name: Optional[str], instruction: Optional[str] = None) -> str:
    """Format the query with the appropriate instruction."""
    instruction = instruction or get_query_instruct_for_model_name(model_name)
    return f"{instruction} {query}".strip()

def get_text_instruct_for_model_name(model_name: Optional[str]) -> str:
    """Return the text instruction for the given model."""
    return DEFAULT_EMBED_INSTRUCTION if model_name in BGE_MODELS else ""

def format_text(text: str, model_name: Optional[str], instruction: Optional[str] = None) -> str:
    """Format the text with the appropriate instruction."""
    instruction = instruction or get_text_instruct_for_model_name(model_name)
    return f"{instruction} {text}".strip()
