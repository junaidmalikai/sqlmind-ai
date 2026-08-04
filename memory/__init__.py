"""Memory package — conversation, checkpoints, summaries, episodic, workflow, vector fabric."""

from memory.checkpointing import build_checkpointer
from memory.conversation import ConversationMemory, HistoryStore, QueryHistoryItem
from memory.embeddings import (
    EmbeddingProvider,
    HashingEmbeddingProvider,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    build_embedding_provider,
)
from memory.episodic import EpisodicMemoryStore
from memory.summarizer import ConversationSummarizer
from memory.vector_store import MemoryFabric, VectorMemoryStore, compress_text
from memory.workflow_memory import WorkflowMemoryStore

__all__ = [
    "ConversationMemory",
    "ConversationSummarizer",
    "EmbeddingProvider",
    "EpisodicMemoryStore",
    "HashingEmbeddingProvider",
    "HistoryStore",
    "MemoryFabric",
    "OllamaEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "QueryHistoryItem",
    "SentenceTransformerEmbeddingProvider",
    "VectorMemoryStore",
    "WorkflowMemoryStore",
    "build_checkpointer",
    "build_embedding_provider",
    "compress_text",
]
