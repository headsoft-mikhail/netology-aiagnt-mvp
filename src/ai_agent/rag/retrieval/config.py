import typing
from pathlib import Path

import pydantic
import pydantic_settings


class RetrievalConfig(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(
        env_prefix="RAG_RETRIEVAL_",
    )

    embedding_model: str = "intfloat/multilingual-e5-base"
    vector_store_path: Path = Path("src/ai_agent/rag/knowledge_base/vector_store")
    collection_name: str = "store_knowledge"
    search_top_k: int = pydantic.Field(default=5, gt=0)
    min_score: float = pydantic.Field(default=0.80, ge=0, le=1)
    excluded_documents: list[str] = pydantic.Field(default_factory=list)
    max_context_tokens: int = pydantic.Field(default=1800, gt=0)
    context_tokenizer_model: str = "text-embedding-3-small"


rag_retrieval_config: typing.Final = RetrievalConfig()
