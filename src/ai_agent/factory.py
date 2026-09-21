import typing

from ai_agent.agent import AgentRunner
from ai_agent.catalog.config import catalog_config
from ai_agent.catalog.repository import ProductsRepository
from ai_agent.llm import client as llm_client
from ai_agent.llm.config import llm_config
from ai_agent.llm.service import LLMService
from ai_agent.memory import repository as memory_repository
from ai_agent.memory.config import memory_config
from ai_agent.rag.retrieval import context, retrieval
from ai_agent.rag.retrieval.config import rag_retrieval_config
from ai_agent.tools import registry, search_knowledge_base, search_products


def create_agent() -> AgentRunner:
    """Build the configured agent shared by CLI and future HTTP entry points."""
    if not catalog_config.database_path.exists():
        raise RuntimeError("Каталог не подготовлен. Выполните `just catalog_restore`.")

    if not retrieval.is_vector_store_ready(rag_retrieval_config):
        raise RuntimeError("Vector store не подготовлен. Выполните `just rag_rebuild`.")

    memory: typing.Final = memory_repository.MemoryRepository(
        database_path=memory_config.database_path,
    )
    return AgentRunner(
        memory=memory,
        llm=LLMService(chat_client=llm_client.LLMClient(config=llm_config)),
        tool_registry=registry.ToolRegistry(
            tools=(
                search_knowledge_base.KnowledgeBaseSearchTool(
                    retrieval_client=retrieval.RetrievalClient(rag_retrieval_config),
                    context_builder=context.ContextBuilder(
                        min_score=rag_retrieval_config.min_score,
                        max_context_tokens=rag_retrieval_config.max_context_tokens,
                        tokenizer_model=rag_retrieval_config.context_tokenizer_model,
                    ),
                ),
                search_products.ProductSearchTool(
                    catalog=ProductsRepository(database_path=catalog_config.database_path),
                ),
            )
        ),
        memory_limit=memory_config.retrieval_limit,
    )
