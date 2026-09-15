import argparse
import logging as standard_logging
import sys
import typing
import uuid

from ai_agent import contracts, logging
from ai_agent.agent import AgentRunner
from ai_agent.catalog.config import catalog_config
from ai_agent.catalog.repository import ProductsRepository
from ai_agent.llm import client as llm_client
from ai_agent.llm import config as llm_config
from ai_agent.llm.service import LLMService
from ai_agent.memory import repository as memory_repository
from ai_agent.memory.config import memory_config
from ai_agent.rag import config as rag_config
from ai_agent.rag import context, retrieval
from ai_agent.tools import registry, search_knowledge_base, search_products

EXIT_SUCCESS: typing.Final = 0
EXIT_FAILURE: typing.Final = 1
LOGGER_OBJ: typing.Final = standard_logging.getLogger(__name__)


def main() -> int:
    parser: typing.Final = argparse.ArgumentParser(description="MVP консультанта по сетевому оборудованию")
    parser.add_argument("question", nargs="?", help="Один вопрос без запуска интерактивной сессии")
    parser.add_argument(
        "--user-id",
        "--user_id",
        dest="user_id",
        default="local_user",
        help="Владелец долговременной памяти (по умолчанию: local_user)",
    )
    parser.add_argument(
        "--session-id",
        "--session_id",
        dest="session_id",
        default=None,
        help="Идентификатор текущего диалога (по умолчанию создаётся автоматически)",
    )
    parser.add_argument("--trace", action="store_true", help="Вывести структурированный trace в stderr")
    log_level: typing.Final = parser.add_mutually_exclusive_group()
    log_level.add_argument("--verbose", action="store_true", help="Показывать этапы работы агента")
    log_level.add_argument("--debug", action="store_true", help="Показывать подробную диагностику")
    args: typing.Final = parser.parse_args()

    logging.configure(verbose=args.verbose, debug=args.debug)

    try:
        return start_agent(
            args.user_id,
            args.session_id,
            question=args.question,
            show_trace=args.trace,
        )
    except Exception:
        LOGGER_OBJ.debug("Agent startup failed", exc_info=True)
        print(
            "Не удалось запустить агента. Проверьте .env и выполните `just prepare_agent`.",
            file=sys.stderr,
        )
        return EXIT_FAILURE


def start_agent(
    user_id: str,
    session_id: str | None = None,
    *,
    question: str | None = None,
    show_trace: bool = False,
) -> int:
    print("Инициализация консультанта по сетевому оборудованию...")
    agent: typing.Final = create_agent()
    actual_session_id: typing.Final = session_id or str(uuid.uuid4())

    if question is not None:
        try:
            result: typing.Final = agent.run(user_id, question, session_id=actual_session_id)
            _print_result(result, show_trace=show_trace)
            return EXIT_FAILURE if result.status is contracts.AgentRunStatus.FAILED else EXIT_SUCCESS
        finally:
            agent.end_session(actual_session_id)

    return _run_interactive(agent, user_id, actual_session_id, show_trace=show_trace)


def create_agent() -> AgentRunner:
    loaded_retrieval_config: typing.Final = rag_config.load_retrieval_config()

    if not catalog_config.database_path.exists():
        raise RuntimeError("Каталог не подготовлен. Выполните `just catalog_restore`.")

    if not loaded_retrieval_config.vector_store_path.exists():
        raise RuntimeError("Vector store не подготовлен. Выполните `just rag_rebuild`.")

    memory: typing.Final = memory_repository.MemoryRepository(
        database_path=memory_config.database_path,
    )
    return AgentRunner(
        memory=memory,
        llm=LLMService(chat_client=llm_client.LLMClient(config=llm_config.LLMConfig())),
        tool_registry=registry.ToolRegistry(
            tools=(
                search_knowledge_base.KnowledgeBaseSearchTool(
                    retrieval_client=retrieval.RetrievalClient(loaded_retrieval_config),
                    context_builder=context.ContextBuilder(min_score=loaded_retrieval_config.min_score),
                ),
                search_products.ProductSearchTool(
                    catalog=ProductsRepository(database_path=catalog_config.database_path),
                ),
            )
        ),
        memory_limit=memory_config.retrieval_limit,
    )


def _run_interactive(
    agent: AgentRunner,
    user_id: str,
    session_id: str,
    *,
    show_trace: bool,
) -> int:
    print("Каталог и vector store готовы.")
    print("\n" + "=" * 60)
    print(f" КОНСУЛЬТАНТ МАГАЗИНА (Пользователь: {user_id})")
    print("=" * 60)
    print("Агент запущен и готов к работе в терминале.")
    print("Введите 'exit' или 'выход' для завершения сессии.\n")

    try:
        while True:
            try:
                user_input = input("Покупатель >>> ")
                if user_input.lower() in ["exit", "выход"]:
                    print("Сессия завершена. До встречи!")
                    break

                if not user_input.strip():
                    continue

                result = agent.run(user_id, user_input, session_id=session_id)
                _print_result(result, show_trace=show_trace)
            except KeyboardInterrupt:
                print("\nСессия прервана.")
                break
    finally:
        agent.end_session(session_id)
    return EXIT_SUCCESS


def _print_result(result: contracts.AgentResponse, *, show_trace: bool) -> None:
    print(f"\nАгент: {result.answer}\n")
    if show_trace:
        print(result.model_dump_json(indent=2), file=sys.stderr)
