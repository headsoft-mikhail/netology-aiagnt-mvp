import argparse
import typing

from ai_agent.agent import MockAgentRunner
from ai_agent.catalog.config import catalog_config
from ai_agent.memory import repository as memory_repository
from ai_agent.memory.config import memory_config
from ai_agent.rag import config as rag_config


def main() -> None:
    parser: typing.Final = argparse.ArgumentParser(description="MVP консультанта по сетевому оборудованию")
    parser.add_argument("--user-id", default="local_user")
    args: typing.Final = parser.parse_args()

    start_agent(args.user_id)


def start_agent(user_id: str) -> None:
    print("Инициализация консультанта по сетевому оборудованию...")
    loaded_retrieval_config: typing.Final = rag_config.load_retrieval_config()

    if not catalog_config.database_path.exists():
        raise RuntimeError("Каталог не подготовлен. Выполните `just catalog_restore`.")

    if not loaded_retrieval_config.vector_store_path.exists():
        raise RuntimeError("Vector store не подготовлен. Выполните `just rag_rebuild`.")

    print("Каталог и vector store готовы.")

    memory: typing.Final = memory_repository.MemoryRepository(
        database_path=memory_config.database_path,
    )
    agent: typing.Final = MockAgentRunner(memory)

    print("\n" + "=" * 60)
    print(f" КОНСУЛЬТАНТ МАГАЗИНА (Пользователь: {user_id})")
    print("=" * 60)
    print("Агент запущен и готов к работе в терминале.")
    print("Введите 'exit' или 'выход' для завершения сессии.\n")

    while True:
        try:
            user_input = input("Покупатель >>> ")
            if user_input.lower() in ["exit", "выход"]:
                print("Сессия завершена. До встречи!")
                break

            if not user_input.strip():
                continue

            result = agent.run(user_id, user_input)
            print(f"\nАгент: {result.answer}\n")
        except KeyboardInterrupt:
            print("\nСессия прервана.")
            break
