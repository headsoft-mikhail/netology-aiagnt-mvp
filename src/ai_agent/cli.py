import argparse
import logging as standard_logging
import sys
import typing
import uuid

from ai_agent import contracts, logging
from ai_agent.agent import AgentRunner
from ai_agent.factory import create_agent

EXIT_SUCCESS: typing.Final = 0
EXIT_FAILURE: typing.Final = 1
LOGGER_OBJ: typing.Final = standard_logging.getLogger(__name__)


def main() -> int:
    parser: typing.Final = argparse.ArgumentParser(description="MVP консультанта по сетевому оборудованию")
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
    parser.add_argument(
        "--log-level",
        choices=logging.LOG_LEVEL_NAMES,
        default="info",
        help="Уровень логирования (по умолчанию: info)",
    )
    args: typing.Final = parser.parse_args()

    logging.configure(level=logging.level_from_name(args.log_level))

    try:
        return start_agent(
            args.user_id,
            args.session_id,
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
    show_trace: bool = False,
) -> int:
    print("Инициализация консультанта по сетевому оборудованию...")
    agent: typing.Final = create_agent()
    actual_session_id: typing.Final = session_id or str(uuid.uuid4())

    return _run_interactive(agent, user_id, actual_session_id, show_trace=show_trace)


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
