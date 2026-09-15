import argparse
import typing

from ai_agent.memory import repository as memory_repository
from ai_agent.memory.config import memory_config


def main() -> None:
    parser: typing.Final = argparse.ArgumentParser(description="Управление локальной памятью агента")
    subparsers: typing.Final = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("clear", help="Удалить долговременную память")

    args: typing.Final = parser.parse_args()

    if args.command == "clear":
        memory_repository.MemoryRepository(database_path=memory_config.database_path).clear()
        memory_config.database_path.unlink(missing_ok=True)
        print("Долговременная память очищена.")
    else:
        raise ValueError(f"Unknown command: {args.command}")
