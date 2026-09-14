import argparse
import typing

from ai_agent.catalog import restore
from ai_agent.catalog.config import catalog_config


def main() -> None:
    parser: typing.Final = argparse.ArgumentParser(description="Управление локальным каталогом товаров")
    subparsers: typing.Final = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("restore", help="Восстановить SQLite-каталог из CSV-слепка")
    subparsers.add_parser("clear", help="Удалить сгенерированный SQLite-каталог")

    args: typing.Final = parser.parse_args()

    if args.command == "restore":
        restored_count: typing.Final = restore.restore_catalog(catalog_config)
        print(f"Каталог восстановлен из CSV: {restored_count} товаров.")
    elif args.command == "clear":
        catalog_config.database_path.unlink(missing_ok=True)
        print("SQLite-каталог удалён.")
    else:
        raise ValueError(f"Unknown command: {args.command}")
