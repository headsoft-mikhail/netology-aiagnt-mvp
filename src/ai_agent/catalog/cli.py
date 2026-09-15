import argparse
import typing

from ai_agent.catalog.config import catalog_config
from ai_agent.catalog.repository import ProductsRepository


def main() -> None:
    parser: typing.Final = argparse.ArgumentParser(description="Управление локальным каталогом товаров")
    subparsers: typing.Final = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("restore", help="Восстановить SQLite-каталог из CSV-слепка")
    subparsers.add_parser("clear", help="Удалить сгенерированный SQLite-каталог")

    args: typing.Final = parser.parse_args()
    catalog: typing.Final = ProductsRepository(database_path=catalog_config.database_path)

    if args.command == "restore":
        restored_count: typing.Final = catalog.restore_from_snapshot(catalog_config.snapshot_path)
        print(f"Каталог восстановлен из CSV: {restored_count} товаров.")
    elif args.command == "clear":
        catalog_config.database_path.unlink(missing_ok=True)
        print("SQLite-каталог удалён.")
    else:
        raise ValueError(f"Unknown command: {args.command}")
