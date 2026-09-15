import pathlib
import sqlite3
import sys
import typing

import pytest

from ai_agent.catalog import cli, config, repository

EXPECTED_PRODUCT_COUNT: typing.Final = 35
EXPECTED_PRICE_RUB: typing.Final = 6490
CHANGED_PRICE_RUB: typing.Final = 1
TEST_PRODUCT_CODE: typing.Final = "RTR-TP-AX23"


def test_restore_catalog_replaces_database_from_csv_snapshot(tmp_path: pathlib.Path) -> None:
    db_path: typing.Final = tmp_path / "products.db"
    csv_path: typing.Final = pathlib.Path(repository.__file__).parent / "data" / "products.csv"

    catalog_config: typing.Final = config.CatalogConfig(
        snapshot_path=csv_path,
        database_path=db_path,
    )
    catalog: typing.Final = repository.ProductsRepository(database_path=db_path)
    assert catalog.restore_from_snapshot(catalog_config.snapshot_path) == EXPECTED_PRODUCT_COUNT

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE products SET price_rub = ? WHERE product_code = ?",
            (CHANGED_PRICE_RUB, TEST_PRODUCT_CODE),
        )

    assert catalog.restore_from_snapshot(catalog_config.snapshot_path) == EXPECTED_PRODUCT_COUNT

    with sqlite3.connect(db_path) as connection:
        restored_price: typing.Final = connection.execute(
            "SELECT price_rub FROM products WHERE product_code = ?",
            (TEST_PRODUCT_CODE,),
        ).fetchone()[0]
        product_count: typing.Final = connection.execute("SELECT COUNT(*) FROM products").fetchone()[0]

    assert restored_price == EXPECTED_PRICE_RUB
    assert product_count == EXPECTED_PRODUCT_COUNT


def test_clear_catalog_removes_generated_database(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path: typing.Final = tmp_path / "products.db"
    database_path.touch()
    loaded_config: typing.Final = config.CatalogConfig(
        snapshot_path=tmp_path / "products.csv",
        database_path=database_path,
    )

    monkeypatch.setattr(cli, "catalog_config", loaded_config)
    monkeypatch.setattr(sys, "argv", ["ai_agent.catalog", "clear"])

    cli.main()

    assert not database_path.exists()
