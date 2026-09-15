import csv
import pathlib
import sqlite3
import typing

from ai_agent.catalog import config, models

OPTIONAL_INTEGER_FIELDS: typing.Final = {
    "wifi_generation",
    "max_wireless_speed_mbps",
    "wan_speed_mbps",
    "lan_ports",
    "nodes",
    "coverage_sqm",
    "port_count",
    "port_speed_mbps",
}
OPTIONAL_BOOLEAN_FIELDS: typing.Final = {"mesh_support", "managed", "poe"}


def _parse_boolean(value: str) -> bool | None:
    normalized_value: typing.Final = value.strip().lower()
    if not normalized_value:
        return None
    if normalized_value == "true":
        return True
    if normalized_value == "false":
        return False
    raise ValueError(f"Unsupported boolean value: {value}")


def _load_products(csv_path: pathlib.Path) -> list[models.Product]:
    with csv_path.open("r", encoding="utf-8", newline="") as source_file:
        rows: typing.Final = csv.DictReader(source_file)
        products: typing.Final[list[models.Product]] = []

        for source_row in rows:
            row: dict[str, object] = dict(source_row)
            row["price_rub"] = int(source_row["price_rub"])
            row["in_stock"] = _parse_boolean(source_row["in_stock"])

            for field_name in OPTIONAL_INTEGER_FIELDS:
                value = source_row[field_name].strip()
                row[field_name] = int(value) if value else None

            for field_name in OPTIONAL_BOOLEAN_FIELDS:
                row[field_name] = _parse_boolean(source_row[field_name])

            row["connection_type"] = source_row["connection_type"] or None
            products.append(models.Product.model_validate(row))

    if not products:
        raise ValueError("Product catalog snapshot is empty.")
    return products


def restore_catalog(
    catalog_config: config.CatalogConfig,
) -> int:
    """Atomically restore the SQLite catalog from the versioned CSV snapshot."""
    products: typing.Final = _load_products(catalog_config.snapshot_path)
    catalog_config.database_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_db_path: typing.Final = catalog_config.database_path.with_suffix(
        f"{catalog_config.database_path.suffix}.tmp"
    )
    temporary_db_path.unlink(missing_ok=True)

    try:
        with sqlite3.connect(temporary_db_path) as connection:
            connection.execute("""
                CREATE TABLE products (
                    product_code TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    brand TEXT NOT NULL,
                    price_rub INTEGER NOT NULL CHECK (price_rub > 0),
                    in_stock INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
                    wifi_generation INTEGER,
                    max_wireless_speed_mbps INTEGER,
                    wan_speed_mbps INTEGER,
                    lan_ports INTEGER,
                    mesh_support INTEGER,
                    nodes INTEGER,
                    coverage_sqm INTEGER,
                    port_count INTEGER,
                    port_speed_mbps INTEGER,
                    managed INTEGER,
                    poe INTEGER,
                    connection_type TEXT
                )
            """)
            connection.executemany(
                """
                INSERT INTO products (
                    product_code, name, category, brand, price_rub, in_stock,
                    wifi_generation, max_wireless_speed_mbps, wan_speed_mbps,
                    lan_ports, mesh_support, nodes, coverage_sqm, port_count,
                    port_speed_mbps, managed, poe, connection_type
                ) VALUES (
                    :product_code, :name, :category, :brand, :price_rub, :in_stock,
                    :wifi_generation, :max_wireless_speed_mbps, :wan_speed_mbps,
                    :lan_ports, :mesh_support, :nodes, :coverage_sqm, :port_count,
                    :port_speed_mbps, :managed, :poe, :connection_type
                )
                """,
                [product.model_dump(mode="json") for product in products],
            )
            connection.execute("CREATE INDEX products_category_idx ON products (category)")
            connection.execute("CREATE INDEX products_price_idx ON products (price_rub)")

        temporary_db_path.replace(catalog_config.database_path)
    finally:
        temporary_db_path.unlink(missing_ok=True)

    return len(products)
