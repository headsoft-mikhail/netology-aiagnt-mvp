import csv
import dataclasses
import pathlib
import sqlite3
import typing

from ai_agent import sqlite
from ai_agent.catalog import models

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


@dataclasses.dataclass(kw_only=True, slots=True)
class ProductsRepository(sqlite.BaseSQLiteResource):
    def restore_from_snapshot(self, snapshot_path: pathlib.Path) -> int:
        """Atomically restore the SQLite catalog from a versioned CSV snapshot."""
        products: typing.Final = self._load_products(snapshot_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_db_path: typing.Final = self.database_path.with_suffix(f"{self.database_path.suffix}.tmp")
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

            temporary_db_path.replace(self.database_path)
        finally:
            temporary_db_path.unlink(missing_ok=True)

        return len(products)

    @staticmethod
    def _load_products(snapshot_path: pathlib.Path) -> list[models.Product]:
        with snapshot_path.open("r", encoding="utf-8", newline="") as source_file:
            rows: typing.Final = csv.DictReader(source_file)
            products: typing.Final[list[models.Product]] = []

            for source_row in rows:
                row: dict[str, object] = dict(source_row)
                row["price_rub"] = int(source_row["price_rub"])
                row["in_stock"] = ProductsRepository._parse_boolean(source_row["in_stock"])

                for field_name in OPTIONAL_INTEGER_FIELDS:
                    value = source_row[field_name].strip()
                    row[field_name] = int(value) if value else None

                for field_name in OPTIONAL_BOOLEAN_FIELDS:
                    row[field_name] = ProductsRepository._parse_boolean(source_row[field_name])

                row["connection_type"] = source_row["connection_type"] or None
                products.append(models.Product.model_validate(row))

        if not products:
            raise ValueError("Product catalog snapshot is empty.")
        return products

    @staticmethod
    def _parse_boolean(value: str) -> bool | None:
        normalized_value: typing.Final = value.strip().lower()
        if not normalized_value:
            return None
        if normalized_value == "true":
            return True
        if normalized_value == "false":
            return False
        raise ValueError(f"Unsupported boolean value: {value}")

    def search(self, filters: models.ProductFilters, limit: int) -> list[models.Product]:
        if not self.database_path.exists():
            raise FileNotFoundError(f"Catalog database does not exist: {self.database_path}")

        conditions: typing.Final[list[str]] = ["category = ?"]
        parameters: typing.Final[list[str | int]] = [filters.category.value]

        if filters.max_price_rub is not None:
            conditions.append("price_rub <= ?")
            parameters.append(filters.max_price_rub)
        if filters.in_stock is not None:
            conditions.append("in_stock = ?")
            parameters.append(int(filters.in_stock))
        if filters.brands:
            normalized_brands: typing.Final = [brand.strip().lower() for brand in filters.brands if brand.strip()]
            if normalized_brands:
                placeholders: typing.Final = ", ".join("?" for _ in normalized_brands)
                conditions.append(f"lower(brand) IN ({placeholders})")
                parameters.extend(normalized_brands)
        if filters.excluded_brands:
            normalized_excluded_brands: typing.Final = [
                brand.strip().lower() for brand in filters.excluded_brands if brand.strip()
            ]
            if normalized_excluded_brands:
                excluded_placeholders: typing.Final = ", ".join("?" for _ in normalized_excluded_brands)
                conditions.append(f"lower(brand) NOT IN ({excluded_placeholders})")
                parameters.extend(normalized_excluded_brands)

        if isinstance(filters, models.RouterFilters):
            self._add_minimum(conditions, parameters, "wifi_generation", filters.min_wifi_generation)
            self._add_minimum(conditions, parameters, "max_wireless_speed_mbps", filters.min_wireless_speed_mbps)
            self._add_minimum(conditions, parameters, "wan_speed_mbps", filters.min_wan_speed_mbps)
            self._add_minimum(conditions, parameters, "lan_ports", filters.min_lan_ports)
            self._add_boolean(conditions, parameters, "mesh_support", filters.mesh_support)
        elif isinstance(filters, models.MeshSystemFilters):
            self._add_minimum(conditions, parameters, "wifi_generation", filters.min_wifi_generation)
            self._add_minimum(conditions, parameters, "wan_speed_mbps", filters.min_wan_speed_mbps)
            self._add_minimum(conditions, parameters, "nodes", filters.min_nodes)
            self._add_minimum(conditions, parameters, "coverage_sqm", filters.min_coverage_sqm)
        elif isinstance(filters, models.NetworkSwitchFilters):
            self._add_minimum(conditions, parameters, "port_count", filters.min_port_count)
            self._add_minimum(conditions, parameters, "port_speed_mbps", filters.min_port_speed_mbps)
            self._add_boolean(conditions, parameters, "managed", filters.managed)
            self._add_boolean(conditions, parameters, "poe", filters.poe)
        elif isinstance(filters, models.WifiAdapterFilters):
            self._add_minimum(conditions, parameters, "wifi_generation", filters.min_wifi_generation)
            self._add_minimum(conditions, parameters, "max_wireless_speed_mbps", filters.min_wireless_speed_mbps)
            if filters.connection_type is not None:
                conditions.append("connection_type = ?")
                parameters.append(filters.connection_type.value)
        elif isinstance(filters, models.AccessPointFilters):
            self._add_minimum(conditions, parameters, "wifi_generation", filters.min_wifi_generation)
            self._add_minimum(conditions, parameters, "max_wireless_speed_mbps", filters.min_wireless_speed_mbps)
            self._add_boolean(conditions, parameters, "poe", filters.poe)

        parameters.append(limit)
        query: typing.Final = f"""
            SELECT * FROM products
            WHERE {" AND ".join(conditions)}
            ORDER BY in_stock DESC, price_rub ASC, product_code ASC
            LIMIT ?
        """

        with self.connect() as connection:
            connection.row_factory = sqlite3.Row
            rows: typing.Final = connection.execute(query, parameters).fetchall()
        return [models.Product.model_validate(dict(row)) for row in rows]

    @staticmethod
    def _add_minimum(
        conditions: list[str],
        parameters: list[str | int],
        field_name: str,
        value: int | None,
    ) -> None:
        if value is not None:
            conditions.append(f"{field_name} >= ?")
            parameters.append(value)

    @staticmethod
    def _add_boolean(
        conditions: list[str],
        parameters: list[str | int],
        field_name: str,
        value: bool | None,
    ) -> None:
        if value is not None:
            conditions.append(f"{field_name} = ?")
            parameters.append(int(value))
