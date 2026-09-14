import dataclasses
import sqlite3
import typing

from ai_agent import sqlite
from ai_agent.catalog import models


@dataclasses.dataclass(kw_only=True, slots=True)
class ProductsRepository(sqlite.BaseSQLiteResource):
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
            ORDER BY in_stock DESC, price_rub ASC, sku ASC
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
