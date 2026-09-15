import pathlib
import typing

import pytest

from ai_agent import contracts
from ai_agent.catalog import models, repository
from ai_agent.tools import search_products

EXPECTED_ROUTER_PRODUCT_CODES: typing.Final = {
    "RTR-AS-BE58",
    "RTR-KN-GIGA",
    "RTR-TP-BE65",
}
EXPECTED_MANAGED_POE_SWITCH_PRODUCT_CODES: typing.Final = [
    "SWT-TP-SG108PE",
    "SWT-UB-USW-L8",
]
EXCLUDED_BRAND: typing.Final = "D-Link"


@pytest.fixture
def product_catalog(tmp_path: pathlib.Path) -> repository.ProductsRepository:
    db_path: typing.Final = tmp_path / "products.db"
    csv_path: typing.Final = pathlib.Path(repository.__file__).parent / "data" / "products.csv"
    catalog: typing.Final = repository.ProductsRepository(database_path=db_path)
    catalog.restore_from_snapshot(csv_path)
    return catalog


def test_search_products_filters_routers_by_characteristics(
    product_catalog: repository.ProductsRepository,
) -> None:
    args: typing.Final = models.ProductSearchInput(
        filters=models.RouterFilters(
            category=models.ProductCategory.ROUTER,
            min_wifi_generation=6,
            min_wan_speed_mbps=2500,
            excluded_brands=[EXCLUDED_BRAND],
        )
    )

    result: typing.Final = search_products.search_products(args, product_catalog)

    assert result.status is contracts.ToolStatus.OK
    assert {product.product_code for product in result.products} == EXPECTED_ROUTER_PRODUCT_CODES
    assert all(product.brand != EXCLUDED_BRAND for product in result.products)


def test_search_products_uses_category_specific_switch_filters(
    product_catalog: repository.ProductsRepository,
) -> None:
    args: typing.Final = models.ProductSearchInput(
        filters=models.NetworkSwitchFilters(
            category=models.ProductCategory.NETWORK_SWITCH,
            min_port_count=8,
            managed=True,
            poe=True,
        )
    )

    result: typing.Final = search_products.search_products(args, product_catalog)

    assert result.status is contracts.ToolStatus.OK
    assert [product.product_code for product in result.products] == EXPECTED_MANAGED_POE_SWITCH_PRODUCT_CODES


def test_search_products_returns_structured_no_results(
    product_catalog: repository.ProductsRepository,
) -> None:
    args: typing.Final = models.ProductSearchInput(
        filters=models.MeshSystemFilters(
            category=models.ProductCategory.MESH_SYSTEM,
            max_price_rub=100,
        )
    )

    result: typing.Final = search_products.search_products(args, product_catalog)

    assert result.status is contracts.ToolStatus.NO_RESULTS
    assert result.error is not None
    assert result.error.code is contracts.ErrorCode.NO_RESULTS
    assert result.error.message == search_products.NO_PRODUCTS_MESSAGE


def test_search_products_returns_safe_error_for_unavailable_catalog(
    tmp_path: pathlib.Path,
) -> None:
    catalog: typing.Final = repository.ProductsRepository(database_path=tmp_path / "missing.db")
    args: typing.Final = models.ProductSearchInput(
        filters=models.AccessPointFilters(category=models.ProductCategory.ACCESS_POINT)
    )

    result: typing.Final = search_products.search_products(args, catalog)

    assert result.status is contracts.ToolStatus.ERROR
    assert result.error is not None
    assert result.error.code is contracts.ErrorCode.CATALOG_UNAVAILABLE
    assert result.error.message == search_products.CATALOG_UNAVAILABLE_MESSAGE
