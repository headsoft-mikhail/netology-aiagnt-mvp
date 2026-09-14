import typing

from ai_agent import contracts
from ai_agent.catalog import models
from ai_agent.catalog.repository import ProductsRepository

NO_PRODUCTS_MESSAGE: typing.Final = "В каталоге нет товаров, соответствующих заданным фильтрам."
CATALOG_UNAVAILABLE_MESSAGE: typing.Final = "Не удалось выполнить поиск по каталогу товаров."
SEARCH_PRODUCTS_TOOL_NAME: typing.Final = "search_products"


def search_products(
    args: models.ProductSearchInput,
    catalog: ProductsRepository,
) -> models.ProductSearchResult:
    """Find current products by typed category-specific filters.

    Call this Tool when the user asks what is available, requests a product
    recommendation, price or stock information, or supplies concrete catalog
    filters. Do not call it for general setup instructions or policies that do
    not require current product data.
    """
    try:
        products: typing.Final = catalog.search(args.filters, args.limit)
    except Exception:  # noqa: BLE001
        return models.ProductSearchResult(
            status=contracts.ToolStatus.ERROR,
            error=contracts.ToolError(
                code=contracts.ErrorCode.CATALOG_UNAVAILABLE,
                message=CATALOG_UNAVAILABLE_MESSAGE,
            ),
        )

    if not products:
        return models.ProductSearchResult(
            status=contracts.ToolStatus.NO_RESULTS,
            error=contracts.ToolError(
                code=contracts.ErrorCode.NO_RESULTS,
                message=NO_PRODUCTS_MESSAGE,
            ),
        )

    return models.ProductSearchResult(
        status=contracts.ToolStatus.OK,
        products=products,
        total=len(products),
    )
