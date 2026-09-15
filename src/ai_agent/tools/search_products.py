import dataclasses
import logging
import typing

import pydantic

from ai_agent import contracts
from ai_agent.catalog import models
from ai_agent.catalog.repository import ProductsRepository
from ai_agent.tools import protocol

NO_PRODUCTS_MESSAGE: typing.Final = "В каталоге нет товаров, соответствующих заданным фильтрам."
CATALOG_UNAVAILABLE_MESSAGE: typing.Final = "Не удалось выполнить поиск по каталогу товаров."
SEARCH_PRODUCTS_TOOL_NAME: typing.Final = "search_products"
LOGGER_OBJ: typing.Final = logging.getLogger(__name__)


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
    except Exception:
        LOGGER_OBJ.debug("Catalog search failed", exc_info=True)
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


@dataclasses.dataclass(kw_only=True, slots=True)
class ProductSearchTool:
    catalog: ProductsRepository
    name: typing.ClassVar[str] = SEARCH_PRODUCTS_TOOL_NAME
    input_model: typing.ClassVar[type[pydantic.BaseModel]] = models.ProductSearchInput

    def invoke(self, tool_input: pydantic.BaseModel) -> protocol.ToolResult:
        if not isinstance(tool_input, models.ProductSearchInput):
            raise TypeError("ProductSearchTool received an invalid input model.")
        return search_products(tool_input, self.catalog)
