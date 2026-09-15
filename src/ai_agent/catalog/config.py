import pathlib
import typing

import pydantic


class CatalogConfig(pydantic.BaseModel):
    snapshot_path: pathlib.Path = pathlib.Path("src/ai_agent/catalog/data/products.csv")
    database_path: pathlib.Path = pathlib.Path("src/ai_agent/catalog/data/products.db")


catalog_config: typing.Final = CatalogConfig()
