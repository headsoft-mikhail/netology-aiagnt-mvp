import pathlib
import typing

import pydantic_settings


class CatalogConfig(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(
        env_prefix="CATALOG_",
    )

    snapshot_path: pathlib.Path = pathlib.Path("src/ai_agent/catalog/data/products.csv")
    database_path: pathlib.Path = pathlib.Path("src/ai_agent/catalog/data/products.db")


catalog_config: typing.Final = CatalogConfig()
