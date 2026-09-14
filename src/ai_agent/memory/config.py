import pathlib
import typing

import pydantic


class MemoryConfig(pydantic.BaseModel):
    database_path: pathlib.Path = pathlib.Path("src/ai_agent/catalog/data/agent_memory.db")


memory_config: typing.Final = MemoryConfig()
