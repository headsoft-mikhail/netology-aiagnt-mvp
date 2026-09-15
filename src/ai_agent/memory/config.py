import pathlib
import typing

import pydantic


class MemoryConfig(pydantic.BaseModel):
    database_path: pathlib.Path = pathlib.Path("src/ai_agent/memory/data/agent_memory.db")
    retrieval_limit: int = pydantic.Field(default=10, gt=0)


memory_config: typing.Final = MemoryConfig()
