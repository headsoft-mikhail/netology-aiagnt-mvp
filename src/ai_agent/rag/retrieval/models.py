import pydantic


class KnowledgeSearchInput(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(str_strip_whitespace=True)

    query: str = pydantic.Field(min_length=1)
