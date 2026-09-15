import typing

import pydantic

from ai_agent import contracts

CLARIFICATION_MESSAGE: typing.Final = (
    "Уточните задачу: нужна инструкция, проверка конкретного товара или подбор оборудования?"
)
UNSUPPORTED_MESSAGE: typing.Final = "Я могу помочь только с выбором, покупкой и настройкой сетевого оборудования."


class AgentPlan(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", str_strip_whitespace=True)

    actions: list[contracts.AgentAction] = pydantic.Field(min_length=1, max_length=2)
    knowledge_query: str | None = pydantic.Field(default=None, min_length=1)
    memory_update: contracts.MemoryUpdate | None = None
    memory_key: contracts.MemoryKey | None = None
    response_message: str | None = pydantic.Field(default=None, min_length=1)

    @pydantic.model_validator(mode="after")
    def validate_plan(self) -> typing.Self:
        if len(self.actions) != len(set(self.actions)):
            raise ValueError("Plan actions must be unique.")
        if set(self.actions) == {
            contracts.AgentAction.SEARCH_KNOWLEDGE_BASE,
            contracts.AgentAction.SEARCH_PRODUCTS,
        } and self.actions != [
            contracts.AgentAction.SEARCH_KNOWLEDGE_BASE,
            contracts.AgentAction.SEARCH_PRODUCTS,
        ]:
            raise ValueError("Knowledge search must run before product search.")
        if contracts.AgentAction.SEARCH_KNOWLEDGE_BASE in self.actions and not self.knowledge_query:
            raise ValueError("Knowledge search requires knowledge_query.")
        if contracts.AgentAction.UPDATE_MEMORY in self.actions and (
            len(self.actions) != 1 or self.memory_update is None
        ):
            raise ValueError("Memory update must be the only action and requires memory_update.")
        if contracts.AgentAction.DELETE_MEMORY in self.actions and (len(self.actions) != 1 or self.memory_key is None):
            raise ValueError("Memory deletion must be the only action and requires memory_key.")
        if contracts.AgentAction.CLEAR_MEMORY in self.actions and len(self.actions) != 1:
            raise ValueError("Full memory deletion must be the only action.")
        direct_actions: typing.Final = {
            contracts.AgentAction.CLARIFY,
            contracts.AgentAction.UNSUPPORTED,
            contracts.AgentAction.ANSWER,
        }
        if any(action in self.actions for action in direct_actions) and len(self.actions) != 1:
            raise ValueError("Direct response actions must not be combined with other actions.")
        if self.actions == [contracts.AgentAction.CLARIFY] and not self.response_message:
            self.response_message = CLARIFICATION_MESSAGE
        if self.actions == [contracts.AgentAction.UNSUPPORTED] and not self.response_message:
            self.response_message = UNSUPPORTED_MESSAGE
        if self.actions == [contracts.AgentAction.ANSWER] and not self.response_message:
            raise ValueError("Answer action requires response_message.")
        return self


class FinalAnswerDraft(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: str = pydantic.Field(min_length=1)
    product_codes: list[str] = pydantic.Field(default_factory=list)
