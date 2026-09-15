import dataclasses
import typing

import pydantic

from ai_agent import contracts, prompts
from ai_agent.catalog import models as catalog_models
from ai_agent.llm import client, models

ResponseModel = typing.TypeVar("ResponseModel", bound=pydantic.BaseModel)


@dataclasses.dataclass(kw_only=True, slots=True)
class LLMService:
    chat_client: client.ChatClientProtocol

    @property
    def model_name(self) -> str:
        return self.chat_client.model_name

    def plan(
        self,
        query: str,
        memory_facts: list[contracts.MemoryFact],
        conversation: list[dict[str, str]],
    ) -> models.AgentPlan:
        return self._complete_model(
            models.AgentPlan,
            prompts.compile_system_prompt(memory_facts),
            prompts.compile_planning_prompt(query, memory_facts, conversation),
        )

    def create_product_search(
        self,
        query: str,
        memory_facts: list[contracts.MemoryFact],
        knowledge_context: str,
    ) -> catalog_models.ProductSearchInput:
        return self._complete_model(
            catalog_models.ProductSearchInput,
            prompts.compile_system_prompt(memory_facts),
            prompts.compile_product_filters_prompt(query, memory_facts, knowledge_context),
        )

    def create_final_answer(
        self,
        query: str,
        memory_facts: list[contracts.MemoryFact],
        knowledge_result: contracts.KnowledgeSearchResult | None,
        catalog_result: catalog_models.ProductSearchResult | None,
    ) -> models.FinalAnswerDraft:
        return self._complete_model(
            models.FinalAnswerDraft,
            prompts.compile_system_prompt(memory_facts),
            prompts.compile_final_answer_prompt(
                query,
                memory_facts,
                knowledge_result,
                catalog_result,
            ),
        )

    def _complete_model(
        self,
        model_type: type[ResponseModel],
        system_prompt: str,
        user_prompt: str,
    ) -> ResponseModel:
        raw_response: typing.Final = self.chat_client.complete(
            system_prompt,
            user_prompt,
            json_mode=True,
        )
        try:
            return model_type.model_validate_json(raw_response)
        except pydantic.ValidationError as error:
            raise client.InvalidLLMResponseError(f"LLM returned invalid {model_type.__name__}.") from error
