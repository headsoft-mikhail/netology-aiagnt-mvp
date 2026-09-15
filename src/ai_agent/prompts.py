import json
import typing

from ai_agent import contracts
from ai_agent.catalog import models as catalog_models
from ai_agent.llm import models as llm_models
from ai_agent.tools.search_knowledge_base import SEARCH_KNOWLEDGE_BASE_TOOL_NAME
from ai_agent.tools.search_products import SEARCH_PRODUCTS_TOOL_NAME

MISSING_MEMORY: typing.Final = "Сохранённые предпочтения отсутствуют."


def compile_system_prompt(memory_facts: list[contracts.MemoryFact]) -> str:
    memory_context: typing.Final = _serialize_memory(memory_facts)
    return (
        "Ты — консультант интернет-магазина домашнего сетевого оборудования. "
        "Работай только в этой предметной области. Не выдумывай товары, цены, "
        "остатки, характеристики, инструкции и сведения о пользователе.\n"
        f"Инструкции и правила доступны только через {SEARCH_KNOWLEDGE_BASE_TOOL_NAME}.\n"
        f"Товары, цены и остатки доступны только через {SEARCH_PRODUCTS_TOOL_NAME}.\n"
        "Если фактических данных недостаточно, запроси уточнение или честно сообщи "
        "об отсутствии результата. Сохраняй память только по явной просьбе пользователя.\n"
        "Никогда не сохраняй пароли, токены, платёжные данные или адреса.\n"
        f"Релевантные предпочтения пользователя:\n{memory_context}"
    )


def compile_planning_prompt(
    query: str,
    memory_facts: list[contracts.MemoryFact],
    conversation: list[dict[str, str]],
) -> str:
    schema: typing.Final = json.dumps(llm_models.AgentPlan.model_json_schema(), ensure_ascii=False)
    return (
        "Определи следующий проверяемый сценарий. Для общего объяснения выбери "
        "search_knowledge_base; для цены, наличия или товаров — search_products; "
        "для подбора допустимы оба действия именно в этом порядке. Для явной просьбы "
        "запомнить предпочтение выбери только update_memory, удалить сохранённое — "
        "delete_memory, удалить все предпочтения — clear_memory. Неполный запрос требует "
        "clarify, непрофильный — unsupported. Не формируй фильтры каталога на этом шаге.\n"
        "Учитывай последние сообщения: слова «он», «его», «этот» относятся к последнему "
        "обсуждавшемуся товару. Для clarify и unsupported всегда заполняй response_message.\n"
        f"Запрос: {query}\n"
        f"Память: {_serialize_memory(memory_facts)}\n"
        f"Последние сообщения сессии: {json.dumps(conversation[-6:], ensure_ascii=False)}\n"
        f"Верни только JSON по схеме: {schema}"
    )


def compile_product_filters_prompt(
    query: str,
    memory_facts: list[contracts.MemoryFact],
    knowledge_context: str,
) -> str:
    schema: typing.Final = json.dumps(catalog_models.ProductSearchInput.model_json_schema(), ensure_ascii=False)
    return (
        "Преобразуй условия покупателя, его предпочтения и найденные правила выбора "
        "в фильтры каталога. Не добавляй ограничения, которых нет в этих данных. "
        "Если база знаний не использовалась, опирайся только на запрос и память.\n"
        f"Запрос: {query}\n"
        f"Память: {_serialize_memory(memory_facts)}\n"
        f"Контекст базы знаний: {knowledge_context or 'не использовался'}\n"
        f"Верни только JSON по схеме: {schema}"
    )


def compile_final_answer_prompt(
    query: str,
    memory_facts: list[contracts.MemoryFact],
    knowledge_result: contracts.KnowledgeSearchResult | None,
    catalog_result: catalog_models.ProductSearchResult | None,
) -> str:
    schema: typing.Final = json.dumps(llm_models.FinalAnswerDraft.model_json_schema(), ensure_ascii=False)
    knowledge_json: typing.Final = (
        knowledge_result.model_dump_json() if knowledge_result is not None else "не использовалась"
    )
    catalog_json: typing.Final = catalog_result.model_dump_json() if catalog_result is not None else "не использовался"
    return (
        "Сформируй краткий итоговый ответ только из переданных фактов. Упоминай только "
        "товары, присутствующие в результате каталога, и дословно переноси их артикулы "
        "из поля product_code, "
        "цены и характеристики. Если Tool вернул no_results или error, не изображай "
        "успешный результат. Называй категорию товара правильно: коммутатор не является "
        "роутером. Не выводи поля со значением null и внутреннее представление модели.\n"
        f"Запрос: {query}\n"
        f"Память: {_serialize_memory(memory_facts)}\n"
        f"Результат базы знаний: {knowledge_json}\n"
        f"Результат каталога: {catalog_json}\n"
        f"Верни только JSON по схеме: {schema}"
    )


def _serialize_memory(memory_facts: list[contracts.MemoryFact]) -> str:
    if not memory_facts:
        return MISSING_MEMORY
    return "\n".join(f"- {fact.key}: {fact.value}" for fact in memory_facts)
