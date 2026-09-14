import typing

from ai_agent import contracts
from ai_agent.memory.repository import MemoryRepository


class MockAgentRunner:
    """Эмуляция цикла агента: анализ, вызов Tool и формирование ответа."""

    def __init__(self, db: MemoryRepository):
        self.db = db
        self.chat_history: list[dict[str, str]] = []

    def run(self, user_id: str, user_message: str) -> contracts.AgentResponse:
        request: typing.Final = contracts.AgentRequest(user_id=user_id, query=user_message)
        self.chat_history.append({"role": "user", "content": request.query})
        msg_lower: typing.Final = request.query.lower()

        if any(keyword in msg_lower for keyword in ("бюджет", "бренд", "площад", "тариф", "устройств")):
            facts: typing.Final = self.db.get_facts(user_id)
            reply: typing.Final = (
                f"Согласно долговременной памяти: {facts}"
                if facts
                else "В памяти нет сохранённых предпочтений покупателя."
            )

            return contracts.AgentResponse(
                status=contracts.AgentRunStatus.COMPLETED,
                answer=reply,
            )

        return contracts.AgentResponse(
            status=contracts.AgentRunStatus.NEEDS_INPUT,
            answer=("Уточните, нужна инструкция по настройке или подбор сетевого оборудования по характеристикам."),
        )
