import typing
from typing import Any

import constants
import tools
from database import SQLiteMemoryStore
from prompts import compile_system_prompt


class MockAgentRunner:
    """
    Готовый runtime-компонент агента.
    Эмулирует цикл LangGraph/LangChain (Анализ -> Вызов Tool -> Ответ).
    """

    def __init__(self, db: SQLiteMemoryStore):
        self.db = db
        # Краткосрочная память (Short-term memory) в рамках текущей сессии
        self.chat_history: list[dict[str, str]] = []

    def run(self, user_id: str, user_message: str) -> dict[str, Any]:
        # 1. Формируем актуальный системный промпт (включая Long-term memory)
        system_prompt: typing.Final = compile_system_prompt(user_id, self.db)

        # 2. Сохраняем сообщение пользователя в краткосрочную память
        self.chat_history.append({"role": "user", "content": user_message})

        # Переводим сообщение в нижний регистр для детерминированного мок-анализа намерения (Intent Detection)
        msg_lower: typing.Final = user_message.lower()

        # Имитация логики «мышления» модели (Orchestrator/Planner)
        # Проверяем, есть ли запрос на вызов Tool (Шаг 1)
        if "доступ" in msg_lower or "прав" in msg_lower or "id" in msg_lower:
            # Извлекаем ID из текста сообщения для передачи в контракт
            # Упрощенный парсинг для учебного мока
            engineer_id: typing.Final = (
                constants.EXISTING_ENGINEER_ID
                if str(constants.EXISTING_ENGINEER_ID) in msg_lower
                else (
                    constants.INEXISTENT_ENGINEER_ID
                    if str(constants.INEXISTENT_ENGINEER_ID) in msg_lower
                    else None
                )
            )

            if engineer_id is None:
                return {
                    "agent_response": "Для проверки прав, пожалуйста, укажите ваш числовой ID инженера.",
                    "tool_called": None,
                    "tool_status": "skipped",
                }

            # Эмуляция генерации Function Calling от модели.
            # Модель пытается собрать аргументы по Pydantic-контракту, созданному студентом
            try:
                # Проверяем, создал ли студент поля в контракте (валидация на уровне рантайма)
                input_data: typing.Final = tools.PermissionCheckInput(
                    engineer_id=engineer_id
                )

                # Вызов инструмента (Выполнение детерминированного кода)
                tool_result_str: typing.Final = tools.get_engineer_permissions(
                    input_data
                )

                # Модель получает результат инструмента и генерирует финальный ответ
                if "error" in tool_result_str:
                    final_reply = f"Произошла ошибка при верификации прав. Ответ системы: {tool_result_str}"
                else:
                    final_reply = f"Я проверил ваши доступы через корпоративный шлюз. Вот актуальные данные: {tool_result_str}"

                return {
                    "agent_response": final_reply,
                    "tool_called": tools.PERMISSIONS_TOOL_NAME,
                    "tool_args": {"engineer_id": engineer_id},
                    "tool_result": tool_result_str,
                }

            except Exception as e:  # noqa: BLE001
                # Ошибка валидации контракта Pydantic (если студент не настроил класс)
                return {
                    "agent_response": "Критическая ошибка: Агент не смог сформировать валидный JSON-контракт для инструмента. Проверьте класс PermissionCheckInput.",
                    "tool_called": tools.PERMISSIONS_TOOL_NAME,
                    "error": str(e),
                }

        # Проверяем, опирается ли модель на Long-term memory (Шаг 2)
        elif "проект" in msg_lower or "язык" in msg_lower or "стек" in msg_lower:
            # Если студент правильно интегрировал память в prompts.py,
            # Факты о проекте и языке будут внутри system_prompt.
            if (
                f"Проект {constants.TEST_USER_PROJECT}" in system_prompt
                and constants.TEST_USER_LANGUAGE in system_prompt
            ):
                reply = (
                    "Согласно вашей долговременной карте профиля, вы сейчас "
                    f"закреплены за проектом **{constants.TEST_USER_PROJECT}** и "
                    "вашим основным языком является "
                    f"**{constants.TEST_USER_LANGUAGE}** "
                    f"(уровень {constants.TEST_USER_LEVEL})."
                )
            else:
                reply = "К сожалению, в моей системной памяти сейчас нет информации о вашем текущем проекте или технологическом стеке. Проверьте интеграцию Long-term памяти."

            return {"agent_response": reply, "tool_called": None}

        else:
            # Обычный фоллбэк ответ без инструментов и памяти
            return {
                "agent_response": "Привет! Я готов помочь. Вы можете спросить меня о вашем текущем проекте (проверка Long-term памяти) или попросить проверить доступы по вашему ID (вызов Tools).",
                "tool_called": None,
            }
