import typing

import constants
import database
from agent import MockAgentRunner


def start_agent():
    print("Инициализация окружения ИИ-агента поддержки...")
    db: typing.Final = database.SQLiteMemoryStore()
    agent: typing.Final = MockAgentRunner(db)

    user_id: typing.Final = constants.TEST_USER_ID

    print("\n" + "=" * 60)
    print(f" СИСТЕМА ПРИВЕТСТВИЯ АГЕНТА (Авторизован: {user_id})")
    print("=" * 60)
    print("Агент запущен и готов к работе в терминале.")
    print("Введите 'exit' или 'выход' для завершения сессии.\n")

    while True:
        try:
            user_input = input("Инженер >>> ")
            if user_input.lower() in ["exit", "выход"]:
                print("Сессия завершена. До встречи!")
                break

            if not user_input.strip():
                continue

            # Запуск агентского цикла
            result = agent.run(user_id, user_input)

            # Вывод ответа пользователю
            print(f"\nАгент: {result['agent_response']}\n")

            # Вывод логов отладки (Системное логирование из ТЗ)
            print("-" * 40)
            print(" [СИСТЕМНЫЙ ЛОГ ОТЛАДКИ]")
            print(f"Вызванный инструмент: {result.get('tool_called')}")
            if result.get("tool_called"):
                print(f"Переданные аргументы: {result.get('tool_args')}")
                print(f"Результат выполнения кода: {result.get('tool_result')}")
            if result.get("error"):
                print(
                    f"Статус валидации контракта: ИМЕЮТСЯ ОШИБКИ ({result.get('error')})"
                )
            print("-" * 40 + "\n")

        except KeyboardInterrupt:
            print("\nСессия прервана.")
            break
