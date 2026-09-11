import sqlite3
import typing

import constants


class SQLiteMemoryStore:
    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self._init_db()
        self.create_test_user()

    def _init_db(self):
        """Инициализация базы данных и создание таблицы долгосрочной памяти."""
        with sqlite3.connect(self.db_path) as conn:
            cursor: typing.Final = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS engineer_profiles (
                    user_id TEXT PRIMARY KEY,
                    facts TEXT
                )
            """)
            conn.commit()

    def create_test_user(self):
        """Создание тестового профиля для проверочных сценариев."""
        with sqlite3.connect(self.db_path) as conn:
            cursor: typing.Final = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO engineer_profiles (user_id, facts)
                VALUES (?, ?)
            """,
                (constants.TEST_USER_ID, constants.TEST_USER_FACTS),
            )
            conn.commit()

    def get_facts(self, user_id: str) -> str | None:
        """Извлечение фактов о пользователе (Long-term Memory)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor: typing.Final = conn.cursor()
            cursor.execute(
                "SELECT facts FROM engineer_profiles WHERE user_id = ?", (user_id,)
            )
            row: typing.Final = cursor.fetchone()
            return row[0] if row else None

    def save_fact(self, user_id: str, new_facts: str):
        """Обновление или сохранение новых фактов (для расширения функционала)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor: typing.Final = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO engineer_profiles (user_id, facts)
                VALUES (?, ?)
            """,
                (user_id, new_facts),
            )
            conn.commit()
