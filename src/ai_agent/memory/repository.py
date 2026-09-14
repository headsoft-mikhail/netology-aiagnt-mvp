import dataclasses
import typing

from ai_agent import sqlite

MISSING_USER_FACTS_STR: typing.Final = "Факты о пользователе отсутствуют."


@dataclasses.dataclass(kw_only=True, slots=True)
class MemoryRepository(sqlite.BaseSQLiteResource):
    def __post_init__(self) -> None:
        with self.connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    facts TEXT
                )
            """)

    def get_facts(self, user_id: str) -> str:
        """Получить факты из долговременной памяти пользователя."""
        with self.connect() as connection:
            cursor: typing.Final = connection.execute(
                "SELECT facts FROM user_profiles WHERE user_id = ?",
                (user_id,),
            )
            row: typing.Final = cursor.fetchone()
        return row[0] if row else MISSING_USER_FACTS_STR

    def save_fact(self, user_id: str, new_facts: str) -> None:
        """Сохранить или заменить факты пользователя."""
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO user_profiles (user_id, facts)
                VALUES (?, ?)
                """,
                (user_id, new_facts),
            )

    def clear(self) -> None:
        """Удалить все записи памяти, сохранив файл и схему базы данных."""
        with self.connect() as connection:
            connection.execute("DELETE FROM user_profiles")
