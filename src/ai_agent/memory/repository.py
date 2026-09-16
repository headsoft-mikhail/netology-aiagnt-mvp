import dataclasses
import datetime
import sqlite3
import typing

from ai_agent import contracts, sqlite_resource

SENSITIVE_MARKERS: typing.Final = (
    "пароль",
    "password",
    "token",
    "токен",
    "номер карты",
    "cvv",
)
POSITIVE_INTEGER_KEYS: typing.Final = {
    contracts.MemoryKey.BUDGET_RUB,
    contracts.MemoryKey.AREA_SQM,
    contracts.MemoryKey.DEVICE_COUNT,
    contracts.MemoryKey.TARIFF_SPEED_MBPS,
}
POSITIVE_INTEGER_ERROR: typing.Final = "Numeric memory values must be positive integers."


@dataclasses.dataclass(kw_only=True, slots=True)
class MemoryRepository(sqlite_resource.BaseSQLiteResource):
    def __post_init__(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS memory_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT,
                    UNIQUE (user_id, memory_type, key)
                )
            """)

    def save_fact(
        self,
        user_id: str,
        key: contracts.MemoryKey,
        value: str,
        *,
        source: str,
        session_id: str = "global",
        expires_at: datetime.datetime | None = None,
    ) -> contracts.MemoryFact:
        normalized_user_id: typing.Final = user_id.strip()
        normalized_value: typing.Final = self._normalize_value(key, value)
        normalized_source: typing.Final = source.strip()
        if not normalized_user_id or not normalized_value or not normalized_source:
            raise ValueError("Memory fields must not be empty.")

        now: typing.Final = datetime.datetime.now(datetime.UTC)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO memory_facts (
                    user_id, session_id, memory_type, key, value, source,
                    created_at, updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (user_id, memory_type, key) DO UPDATE SET
                    session_id = excluded.session_id,
                    value = excluded.value,
                    source = excluded.source,
                    updated_at = excluded.updated_at,
                    expires_at = excluded.expires_at
                """,
                (
                    normalized_user_id,
                    session_id,
                    contracts.MemoryType.LONG_TERM,
                    key,
                    normalized_value,
                    normalized_source,
                    now.isoformat(),
                    now.isoformat(),
                    expires_at.isoformat() if expires_at else None,
                ),
            )
            connection.row_factory = sqlite3.Row
            row: typing.Final = connection.execute(
                """
                SELECT * FROM memory_facts
                WHERE user_id = ? AND memory_type = ? AND key = ?
                """,
                (normalized_user_id, contracts.MemoryType.LONG_TERM, key),
            ).fetchone()
        if row is None:
            raise RuntimeError("Saved memory fact was not found.")
        return self._to_fact(row)

    def get_relevant(self, user_id: str, *, limit: int) -> list[contracts.MemoryFact]:
        now: typing.Final = datetime.datetime.now(datetime.UTC).isoformat()
        with self.connect() as connection:
            connection.row_factory = sqlite3.Row
            rows: typing.Final = connection.execute(
                """
                SELECT * FROM memory_facts
                WHERE user_id = ?
                  AND memory_type = ?
                  AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (user_id, contracts.MemoryType.LONG_TERM, now, limit),
            ).fetchall()
        return [self._to_fact(row) for row in rows]

    def delete_fact(self, user_id: str, key: contracts.MemoryKey) -> None:
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM memory_facts WHERE user_id = ? AND memory_type = ? AND key = ?",
                (user_id, contracts.MemoryType.LONG_TERM, key),
            )

    def clear_user(self, user_id: str) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM memory_facts WHERE user_id = ?", (user_id,))

    def clear(self) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM memory_facts")

    @staticmethod
    def _normalize_value(key: contracts.MemoryKey, value: str) -> str:
        normalized_value: typing.Final = value.strip()
        if any(marker in normalized_value.lower() for marker in SENSITIVE_MARKERS):
            raise ValueError("Sensitive data must not be stored in memory.")
        if key in POSITIVE_INTEGER_KEYS and (not normalized_value.isdecimal() or int(normalized_value) <= 0):
            raise ValueError(POSITIVE_INTEGER_ERROR)
        return normalized_value

    @staticmethod
    def _to_fact(row: sqlite3.Row) -> contracts.MemoryFact:
        return contracts.MemoryFact(
            id=row["id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            memory_type=row["memory_type"],
            key=row["key"],
            value=row["value"],
            source=row["source"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row["expires_at"],
        )
