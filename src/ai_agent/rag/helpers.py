import sqlite3
import typing

from qdrant_client.local import persistence

SQLITE_THREADSAFE_QUERY: typing.Final = (
    "SELECT compile_options FROM pragma_compile_options WHERE compile_options LIKE 'THREADSAFE=%'"
)


def configure_local_sqlite_thread_check() -> None:
    """Configure local Qdrant without leaking its SQLite probe connection."""
    if persistence.CollectionPersistence.CHECK_SAME_THREAD is not None:
        return

    connection: typing.Final = sqlite3.connect(":memory:")
    try:
        row: typing.Final = connection.execute(SQLITE_THREADSAFE_QUERY).fetchone()
    finally:
        connection.close()

    if row is None:
        raise RuntimeError("SQLite THREADSAFE compile option is unavailable.")

    persistence.CollectionPersistence.CHECK_SAME_THREAD = row[0] != "THREADSAFE=1"
