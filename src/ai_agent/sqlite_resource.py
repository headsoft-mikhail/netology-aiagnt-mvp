import contextlib
import dataclasses
import pathlib
import sqlite3
import typing


@contextlib.contextmanager
def connect(database_path: pathlib.Path) -> typing.Iterator[sqlite3.Connection]:
    connection: typing.Final = sqlite3.connect(database_path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


@dataclasses.dataclass(kw_only=True, slots=True)
class BaseSQLiteResource:
    database_path: pathlib.Path

    def connect(self) -> contextlib.AbstractContextManager[sqlite3.Connection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        return connect(self.database_path)
