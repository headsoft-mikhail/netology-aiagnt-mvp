import dataclasses
import pathlib
import sqlite3


@dataclasses.dataclass(kw_only=True, slots=True)
class BaseSQLiteResource:
    database_path: pathlib.Path

    def connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.database_path)
