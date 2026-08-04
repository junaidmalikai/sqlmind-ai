"""Database connection management with SQLAlchemy."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

from utils.logging_config import get_logger

logger = get_logger(__name__)

DialectName = Literal["postgresql", "mysql", "sqlite"]


@dataclass
class DatabaseConfig:
    """Connection parameters for a SQL database."""

    dialect: DialectName
    host: str = "localhost"
    port: int | None = None
    username: str = ""
    password: str = ""
    database: str = ""
    sqlite_path: str = ""
    display_name: str = ""
    connect_timeout: int = 10

    def __post_init__(self) -> None:
        if not self.display_name:
            if self.dialect == "sqlite":
                self.display_name = Path(self.sqlite_path).name or "sqlite"
            else:
                self.display_name = f"{self.dialect}://{self.database}"

        if self.port is None:
            defaults = {"postgresql": 5432, "mysql": 3306, "sqlite": 0}
            self.port = defaults.get(self.dialect, 0)

    def to_safe_dict(self) -> dict[str, Any]:
        """Serialize without password for UI / logs."""
        return {
            "dialect": self.dialect,
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "database": self.database,
            "sqlite_path": self.sqlite_path,
            "display_name": self.display_name,
        }

    def build_url(self) -> str:
        """Build a SQLAlchemy connection URL."""
        if self.dialect == "sqlite":
            path = self.sqlite_path or ":memory:"
            if path != ":memory:":
                # Absolute path for reliability
                path = str(Path(path).resolve())
            return f"sqlite:///{path}"

        user = quote_plus(self.username)
        pwd = quote_plus(self.password)
        host = self.host or "localhost"
        port = self.port
        db = self.database

        if self.dialect == "postgresql":
            return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"
        if self.dialect == "mysql":
            return f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}"
        raise ValueError(f"Unsupported dialect: {self.dialect}")


@dataclass
class DatabaseConnector:
    """Creates and validates SQLAlchemy engines safely."""

    config: DatabaseConfig
    engine: Engine | None = field(default=None, init=False, repr=False)

    def connect(self) -> Engine:
        """Create (or reuse) a SQLAlchemy engine."""
        if self.engine is not None:
            return self.engine

        url = self.config.build_url()
        connect_args: dict[str, Any] = {}

        if self.config.dialect == "sqlite":
            connect_args["check_same_thread"] = False
        elif self.config.dialect == "postgresql":
            connect_args["connect_timeout"] = self.config.connect_timeout
        elif self.config.dialect == "mysql":
            connect_args["connect_timeout"] = self.config.connect_timeout

        # Read-only is enforced by SQLSecurityGuard + session flags in QueryExecutor.
        self.engine = create_engine(
            url,
            poolclass=NullPool,
            connect_args=connect_args,
            future=True,
        )
        logger.info(
            "Connected to database '%s' (%s)",
            self.config.display_name,
            self.config.dialect,
        )
        return self.engine

    def test_connection(self) -> tuple[bool, str]:
        """Ping the database; return (ok, message)."""
        try:
            engine = self.connect()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True, "Connection successful"
        except Exception as exc:  # noqa: BLE001 — surface to UI
            logger.exception("Connection test failed")
            self.dispose()
            return False, str(exc)

    def dispose(self) -> None:
        """Dispose engine and release connections."""
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None

    @staticmethod
    def save_uploaded_sqlite(file_bytes: bytes, filename: str, dest_dir: str) -> str:
        """Persist an uploaded SQLite file and return its path."""
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename).name
        target = dest / safe_name
        target.write_bytes(file_bytes)
        return str(target.resolve())

    @staticmethod
    def create_temp_sqlite(file_bytes: bytes) -> str:
        """Write upload to a temp file (Streamlit Cloud friendly)."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.write(file_bytes)
        tmp.flush()
        tmp.close()
        return tmp.name
