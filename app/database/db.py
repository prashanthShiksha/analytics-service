import asyncio
import logging
from pathlib import Path

import asyncpg

from app.config import settings

logger = logging.getLogger("analytics_service.db")

BASE_DIR = Path(__file__).resolve().parents[2]
SCHEMA_FILE = BASE_DIR / "schema.sql"
SEED_PROMPTS_FILE = BASE_DIR / "seed_prompts.sql"
SEED_THEMES_FILE = BASE_DIR / "seed_themes.sql"

class Database:
    def __init__(self):
        self.pool = None
        self._connect_lock = None

    async def initialize_schema(self) -> None:
        """
        Creates the required database tables if they do not already exist.
        This prevents startup failures when the configured PostgreSQL database is empty.
        """
        if not self.pool:
            raise RuntimeError("Database pool is not initialized. Call connect() first.")

        async with self.pool.acquire() as conn:
            if settings.RESET_DB:
                await conn.execute("DROP SCHEMA IF EXISTS public CASCADE;")
                await conn.execute("CREATE SCHEMA public;")
                logger.warning("Database schema reset requested; dropped and recreated public schema.")

            schema_sql = SCHEMA_FILE.read_text(encoding="utf-8")
            schema_sql = schema_sql.replace("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS ")
            schema_sql = schema_sql.replace("CREATE INDEX ", "CREATE INDEX IF NOT EXISTS ")
            await conn.execute(schema_sql)

            # Dynamic migrations — add new columns to existing tables if they don't exist
            await conn.execute(
                "ALTER TABLE analysis_results ADD COLUMN IF NOT EXISTS content_quality TEXT;"
            )
            await conn.execute(
                "ALTER TABLE analysis_results ADD COLUMN IF NOT EXISTS similarity_score FLOAT;"
            )

            # Always run the seed script to keep prompts in sync with seed_prompts.sql
            seed_sql = SEED_PROMPTS_FILE.read_text(encoding="utf-8")
            await conn.execute(seed_sql)

            # Always run the themes seed script to seed initial approved taxonomies
            seed_themes_sql = SEED_THEMES_FILE.read_text(encoding="utf-8")
            await conn.execute(seed_themes_sql)

            logger.info("Database schema initialized successfully.")

    async def connect(self) -> None:
        """
        Creates the asyncpg connection pool if not already initialized.
        """
        if self.pool:
            return

        current_loop = asyncio.get_running_loop()
        if self._connect_lock is None or getattr(self._connect_lock, '_loop', None) is not current_loop:
            self._connect_lock = asyncio.Lock()

        async with self._connect_lock:
            if self.pool:
                return
            try:
                self.pool = await asyncpg.create_pool(
                    dsn=settings.DATABASE_URL,
                    min_size=2,
                    max_size=10
                )
                await self.initialize_schema()
                logger.info("Database connection pool established successfully.")
            except Exception as e:
                logger.error(f"Failed to create database connection pool: {e}")
                raise

    async def disconnect(self) -> None:
        """
        Closes the database connection pool.
        """
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Database connection pool closed.")

    async def get_connection(self) -> asyncpg.Connection:
        """
        Acquires a connection from the pool. Ensures the pool is connected.
        """
        if not self.pool:
            await self.connect()
        return await self.pool.acquire()

    async def release_connection(self, conn: asyncpg.Connection) -> None:
        """
        Releases a connection back to the pool.
        """
        if self.pool and conn:
            await self.pool.release(conn)

db = Database()
