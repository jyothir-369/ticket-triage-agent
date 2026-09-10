"""Alembic environment configuration — async SQLAlchemy."""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from src.config import get_settings
from src.models import Base

# Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set the sqlalchemy.url from application settings
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


def _prepare_asyncpg_url_and_args(url: str):
    """Strip libpq-only query params and translate sslmode to asyncpg ssl arg."""
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    asyncpg_incompatible = {
        "sslmode", "channel_binding", "sslrootcert", "sslcert", "sslkey",
    }
    cleaned = {k: v for k, v in query_params.items() if k.lower() not in asyncpg_incompatible}
    clean_query = urlencode(cleaned, doseq=True) if cleaned else ""
    clean_url = urlunparse(parsed._replace(query=clean_query))

    connect_args = {}
    sslmode = query_params.get("sslmode", [None])[0]
    if sslmode and sslmode != "disable":
        connect_args["ssl"] = True

    return clean_url, connect_args


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode (async)."""
    raw_url = config.get_main_option("sqlalchemy.url")
    clean_url, connect_args = _prepare_asyncpg_url_and_args(raw_url)

    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        url=clean_url,
        connect_args=connect_args,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
