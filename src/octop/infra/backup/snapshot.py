"""SQLite snapshot helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from urllib.parse import quote

from octop.infra.db.pool import SqlitePool


def _readonly_sqlite_uri(path: Path) -> str:
    """Build a read-only SQLite URI with a percent-encoded filesystem path."""
    encoded = quote(path.resolve().as_posix(), safe="/:")
    return f"file:{encoded}?mode=ro"


def snapshot_sqlite_file(source: Path, dest: Path) -> None:
    """Copy *source* into *dest* using SQLite's online backup API."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(_readonly_sqlite_uri(source), uri=True)
    try:
        dest_conn = sqlite3.connect(dest)
        try:
            src.backup(dest_conn)
        finally:
            dest_conn.close()
    finally:
        src.close()


def restore_sqlite_file(backup_file: Path, target: Path) -> None:
    """Replace on-disk *target* with *backup_file* (server should be stopped)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(_readonly_sqlite_uri(backup_file), uri=True)
    try:
        dest = sqlite3.connect(target)
        try:
            src.backup(dest)
            dest.execute("PRAGMA wal_checkpoint(FULL)")
        finally:
            dest.close()
    finally:
        src.close()


def restore_sqlite_into_pool(backup_file: Path, pool: SqlitePool) -> None:
    """Merge a backup file into the live pooled connection."""
    src = sqlite3.connect(_readonly_sqlite_uri(backup_file), uri=True)
    try:
        with pool.connect() as live:
            src.backup(live)
            live.execute("PRAGMA wal_checkpoint(FULL)")
    finally:
        src.close()


# ---------------------------------------------------------------------------
# User helpers for migration restores
# ---------------------------------------------------------------------------

# All columns in the users table (must match schema in 001_initial.sql).
_USER_COLUMNS = (
    "id",
    "username",
    "password_hash",
    "role",
    "display_name",
    "disabled",
    "created_at",
    "locale",
    "preferences_json",
    "login_failed_count",
    "login_locked_until",
)
_USER_COLS_SQL = ", ".join(_USER_COLUMNS)
_USER_PLACEHOLDERS = ", ".join("?" for _ in _USER_COLUMNS)


def capture_users_from_pool(pool: DBPool) -> list[tuple[object, ...]]:
    """Return all rows from the live *users* table as plain tuples."""
    with pool.connect() as conn:
        rows = conn.execute(f"SELECT {_USER_COLS_SQL} FROM users").fetchall()
    return [tuple(r) for r in rows]


def restore_users_into_pool(pool: DBPool, users: list[tuple[object, ...]]) -> None:
    """Replace the *users* table content with *users* (previously captured rows).

    Existing rows are deleted first so that the restore is idempotent even when
    the DB was swapped in-place by :func:`restore_sqlite_into_pool`.
    """
    if not users:
        return
    with pool.transaction() as conn:
        conn.execute("DELETE FROM users")
        conn.executemany(
            f"INSERT INTO users({_USER_COLS_SQL}) VALUES ({_USER_PLACEHOLDERS})",
            users,
        )
