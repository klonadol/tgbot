"""Хранение входящих сообщений и фактического состояния отправки ответов."""

from pathlib import Path
import sqlite3


class Storage:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        try:
            self.connection.execute('PRAGMA journal_mode=WAL')
            self.connection.execute('PRAGMA foreign_keys=ON')
            self._migrate()
        except BaseException:
            self.connection.close()
            raise

    def _migrate(self) -> None:
        version = self.connection.execute('PRAGMA user_version').fetchone()[0]
        if version > 1:
            raise RuntimeError('База данных создана более новой версией приложения')
        if version == 0:
            with self.connection:
                self.connection.executescript('''
                    BEGIN IMMEDIATE;
                    CREATE TABLE messages (
                        id INTEGER PRIMARY KEY,
                        chat_id INTEGER NOT NULL,
                        telegram_message_id INTEGER,
                        user_id INTEGER,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                        text TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (status IN ('received', 'pending', 'sent', 'failed', 'unknown')),
                        reply_to_message_id INTEGER,
                        is_error INTEGER NOT NULL DEFAULT 0,
                        error_kind TEXT,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (chat_id, telegram_message_id)
                    );
                    CREATE INDEX idx_messages_context ON messages(chat_id, id);
                    CREATE UNIQUE INDEX idx_messages_one_reply
                        ON messages(chat_id, reply_to_message_id)
                        WHERE role = 'assistant';
                    PRAGMA user_version=1;
                    COMMIT;
                ''')

    def save_incoming(self, chat_id: int, message_id: int, user_id: int | None, text: str) -> bool:
        with self.connection:
            cursor = self.connection.execute('''
                INSERT OR IGNORE INTO messages
                    (chat_id, telegram_message_id, user_id, role, text, status)
                VALUES (?, ?, ?, 'user', ?, 'received')
            ''', (chat_id, message_id, user_id, text))
        return cursor.rowcount == 1

    def context(self, chat_id: int, limit: int = 20) -> list[dict[str, str]]:
        rows = self.connection.execute('''
            SELECT role, text FROM messages
            WHERE chat_id=? AND status IN ('received', 'sent') AND is_error=0
            ORDER BY id DESC LIMIT ?
        ''', (chat_id, limit)).fetchall()
        return [{'role': role, 'content': content} for role, content in reversed(rows)]

    def prepare_reply(self, chat_id: int, incoming_message_id: int, text: str, is_error: bool) -> int | None:
        with self.connection:
            cursor = self.connection.execute('''
                INSERT OR IGNORE INTO messages
                    (chat_id, role, text, status, reply_to_message_id, is_error)
                VALUES (?, 'assistant', ?, 'pending', ?, ?)
            ''', (chat_id, text, incoming_message_id, int(is_error)))
        return cursor.lastrowid if cursor.rowcount == 1 else None

    def finish_reply(self, record_id: int, *, status: str, message_id: int | None = None, error_kind: str | None = None) -> None:
        if status not in ('sent', 'failed', 'unknown'):
            raise ValueError('Недопустимое состояние отправки')
        if status == 'sent' and message_id is None:
            raise ValueError('Для отправленного ответа нужен идентификатор Telegram')
        with self.connection:
            self.connection.execute('''
                UPDATE messages SET status=?, telegram_message_id=?, error_kind=?
                WHERE id=? AND status='pending'
            ''', (status, message_id, error_kind, record_id))

    def close(self) -> None:
        self.connection.close()
