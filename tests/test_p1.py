"""Проверки P1 с явно имитированными внешними API."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import uuid

import httpx

from tgbot.config import Config
from tgbot.llm import LLMClient, LLMError
from tgbot.storage import Storage
from tgbot.telegram import handle_text, ERROR_TEXT


class FakeMessage:
    def __init__(self, chat_id=17, message_id=1, text='Привет', kind='private'):
        self.chat = SimpleNamespace(id=chat_id, type=kind)
        self.message_id = message_id
        self.text = text
        self.from_user = SimpleNamespace(id=27)


class FakeBot:
    def __init__(self, failure=None):
        self.messages = []
        self.failure = failure

    async def send_message(self, *, chat_id, text):
        if self.failure:
            raise self.failure
        self.messages.append((chat_id, text))
        return SimpleNamespace(message_id=1000 + len(self.messages))


class FakeLLM:
    def __init__(self, exception=None):
        self.model = 'модель-теста'
        self.calls = []
        self.exception = exception

    async def reply(self, messages):
        self.calls.append(messages)
        if self.exception:
            raise self.exception
        return 'Ответ модели'


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / 'nested' / 'history.sqlite3'
        self.db = Storage(self.path)

    def tearDown(self):
        self.db.close()
        self.tempdir.cleanup()

    def test_migration_and_foreign_keys(self):
        self.assertEqual(self.db.connection.execute('PRAGMA user_version').fetchone()[0], 1)
        self.assertEqual(self.db.connection.execute('PRAGMA foreign_keys').fetchone()[0], 1)
        self.assertEqual(self.db.connection.execute('PRAGMA journal_mode').fetchone()[0], 'wal')
        self.assertTrue(self.path.exists())

    def test_duplicate_input_and_one_reply(self):
        self.assertTrue(self.db.save_incoming(10, 1, 77, 'Первое'))
        self.assertFalse(self.db.save_incoming(10, 1, 77, 'Повтор'))
        self.assertEqual(self.db.context(10), [{'role': 'user', 'content': 'Первое'}])
        rid = self.db.prepare_reply(10, 1, 'Привет', False)
        self.assertIsNotNone(rid)
        self.assertIsNone(self.db.prepare_reply(10, 1, 'Лишний ответ', False))
        self.assertEqual(len(self.db.context(10)), 1)  # pending не выдаётся за отправленное
        self.db.finish_reply(rid, status='sent', message_id=42)
        self.assertEqual(self.db.context(10)[-1]['content'], 'Привет')

    def test_failed_and_unknown_not_seen_as_sent(self):
        self.db.save_incoming(10, 1, 77, 'Привет')
        rid = self.db.prepare_reply(10, 1, 'Несостоявшийся ответ', False)
        self.db.finish_reply(rid, status='unknown', error_kind='network')
        self.assertNotIn('Несостоявшийся ответ', str(self.db.context(10)))
        self.assertEqual(self.db.connection.execute('SELECT telegram_message_id FROM messages WHERE id=?', (rid,)).fetchone()[0], None)

    def test_restart_keeps_history(self):
        self.db.save_incoming(5, 12, 77, 'Начало истории')
        rid = self.db.prepare_reply(5, 12, 'Продолжение', False)
        self.db.finish_reply(rid, status='sent', message_id=55)
        self.db.close()
        self.db = Storage(self.path)
        self.assertEqual(self.db.context(5), [
            {'role': 'user', 'content': 'Начало истории'},
            {'role': 'assistant', 'content': 'Продолжение'},
        ])
        self.assertFalse(self.db.save_incoming(5, 12, 77, 'Повтор после перезапуска'))

    def test_context_only_current_chat_and_limit(self):
        for n in range(25):
            self.db.save_incoming(1, n + 1, 8, f'Фраза {n}')
        self.db.save_incoming(2, 1, 9, 'Чужой чат')
        result = self.db.context(1)
        self.assertEqual(len(result), 20)
        self.assertEqual(result[0]['content'], 'Фраза 5')
        self.assertNotIn('Чужой чат', str(result))


class ConfigTests(unittest.TestCase):
    def test_missing_fields_are_named_without_values(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'TELEGRAM_BOT_TOKEN'):
                Config.from_env(Path(tmp) / 'missing.env')

    def test_no_key_required_for_local_api_and_env_precedence(self):
        values = {'TELEGRAM_BOT_TOKEN': 'не-настоящий-токен', 'LLM_BASE_URL': 'http://127.0.0.1:11434/v1', 'LLM_MODEL': 'тестовая-модель'}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, values, clear=True):
            file = Path(tmp) / '.env'
            file.write_text('LLM_MODEL=другое-значение\n', encoding='utf8')
            config = Config.from_env(file)
            self.assertEqual(config.llm_model, values['LLM_MODEL'])
            self.assertEqual(config.llm_api_key, '')

    def test_entrypoint_imports_with_real_aiogram(self):
        from tgbot.main import run
        self.assertTrue(callable(run))

    def test_reject_url_with_credentials(self):
        values = {'TELEGRAM_BOT_TOKEN': 'заполнитель', 'LLM_BASE_URL': 'https://user:pass@api.example.com/v1', 'LLM_MODEL': 'тест'}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, values, clear=True):
            with self.assertRaisesRegex(ValueError, 'LLM_BASE_URL'):
                Config.from_env(Path(tmp) / 'missing.env')


class LLMTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_http_contract_with_mock_transport(self):
        seen = []
        synthetic_key = uuid.uuid4().hex

        def handler(req):
            seen.append((str(req.url), req.headers.get('Authorization'), json.loads(req.content)))
            return httpx.Response(200, json={'choices': [{'message': {'content': '  Осмысленный ответ  '}}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            llm = LLMClient(base_url='https://api.example.org/v1', model='model-1', api_key=synthetic_key, timeout=3, max_tokens=100, client=http)
            text = await llm.reply([{'role': 'user', 'content': 'Привет'}])
        self.assertEqual(text, 'Осмысленный ответ')
        self.assertEqual(seen[0][0], 'https://api.example.org/v1/chat/completions')
        self.assertEqual(seen[0][1], 'Bearer ' + synthetic_key)
        self.assertEqual(seen[0][2]['model'], 'model-1')

    async def test_http_failure_categories(self):
        for status, expected in [(401, 'auth'), (403, 'auth'), (429, 'rate_limit'), (503, 'provider'), (400, 'request')]:
            with self.subTest(status=status):
                async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(status))) as http:
                    llm = LLMClient(base_url='https://example.org/v1', model='model', api_key='', timeout=3, max_tokens=100, client=http)
                    with self.assertRaises(LLMError) as err:
                        await llm.reply([{'role': 'user', 'content': 'Тест'}])
                    self.assertEqual(err.exception.code, expected)

    async def test_timeout_and_invalid_response(self):
        def timeout(req):
            raise httpx.ReadTimeout('секретное диагностическое содержимое')
        async with httpx.AsyncClient(transport=httpx.MockTransport(timeout)) as http:
            llm = LLMClient(base_url='https://example.org/v1', model='model', api_key='', timeout=3, max_tokens=100, client=http)
            with self.assertRaises(LLMError) as err:
                await llm.reply([])
            self.assertEqual(err.exception.code, 'timeout')
            self.assertNotIn('секретное', str(err.exception))
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={'choices': []}))) as http:
            llm = LLMClient(base_url='https://example.org/v1', model='model', api_key='', timeout=3, max_tokens=100, client=http)
            with self.assertRaises(LLMError) as err:
                await llm.reply([])
            self.assertEqual(err.exception.code, 'invalid_response')


class DialogueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = Storage(Path(self.tempdir.name) / 'data.sqlite3')
        self.bot = FakeBot()
        self.llm = FakeLLM()

    async def asyncTearDown(self):
        self.db.close()
        self.tempdir.cleanup()

    async def test_full_fake_path_and_context(self):
        await handle_text(FakeMessage(), self.bot, self.db, self.llm, 'Одна роль')
        await handle_text(FakeMessage(message_id=2, text='Продолжай'), self.bot, self.db, self.llm, 'Одна роль')
        self.assertEqual(self.bot.messages, [(17, 'Ответ модели'), (17, 'Ответ модели')])
        self.assertEqual(self.llm.calls[1], [
            {'role': 'system', 'content': 'Одна роль'},
            {'role': 'user', 'content': 'Привет'},
            {'role': 'assistant', 'content': 'Ответ модели'},
            {'role': 'user', 'content': 'Продолжай'},
        ])
        rows = self.db.connection.execute("SELECT telegram_message_id, status FROM messages WHERE role='assistant' ORDER BY id").fetchall()
        self.assertEqual(rows, [(1001, 'sent'), (1002, 'sent')])

    async def test_duplicate_does_not_resend_or_requery(self):
        msg = FakeMessage()
        await handle_text(msg, self.bot, self.db, self.llm, 'Роль')
        await handle_text(msg, self.bot, self.db, self.llm, 'Роль')
        self.assertEqual(len(self.bot.messages), 1)
        self.assertEqual(len(self.llm.calls), 1)

    async def test_llm_error_is_visible_but_not_model_context(self):
        self.llm = FakeLLM(LLMError('auth', 'Ошибка ключа'))
        await handle_text(FakeMessage(), self.bot, self.db, self.llm, 'Роль')
        self.assertEqual(self.bot.messages[0][1], ERROR_TEXT)
        self.assertEqual(len(self.db.context(17)), 1)
        row = self.db.connection.execute("SELECT status, is_error FROM messages WHERE role='assistant'").fetchone()
        self.assertEqual(row, ('sent', 1))

    async def test_telegram_send_exception_is_not_marked_sent(self):
        self.bot = FakeBot(failure=OSError('сетевая ошибка'))
        await handle_text(FakeMessage(), self.bot, self.db, self.llm, 'Роль')
        row = self.db.connection.execute("SELECT status, telegram_message_id FROM messages WHERE role='assistant'").fetchone()
        self.assertEqual(row, ('unknown', None))
        self.assertEqual(len(self.db.context(17)), 1)

    async def test_telegram_explicit_rejection_is_failed(self):
        from aiogram.exceptions import TelegramBadRequest
        from aiogram.methods import SendMessage

        error = TelegramBadRequest(
            method=SendMessage(chat_id=17, text='Тестовое сообщение'),
            message='Telegram отклонил отправку',
        )
        self.bot = FakeBot(failure=error)
        await handle_text(FakeMessage(), self.bot, self.db, self.llm, 'Роль')
        row = self.db.connection.execute(
            "SELECT status, telegram_message_id, error_kind FROM messages WHERE role='assistant'"
        ).fetchone()
        self.assertEqual(row, ('failed', None, 'TelegramBadRequest'))
        self.assertEqual(len(self.db.context(17)), 1)

    async def test_group_and_nontext_ignored(self):
        await handle_text(FakeMessage(kind='group'), self.bot, self.db, self.llm, 'Роль')
        await handle_text(FakeMessage(text=''), self.bot, self.db, self.llm, 'Роль')
        self.assertFalse(self.llm.calls)
        self.assertFalse(self.bot.messages)
