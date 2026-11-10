"""Последовательная обработка личного текста — только P1."""

import asyncio
import logging

from aiogram.exceptions import TelegramAPIError, TelegramNetworkError

from .llm import LLMClient, LLMError
from .storage import Storage

LOGGER = logging.getLogger(__name__)
ERROR_TEXT = 'Сейчас не получается получить ответ модели. Попробуй написать позже.'


async def handle_text(message, bot, storage: Storage, llm: LLMClient, character_prompt: str) -> None:
    # Страховка поверх фильтра в точке регистрации обработчика.
    if message.chat.type != 'private' or not message.text or message.from_user is None:
        return
    chat_id = message.chat.id
    incoming_id = message.message_id
    if not storage.save_incoming(chat_id, incoming_id, message.from_user.id, message.text):
        LOGGER.info('Повтор сообщения: chat_id=%s message_id=%s; обработка пропущена', chat_id, incoming_id)
        return

    is_error = False
    try:
        answer = await llm.reply([{'role': 'system', 'content': character_prompt}, *storage.context(chat_id)])
    except LLMError as exc:
        LOGGER.warning('Ошибка LLM: категория=%s, chat_id=%s, модель=%s', exc.code, chat_id, llm.model)
        answer = ERROR_TEXT
        is_error = True
    except Exception:
        # Не выводить сырое исключение: оно может содержать URL, тело или заголовки с ключом.
        LOGGER.error('Неожиданная ошибка LLM: chat_id=%s, модель=%s', chat_id, llm.model)
        answer = ERROR_TEXT
        is_error = True

    # Bot API ограничивает размер одного текстового сообщения.
    answer = answer[:4095] + '…' if len(answer) > 4096 else answer
    record_id = storage.prepare_reply(chat_id, incoming_id, answer, is_error)
    if record_id is None:
        return
    try:
        sent = await bot.send_message(chat_id=chat_id, text=answer)
    except TelegramNetworkError:
        storage.finish_reply(record_id, status='unknown', error_kind='telegram_network')
        LOGGER.warning('Неизвестен результат отправки Telegram: chat_id=%s message_id=%s', chat_id, incoming_id)
    except (asyncio.TimeoutError, OSError):
        storage.finish_reply(record_id, status='unknown', error_kind='telegram_timeout')
        LOGGER.warning('Таймаут отправки Telegram: chat_id=%s message_id=%s', chat_id, incoming_id)
    except TelegramAPIError as exc:
        storage.finish_reply(record_id, status='failed', error_kind=type(exc).__name__)
        LOGGER.warning('Telegram отклонил отправку: chat_id=%s категория=%s', chat_id, type(exc).__name__)
    except Exception:
        storage.finish_reply(record_id, status='unknown', error_kind='telegram_unexpected')
        LOGGER.error('Неожиданная ошибка отправки Telegram: chat_id=%s', chat_id)
    else:
        storage.finish_reply(record_id, status='sent', message_id=sent.message_id)
        LOGGER.info('Ответ отправлен: chat_id=%s telegram_message_id=%s', chat_id, sent.message_id)
