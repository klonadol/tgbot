"""Запуск P1: один процесс, long polling, личные текстовые сообщения."""

import asyncio
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatType
import httpx

from .config import Config
from .llm import LLMClient
from .storage import Storage
from .telegram import handle_text


async def run() -> None:
    config = Config.from_env()
    storage = Storage(config.database_path)
    try:
        async with httpx.AsyncClient() as http_client, Bot(token=config.telegram_token) as bot:
            llm = LLMClient(
                base_url=config.llm_base_url,
                model=config.llm_model,
                api_key=config.llm_api_key,
                timeout=config.llm_timeout_seconds,
                max_tokens=config.llm_max_tokens,
                client=http_client,
            )
            dispatcher = Dispatcher()

            @dispatcher.message(F.chat.type == ChatType.PRIVATE, F.text)
            async def on_private_text(message):
                await handle_text(message, bot, storage, llm, config.character_prompt)

            # Получать отложенные update после остановки; webhook с polling несовместим.
            await bot.delete_webhook(drop_pending_updates=False)
            logging.info('Бот запущен, режим long polling')
            await dispatcher.start_polling(
                bot,
                allowed_updates=['message'],
                handle_as_tasks=False,
                close_bot_session=False,
            )
    finally:
        storage.close()
        logging.info('База данных закрыта')


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logging.info('Остановка по запросу пользователя')
    except (ValueError, RuntimeError) as exc:
        # В этих сообщениях нет конфигурационных значений или ключей.
        logging.error('%s', exc)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
