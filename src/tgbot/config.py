"""Локальные настройки P1. Значения секретов не выводятся в исключениях."""

from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    telegram_token: str
    llm_base_url: str
    llm_model: str
    llm_api_key: str
    character_prompt: str
    database_path: Path
    llm_timeout_seconds: float
    llm_max_tokens: int

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Config":
        # Переменные окружения имеют приоритет над локальным .env.
        load_dotenv(dotenv_path=env_file or Path('.env'), override=False)
        required = {
            'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN', '').strip(),
            'LLM_BASE_URL': os.getenv('LLM_BASE_URL', '').strip(),
            'LLM_MODEL': os.getenv('LLM_MODEL', '').strip(),
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise ValueError('Не заданы обязательные настройки: ' + ', '.join(missing))

        url = urlsplit(required['LLM_BASE_URL'])
        if (url.scheme not in ('http', 'https') or not url.netloc
                or url.username or url.password or url.query or url.fragment
                or url.path.rstrip('/').endswith('/chat/completions')):
            raise ValueError('LLM_BASE_URL должен быть адресом API http(s) без ключей, параметров и /chat/completions')

        try:
            timeout = float(os.getenv('LLM_TIMEOUT_SECONDS', '30'))
            tokens = int(os.getenv('LLM_MAX_TOKENS', '512'))
            if not 1 <= timeout <= 300 or not 1 <= tokens <= 4096:
                raise ValueError
        except ValueError as exc:
            raise ValueError('LLM_TIMEOUT_SECONDS (1–300) или LLM_MAX_TOKENS (1–4096) некорректны') from exc

        character = os.getenv('CHARACTER_PROMPT', 'Ты дружелюбный вымышленный персонаж. Отвечай естественно, на русском языке.').strip()
        if not character:
            raise ValueError('CHARACTER_PROMPT не может быть пустым')
        db_value = os.getenv('DATABASE_PATH', 'data/tgbot.sqlite3').strip()
        if not db_value:
            raise ValueError('DATABASE_PATH не может быть пустым')
        return cls(
            telegram_token=required['TELEGRAM_BOT_TOKEN'],
            llm_base_url=required['LLM_BASE_URL'].rstrip('/'),
            llm_model=required['LLM_MODEL'],
            llm_api_key=os.getenv('LLM_API_KEY', '').strip(),
            character_prompt=character,
            database_path=Path(db_value),
            llm_timeout_seconds=timeout,
            llm_max_tokens=tokens,
        )
