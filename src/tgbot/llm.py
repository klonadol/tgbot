"""Один настоящий OpenAI-совместимый HTTP chat/completions API."""

import httpx


class LLMError(Exception):
    def __init__(self, code: str, description: str):
        super().__init__(description)
        self.code = code


class LLMClient:
    def __init__(self, *, base_url: str, model: str, api_key: str, timeout: float, max_tokens: int, client: httpx.AsyncClient):
        self.url = base_url.rstrip('/') + '/chat/completions'
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.client = client

    async def reply(self, messages: list[dict[str, str]]) -> str:
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        try:
            response = await self.client.post(
                self.url,
                headers=headers,
                json={'model': self.model, 'messages': messages, 'max_tokens': self.max_tokens},
                timeout=self.timeout,
                follow_redirects=False,
            )
        except httpx.TimeoutException:
            raise LLMError('timeout', 'Истекло время ожидания LLM API') from None
        except httpx.RequestError:
            raise LLMError('network', 'Ошибка соединения с LLM API') from None

        if response.status_code == 401 or response.status_code == 403:
            raise LLMError('auth', 'Провайдер отклонил API-ключ или доступ (401/403)')
        if response.status_code == 429:
            raise LLMError('rate_limit', 'Провайдер ограничил частоту запросов (429)')
        if response.status_code >= 500:
            raise LLMError('provider', 'Ошибка сервера LLM API (5xx)')
        if response.status_code >= 300:
            raise LLMError('request', f'LLM API отклонил запрос (HTTP {response.status_code})')
        try:
            data = response.json()
            text = data['choices'][0]['message']['content']
            if not isinstance(text, str) or not text.strip():
                raise ValueError
            return text.strip()
        except (ValueError, KeyError, IndexError, TypeError):
            raise LLMError('invalid_response', 'LLM API вернул пустой или некорректный ответ') from None
