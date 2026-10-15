"""Shared Ollama/OpenAI calls with a disk cache configured through .env.

Set LLM_PROVIDER=ollama or openai (chatgpt is an alias), optionally LLM_MODEL,
and LLM_CACHE_DIR. Environment variables take precedence over .env values.

    llm = LLMClient()
    text = llm.chat([{"role": "user", "content": "Say hello"}])
    result = llm.chat(messages, response_model=MyPydanticModel)

Cache only successful, validated responses. Keys include provider, model,
messages, schema, temperature, and cache_version; API keys are never cached.
"""

import json
import os
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TypeVar, overload

from dotenv import dotenv_values
from pydantic import BaseModel


ResultModel = TypeVar("ResultModel", bound=BaseModel)


class LLMClient:
    def __init__(
        self, *, provider: str | None = None, model: str | None = None,
        cache_dir: str | Path | None = None, env_path: str | Path = ".env",
    ):
        settings = {**dotenv_values(env_path), **os.environ}
        self.provider = (provider or settings.get("LLM_PROVIDER") or "ollama").strip().lower()
        if self.provider == "chatgpt":
            self.provider = "openai"
        if self.provider not in {"ollama", "openai"}:
            raise ValueError("LLM_PROVIDER must be ollama, openai, or chatgpt")
        default_model = "qwen3:8b" if self.provider == "ollama" else "gpt-4.1-nano"
        self.model = model or settings.get("LLM_MODEL") or default_model
        self.cache_dir = Path(cache_dir or settings.get("LLM_CACHE_DIR") or "data/cache/llm").expanduser()
        self._api_key = settings.get("OPENAI_API_KEY") or settings.get("openai_api_key")
        self._ollama_host = settings.get("OLLAMA_HOST") or "http://localhost:11434"
        self.last_cache_hit = False

    @overload
    def chat(self, messages: list[dict], *, response_model: type[ResultModel],
             temperature: float = 0, cache_version: str = "1") -> ResultModel: ...

    @overload
    def chat(self, messages: list[dict], *, response_model: None = None,
             temperature: float = 0, cache_version: str = "1") -> str: ...

    def chat(self, messages: list[dict], *, response_model=None,
             temperature: float = 0, cache_version: str = "1"):
        """Return text or a Pydantic object, reusing an identical successful call."""
        request = {
            "cache_format": 1, "provider": self.provider, "model": self.model,
            "messages": messages, "temperature": temperature,
            "schema": response_model.model_json_schema() if response_model else None,
            "cache_version": cache_version,
        }
        if self.provider == "ollama":
            request["host"] = self._ollama_host
        serialized = json.dumps(request, sort_keys=True, ensure_ascii=False)
        key = sha256(serialized.encode("utf-8")).hexdigest()
        path = self.cache_dir / "responses" / f"{key}.json"
        self.last_cache_hit = False
        if path.is_file():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                if cached["request"] != request:
                    raise ValueError("Cached request does not match")
                result = self._validate(cached["response"], response_model)
            except (OSError, ValueError, KeyError, TypeError):
                # Recompute corrupt or invalid responses, without caching errors.
                pass
            else:
                self.last_cache_hit = True
                return result

        text = self._request(messages, response_model, temperature)
        result = self._validate(text, response_model)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                    suffix=".tmp", delete=False) as file:
                temporary_path = Path(file.name)
                json.dump({"request": request, "response": text}, file, ensure_ascii=False, indent=2)
                file.write("\n")
            temporary_path.replace(path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return result

    @staticmethod
    def _validate(text, response_model):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("The LLM returned an empty or non-text response")
        return response_model.model_validate_json(text) if response_model else text

    def _request(self, messages, response_model, temperature):
        if self.provider == "ollama":
            from ollama import Client

            arguments = {
                "model": self.model, "messages": messages,
                "options": {"temperature": temperature},
            }
            if response_model:
                arguments["format"] = response_model.model_json_schema()
            response = Client(host=self._ollama_host).chat(**arguments)
            return response.message.content

        from openai import OpenAI

        if not self._api_key or self._api_key == "your-key-here":
            raise ValueError("Set OPENAI_API_KEY in .env to use LLM_PROVIDER=openai")
        # Avoid the Brotli decoder incompatibility seen in the test notebook.
        with OpenAI(api_key=self._api_key, default_headers={"Accept-Encoding": "gzip, deflate"}) as client:
            arguments = {
                "model": self.model, "input": messages,
                "temperature": temperature, "store": False,
            }
            if response_model:
                response = client.responses.parse(**arguments, text_format=response_model)
                if response.status != "completed" or response.output_parsed is None:
                    raise ValueError("OpenAI returned an incomplete response or refusal")
                return response.output_parsed.model_dump_json()
            response = client.responses.create(**arguments)
            if response.status != "completed":
                raise ValueError("OpenAI returned an incomplete response")
            return response.output_text
