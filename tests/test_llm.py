"""Run with python -m unittest discover -s tests -v; no live LLM calls."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import BaseModel, ValidationError

from utils.llm import LLMClient
from utils.ministry_prediction import predict_ministry


class Answer(BaseModel):
    answer: int


class LLMTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.env = self.root / ".env"
        self.env.write_text(
            f"LLM_PROVIDER=ollama\nLLM_CACHE_DIR={self.root.as_posix()}/cache\n",
            encoding="utf-8",
        )
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.messages = [{"role": "user", "content": "What is 2 + 2?"}]

    def client(self, **kwargs):
        return LLMClient(env_path=self.env, **kwargs)

    def test_env_selection_and_overrides(self):
        self.assertEqual(self.client().model, "qwen3:8b")
        with self.env.open("a", encoding="utf-8") as file:
            file.write("LLM_PROVIDER=chatgpt\nLLM_MODEL=gpt-4.1-nano\n")
        client = self.client()
        self.assertEqual(client.provider, "openai")
        self.assertEqual(client.model, "gpt-4.1-nano")
        self.assertEqual(client.cache_dir, self.root / "cache")
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama", "LLM_MODEL": "local-model"}):
            self.assertEqual(self.client().model, "local-model")
            self.assertEqual(self.client().provider, "ollama")
        with self.assertRaises(ValueError):
            self.client(provider="unknown")

    def test_text_and_structured_cache_persist_between_clients(self):
        with patch.object(LLMClient, "_request", return_value="Four") as request:
            self.assertEqual(self.client().chat(self.messages), "Four")
            client = self.client()
            self.assertEqual(client.chat(self.messages), "Four")
            self.assertTrue(client.last_cache_hit)
            self.assertEqual(request.call_count, 1)
        with patch.object(LLMClient, "_request", return_value='{"answer":4}') as request:
            self.assertEqual(self.client().chat(self.messages, response_model=Answer).answer, 4)
            self.assertEqual(self.client().chat(self.messages, response_model=Answer).answer, 4)
            self.assertEqual(request.call_count, 1)

    def test_cache_separates_request_settings(self):
        with patch.object(LLMClient, "_request", return_value='{"answer":4}') as request:
            client = self.client()
            client.chat(self.messages)
            client.chat(self.messages, temperature=0.5)
            client.chat(self.messages, cache_version="2")
            client.chat(self.messages, response_model=Answer)
            client.chat([{"role": "user", "content": "Changed prompt"}])
            self.client(model="another-model").chat(self.messages)
            self.client(provider="openai").chat(self.messages)
            self.assertEqual(request.call_count, 7)

    def test_errors_and_invalid_responses_are_not_cached(self):
        client = self.client()
        with patch.object(LLMClient, "_request", side_effect=ConnectionError("offline")):
            with self.assertRaises(ConnectionError):
                client.chat(self.messages)
        with patch.object(LLMClient, "_request", return_value='{"answer":"invalid"}'):
            with self.assertRaises(ValidationError):
                client.chat(self.messages, response_model=Answer)
        self.assertEqual(list(client.cache_dir.rglob("*.json")), [])

    def test_corrupt_cache_is_recomputed(self):
        client = self.client()
        with patch.object(LLMClient, "_request", return_value="Four"):
            client.chat(self.messages)
        next(client.cache_dir.rglob("*.json")).write_text("{", encoding="utf-8")
        with patch.object(LLMClient, "_request", return_value="Four again") as request:
            self.assertEqual(client.chat(self.messages), "Four again")
            self.assertFalse(client.last_cache_hit)
            request.assert_called_once()

    def test_ollama_adapter(self):
        with patch("ollama.Client") as factory:
            factory.return_value.chat.return_value = SimpleNamespace(message=SimpleNamespace(content='{"answer":4}'))
            result = self.client().chat(self.messages, response_model=Answer)
            self.assertEqual(result.answer, 4)
            arguments = factory.return_value.chat.call_args.kwargs
            self.assertEqual(arguments["format"], Answer.model_json_schema())
            self.assertEqual(arguments["options"], {"temperature": 0})

    def test_openai_adapter_and_no_cached_credentials(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-secret-key"}), patch("openai.OpenAI") as factory:
            backend = factory.return_value.__enter__.return_value
            backend.responses.parse.return_value = SimpleNamespace(status="completed", output_parsed=Answer(answer=4))
            backend.responses.create.return_value = SimpleNamespace(status="completed", output_text="Four")
            client = self.client(provider="chatgpt")
            self.assertEqual(client.chat(self.messages, response_model=Answer).answer, 4)
            self.assertEqual(client.chat(self.messages), "Four")
            self.assertEqual(factory.call_args.kwargs["default_headers"], {"Accept-Encoding": "gzip, deflate"})
            self.assertFalse(backend.responses.create.call_args.kwargs["store"])
            self.assertIs(backend.responses.parse.call_args.kwargs["text_format"], Answer)
            for path in client.cache_dir.rglob("*.json"):
                self.assertNotIn("test-secret-key", path.read_text(encoding="utf-8"))

    def test_missing_key_and_refusals_are_not_cached(self):
        client = self.client(provider="openai")
        with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
            client.chat(self.messages)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), patch("openai.OpenAI") as factory:
            factory.return_value.__enter__.return_value.responses.parse.return_value = SimpleNamespace(
                status="completed", output_parsed=None,
            )
            with self.assertRaises(ValueError):
                self.client(provider="openai").chat(self.messages, response_model=Answer)
        self.assertEqual(list(client.cache_dir.rglob("*.json")), [])

    def test_ministry_extraction_uses_shared_cache(self):
        with patch.object(LLMClient, "_request", return_value='{"ministry":null}') as request:
            client = self.client()
            page = {"page_url": "https://parish.example/contact", "site_url": "https://parish.example"}
            self.assertIsNone(predict_ministry(page, llm=client).ministry)
            self.assertIsNone(predict_ministry(page, llm=client).ministry)
            self.assertEqual(request.call_count, 1)


if __name__ == "__main__":
    unittest.main()
