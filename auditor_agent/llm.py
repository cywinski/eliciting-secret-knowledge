# ABOUTME: Thin OpenRouter chat-completions client for the auditor model, with retries and raw I/O
# ABOUTME: capture so every auditor request/response can be logged verbatim.
import os
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI


class OpenRouterClient:
    """Auditor LLM client (OpenRouter, OpenAI-compatible tool calling)."""

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = None,
        max_tokens: int = 2048,
        http_referer: str = "https://localhost",
        x_title: str = "auditor-agent",
        max_retries: int = 5,
    ):
        """Create the client.

        Args:
            model: OpenRouter model slug.
            temperature: Sampling temperature, or None to leave it out of the request and
                use the model's own default.
            max_tokens: Max completion tokens per auditor call.
            http_referer: OpenRouter `HTTP-Referer` header.
            x_title: OpenRouter `X-Title` header.
            max_retries: Retries on transient API errors.
        """
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_headers = {"HTTP-Referer": http_referer, "X-Title": x_title}
        self.max_retries = max_retries

    def chat(
        self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Call the auditor model once and return the raw response as a dict."""
        request = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    extra_headers=self.extra_headers, **request
                )
                return response.model_dump()
            except Exception as error:  # noqa: BLE001 - retry transient API failures only
                last_error = error
                status = getattr(error, "status_code", None)
                if status is not None and status < 500 and status != 429:
                    raise
                sleep_s = 2**attempt
                print(f"  [auditor api] attempt {attempt + 1} failed ({error}); retrying in {sleep_s}s")
                time.sleep(sleep_s)
        raise RuntimeError(f"OpenRouter call failed after {self.max_retries} attempts: {last_error}")
