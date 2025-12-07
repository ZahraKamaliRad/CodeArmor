import os
from openai import OpenAI

from src.utils.tokens import add_token_usage 

import re

def sanitize_model_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.gapgpt.app/v1",
        model: str = "gpt-4o-mini",
        #model: str = "deepseek-chat",
        #model: str = "gpt-5-mini",
        #model: str = "gemma-3-27b-it",
        #model: str = "gemini-2.5-flash",
        #model: str ="gemini-2.0-flash",
        #model: str = "qwen3-235b-a22b",
        timeout: float = 60.0,
    ):
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            base_url=base_url,
            timeout=timeout,
        )
        self.model = model

    def generate_text(self,prompt: str,track_tokens: bool = True,log_file: str | None = None,
        extra_token_meta: dict | None = None, seed: int | None = None) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            seed= 1234
        )
        content = resp.choices[0].message.content
        text = content.strip() if isinstance(content, str) else "".join(map(str, content)).strip()

        if track_tokens and getattr(resp, "usage", None) is not None:
            u = resp.usage
            prompt_tokens = getattr(u, "prompt_tokens", 0) or 0
            completion_tokens = getattr(u, "completion_tokens", 0) or 0
            add_token_usage(
                prompt_tokens,
                completion_tokens,
                log_file=log_file,
                extra=extra_token_meta,
            )

        return text
# deepseek-chat
# gpt-4o-mini