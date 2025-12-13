import re
from openai import OpenAI

from src.config_loader import load_config, load_api_key
from src.utils.tokens import add_token_usage


def sanitize_model_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


class LLMClient:
    def __init__(self):
        cfg = load_config().get("llm", {})
        api_key_path = cfg.get("api_key_path")
        if not api_key_path:
            raise KeyError("Missing 'llm.api_key_path' in config.json")

        api_key = load_api_key(api_key_path)

        self.model = cfg.get("model", "gpt-4o-mini")

        seed = cfg.get("seed", None)
        self.seed = int(seed) if seed is not None else None

        timeout = cfg.get("timeout", 60)
        base_url = cfg.get("api_url", None)

        client_kwargs = {"api_key": api_key, "timeout": float(timeout)}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)

    def generate_text(
        self,
        prompt: str,
        track_tokens: bool = True,
        log_file: str | None = None,
        extra_token_meta: dict | None = None,
    ) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            seed=self.seed,
        )

        content = resp.choices[0].message.content
        text = content.strip() if isinstance(content, str) else "".join(map(str, content)).strip()

        if track_tokens and getattr(resp, "usage", None) is not None:
            u = resp.usage
            add_token_usage(
                getattr(u, "prompt_tokens", 0) or 0,
                getattr(u, "completion_tokens", 0) or 0,
                log_file=log_file,
                extra=extra_token_meta,
            )

        return text
