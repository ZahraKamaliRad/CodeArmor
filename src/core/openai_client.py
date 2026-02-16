import re
from openai import OpenAI

from src.config_loader import load_config, load_api_key
from src.utils.tokens import add_token_usage


def sanitize_model_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        root = load_config().get("llm", {})
        active_provider = provider or root.get("provider", "api")

        profiles = root.get("profiles", {})
        cfg = profiles.get(active_provider, root)

        self.provider = active_provider
        self.model = model or cfg.get("model")
        if not self.model:
            raise ValueError("No model specified in config.json for the selected provider")

        seed = cfg.get("seed", None)
        self.seed = int(seed) if seed is not None else None

        timeout = cfg.get("timeout", 60)
        base_url = cfg.get("api_url", None)

        api_key = "local"
        api_key_path = cfg.get("api_key_path")

        if active_provider != "local":
            if not api_key_path:
                raise KeyError("Missing api_key_path for provider=api in config.json")
            api_key = load_api_key(api_key_path)

        client_kwargs = {"api_key": api_key, "timeout": float(timeout)}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)

        print(f"[LLM INIT] Provider: {self.provider} | Model: {self.model}")
        if base_url:
            print(f"[LLM INIT] Base URL: {base_url}")

    def generate_text(self,prompt: str,track_tokens: bool = True,
        log_file: str | None = None,extra_token_meta: dict | None = None) -> str:

        print(f"[LLM CALL] Sending request to model: {self.model}")

        is_local = self.provider == "local"

        if is_local:
            print("[LLM CALL] Running in LOCAL mode...")

            resp = self.client.completions.create(
                model=self.model,
                prompt=prompt,
            )

            text = (resp.choices[0].text or "").strip()

        else:
            print("[LLM CALL] Running in API mode...")

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                seed=self.seed,
            )

            content = resp.choices[0].message.content
            text = (
                content.strip()
                if isinstance(content, str)
                else "".join(map(str, content)).strip()
            )

        print("[LLM CALL] Response received.")

        if track_tokens and getattr(resp, "usage", None) is not None:
            u = resp.usage
            add_token_usage(
                getattr(u, "prompt_tokens", 0) or 0,
                getattr(u, "completion_tokens", 0) or 0,
                log_file=log_file,
                extra=extra_token_meta
            )

        return text