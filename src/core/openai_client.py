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

        self.temperature = cfg.get("temperature", None)

        seed = cfg.get("seed", None)
        self.seed = int(seed) if seed is not None else None

        timeout = cfg.get("timeout", 120)
        base_url = cfg.get("api_url", None)

        api_key_path = cfg.get("api_key_path")

        if api_key_path:
            api_key = load_api_key(api_key_path)
        else:
            api_key = None

        client_kwargs = {"timeout": float(timeout)}
        
        if api_key is None:
            api_key = "dummy"
        client_kwargs["api_key"] = api_key

        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

        print(f"[LLM INIT] Provider: {self.provider} | Model: {self.model}")
        if base_url:
            print(f"[LLM INIT] Base URL: {base_url}")

    def generate_text(self, prompt: str, track_tokens: bool = True, log_file: str | None = None, extra_token_meta: dict | None = None, temperature: float | None = None) -> str:

        temp = temperature if temperature is not None else self.temperature

        print(f"[LLM CALL] Sending request to model: {self.model}")

        is_local = self.provider == "local"

        if is_local:
            print("[LLM CALL] Running in LOCAL mode...")

            local_kwargs = {"model": self.model, "prompt": prompt}

            options = {}
            if temp is not None:
                options["temperature"] = temp
            if self.seed is not None:
                options["seed"] = self.seed
                options["temperature"] = 0  # Required for deterministic output
                options["num_predict"] = 8192  # Required for seed to work
            
            if options:
                local_kwargs["extra_body"] = {"options": options}

            resp = self.client.completions.create(**local_kwargs)

            text = (resp.choices[0].text or "").strip()
        else:
            print("[LLM CALL] Running in API mode...")

            kwargs = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
            
            if temp is not None:
                kwargs["temperature"] = temp

            if self.seed is not None:
                kwargs["seed"] = self.seed
                kwargs["max_tokens"] = 8192

            resp = self.client.chat.completions.create(**kwargs)
            
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