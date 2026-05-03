import re
import time
from openai import (OpenAI,APITimeoutError,APIConnectionError,APIStatusError,RateLimitError,AuthenticationError)
from src.config_loader import load_config, load_api_key
from src.utils.usage_stats import add_token_usage,record_llm_time,increment_api_calls
import random

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
            if not api_key:
                raise ValueError(f"[LLM ERROR] API key file is empty: {api_key_path}")
        else:
            api_key = None

        if api_key is None and active_provider != "local":
            raise RuntimeError(
                f"[LLM ERROR] Missing API key for provider '{self.provider}'."
            )

        client_kwargs = {"timeout": float(timeout)}
        client_kwargs["api_key"] = api_key

        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)

    def _call_with_retry(self, fn, retries=10, base_delay=2):
        for attempt in range(retries + 1):
            try:
                increment_api_calls()
                start_time = time.time()
                resp = fn()
                record_llm_time(time.time() - start_time)
                return resp
            except (APITimeoutError, APIConnectionError, RateLimitError) as e:
                wait = base_delay * (2 ** attempt)
                print(f"[LLM RETRY] {type(e).__name__} - retry {attempt+1}/{retries} in {wait}s")
                #print(e)
                time.sleep(wait)
                continue

            except APIStatusError as e:
                status = e.status_code
                if e.status_code in (500, 502, 503, 504):
                    wait = base_delay * (2 ** attempt)
                    print(f"[LLM RETRY] HTTP {status} - retry {attempt+1}/{retries} in {wait}s")
                    #print(e)
                    time.sleep(wait)
                    continue

                if status == 401 or status == 429:
                    wait = base_delay * (2 ** attempt) + random.uniform(0, 1)                    
                    print(f"[LLM RETRY] {status} (likely rate limit) - retry {attempt+1}/{retries} in {wait}s")
                    #print(e)
                    time.sleep(wait)
                    continue
                raise
            except AuthenticationError:
                raise

        raise RuntimeError("Max retries exceeded.")

    def generate_text(self,prompt: str,track_tokens: bool = True,log_file: str | None = None,
        extra_token_meta: dict | None = None,temperature: float | None = None) -> str:

        temp = temperature if temperature is not None else self.temperature
        is_local = self.provider == "local"
        print(f"[LLM CALL] Sending request to model: {self.model}")

        if is_local:
            print("[LLM CALL] Running in LOCAL mode...")
            local_kwargs = {"model": self.model, "prompt": prompt}

            options = {}
            if temp is not None:
                options["temperature"] = temp
            if self.seed is not None:
                options["seed"] = self.seed
                options["temperature"] = 0
                options["num_predict"] = 8192

            if options:
                local_kwargs["extra_body"] = {"options": options}

            resp = self._call_with_retry(
                lambda: self.client.completions.create(**local_kwargs)
            )

            text = (resp.choices[0].text or "").strip()

        else:
            print("[LLM CALL] Running in API mode...")
            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}]
            }

            if temp is not None:
                kwargs["temperature"] = temp
            if self.seed is not None:
                kwargs["seed"] = self.seed

            resp = self._call_with_retry(
                lambda: self.client.chat.completions.create(**kwargs)
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

        time.sleep(0.5 + random.random() * 0.8)
        return text
