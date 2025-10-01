import os
from openai import OpenAI

class LLMClient:
    def __init__(self,
                 api_key: str | None = None,
                 base_url: str = "https://api.gapgpt.app/v1",
                 model: str = "gpt-4o-mini",
                 timeout: float = 60.0):
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            base_url=base_url,
            timeout=timeout,
        )
        self.model = model

    def generate_text(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp.choices[0].message.content
        return content.strip() if isinstance(content, str) else "".join(map(str, content)).strip()
