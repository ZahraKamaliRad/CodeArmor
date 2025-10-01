from pathlib import Path
from typing import Iterable
from .base import BaseDataset, Sample, register
from ..utils.io import read_jsonl

@register("llmseceval")
class LLMSecEval(BaseDataset):
    def __iter__(self) -> Iterable[Sample]:
        for i, row in enumerate(read_jsonl(Path(self.path)), 1):
            tid = row.get("Prompt ID") or row.get("ID") or f"llmseceval_{i}"
            lang_raw = (row.get("Language") or row.get("language") or "python").strip()
            lang = "python" if lang_raw.lower().startswith("py") else "c" if lang_raw.lower().startswith("c") else lang_raw.lower()
            prompt_text = row.get("Manually-fixed NL Prompt") or row.get("LLM-generated NL Prompt") or ""
            if "<language>" in (prompt_text or ""):
                replace_with = "Python" if lang == "python" else "C" if lang == "c" else lang_raw
                prompt_text = prompt_text.replace("<language>", replace_with)
            yield Sample(task_id=str(tid), prompt=prompt_text, ref_code=None, language=lang)
