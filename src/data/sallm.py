# src/data/sallm.py
from pathlib import Path
from typing import Iterable
from .base import BaseDataset, Sample, register
from ..utils.io import read_jsonl

@register("sallm")
class SALLM(BaseDataset):
    
    def __iter__(self) -> Iterable[Sample]:
        for i, row in enumerate(read_jsonl(Path(self.path)), 1):
            tid = row.get("id") or row.get("ID") or f"sallm_{i}"
            prompt = row.get("prompt") or row.get("Prompt") or ""
            ref = row.get("insecure_code") or row.get("Insecure_code")
            lang = (row.get("Language") or row.get("language") or "python").lower()
            yield Sample(task_id=str(tid), prompt=prompt, ref_code=ref, language=lang)
