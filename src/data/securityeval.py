# src/data/securityeval.py
from pathlib import Path
from typing import Iterable
from .base import BaseDataset, Sample, register
from ..utils.io import read_jsonl

@register("securityeval")
class SecurityEval(BaseDataset):
   
    def __iter__(self) -> Iterable[Sample]:
        for row in read_jsonl(Path(self.path)):
            task = row.get("ID") or row.get("task") or row.get("id")
            prompt = row.get("Prompt") or row.get("prompt") or row.get("problem") or ""
            ref = row.get("Insecure_code") or row.get("file") or row.get("solution")
            lang = (row.get("Language") or row.get("language") or "python").lower()
            yield Sample(task_id=str(task), prompt=prompt, ref_code=ref, language=lang)
