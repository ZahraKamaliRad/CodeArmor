from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Dict, Any, Optional, List

@dataclass
class Sample:
    task_id: str
    prompt: str
    ref_code: Optional[str] = None
    language: str = "python"
    metadata: Optional[Dict[str, Any]] = None
class BaseDataset:
    name: str
    def __init__(self, path: Path):
        self.path = path
    def __iter__(self) -> Iterable[Sample]:
        raise NotImplementedError

_DATASETS: Dict[str, type] = {}
def register(name: str):
    def deco(cls):
        _DATASETS[name] = cls
        cls.name = name
        return cls
    return deco

def get_dataset(name: str, path: Path) -> BaseDataset:
    if name not in _DATASETS:
        raise KeyError(f"Unknown dataset: {name}. Known: {list(_DATASETS)}")
    return _DATASETS[name](path)

def to_direct_records(samples: Iterable[Sample]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for s in samples:
        meta = s.metadata or {}

        out.append({
            "ID": s.task_id,
            "Prompt": s.prompt,
            "framework": None,
            "language": s.language,
            "test_case": meta.get("test"),
            "entry_point": meta.get("entry_point"),
        })

    return out