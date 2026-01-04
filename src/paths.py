from dataclasses import dataclass
from pathlib import Path
import os
import re
import json

from src.utils.git_info import get_commit


def _safe_name(s: str) -> str:
    s = (s or "").strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "-", s)
    return s or "unknown"


@dataclass
class Paths:
    root: Path = Path(__file__).resolve().parents[1]

    def __post_init__(self):
        if "DATASETS_DIR" in os.environ:
            self.datasets = Path(os.environ["DATASETS_DIR"])
        elif (self.root / "datasets").exists():
            self.datasets = self.root / "datasets"
        elif (self.root / "src" / "datasets").exists():
            self.datasets = self.root / "src" / "datasets"
        else:
            self.datasets = self.root / "datasets"

        self.outputs = self.root / "outputs"

    def ensure(self):
        self.outputs.mkdir(parents=True, exist_ok=True)
        return self

    def run_dir(self, dataset: str, model_name: str, technique: str) -> Path:
        ds = _safe_name(dataset)
        mdl = _safe_name(model_name)
        tech = _safe_name(technique)

        base = self.outputs / ds / mdl
        base.mkdir(parents=True, exist_ok=True)

        base_tech = base / tech
        if not base_tech.exists():
            out_dir = base_tech
            out_dir.mkdir(parents=True, exist_ok=True)

        else:
            
            pat_paren = re.compile(rf"^{re.escape(tech)}\((\d+)\)$")
            pat_legacy = re.compile(rf"^{re.escape(tech)}(\d+)$")  
            max_n = 1 
            for p in base.iterdir():
                if not p.is_dir():
                    continue
                if p.name == tech:
                    max_n = max(max_n, 1)
                    continue
                m = pat_paren.match(p.name) or pat_legacy.match(p.name)
                if m:
                    max_n = max(max_n, int(m.group(1)))
            out_dir = base / f"{tech}({max_n + 1})"
            out_dir.mkdir(parents=True, exist_ok=False)


        commit = get_commit(self.root)
        meta = {
            "git_commit": commit
        }

        (out_dir / "run_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        return out_dir


PATHS = Paths().ensure()
