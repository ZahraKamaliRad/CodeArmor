from dataclasses import dataclass
from pathlib import Path
import os

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

    def dataset_run_dir(self, dataset: str) -> Path:
        base = self.outputs
        run_dir = base / dataset
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=True)
            return run_dir
        i = 2
        while True:
            cand = base / f"{dataset}({i})"
            if not cand.exists():
                cand.mkdir(parents=True, exist_ok=True)
                return cand
            i += 1

PATHS = Paths().ensure()
