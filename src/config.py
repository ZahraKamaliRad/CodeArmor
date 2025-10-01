# src/config.py
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
        (self.outputs / "plan").mkdir(parents=True, exist_ok=True)
        (self.outputs / "code").mkdir(parents=True, exist_ok=True)
        (self.outputs / "prompts").mkdir(parents=True, exist_ok=True)
        return self

    def dataset_run_dir(self, category: str, dataset: str) -> Path:
        base = self.outputs / category
        base.mkdir(parents=True, exist_ok=True)

        run_dir = base / dataset
        counter = 1
        while run_dir.exists():
            counter += 1
            run_dir = base / f"{dataset}{counter}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir


PATHS = Paths().ensure()
