from pathlib import Path
from typing import Iterable

from .base import BaseDataset, Sample, register
from ..utils.io import read_jsonl


@register("humaneval")
class HumanEval(BaseDataset):

    def __iter__(self) -> Iterable[Sample]:

        for row in read_jsonl(Path(self.path)):

            yield Sample(
            task_id=row["task_id"],
            prompt=row["prompt"],
            ref_code=row.get("canonical_solution"),
            language="python",

            metadata={
                "test": row.get("test"),
                "entry_point": row.get("entry_point")
            }
        )