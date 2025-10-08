from pathlib import Path
import subprocess

BASE = Path(__file__).resolve().parent

inputs = [
    BASE / "outputs/code/securityeval_direct/securityeval_direct.jsonl",
    BASE / "outputs/code/securityeval_pe03a2/securityeval_pe03a.jsonl",
    BASE / "outputs/code/securityeval_rci_iter1/securityeval_rci_iter1.jsonl",
    #BASE / "outputs/code/securityeval_planning/securityeval_planning.jsonl",
    BASE / "outputs/code/securityeval_planning6/securityeval_planning.jsonl",
    #BASE / "outputs/code/securityeval_planning5/securityeval_planning.jsonl",
    BASE / "outputs/code/sallm_direct/sallm_direct.jsonl",
    BASE / "outputs/code/sallm_pe03a/sallm_pe03a.jsonl",
    BASE / "outputs/code/sallm_rci_iter1/sallm_rci_iter1.jsonl",
    #BASE / "outputs/code/sallm_planning/sallm_planning.jsonl",
    BASE / "outputs/code/sallm_planning3/sallm_planning.jsonl",
    #BASE / "outputs/code/sallm_planning2/sallm_planning.jsonl",
]
output_csv = BASE / "results_all.csv"
cmd = ["python", "src/core/metrics.py", *map(str, inputs), "--csv", str(output_csv)]
print("Running metrics command:\n", " ".join(map(str, cmd)))
subprocess.run(cmd, check=True)
