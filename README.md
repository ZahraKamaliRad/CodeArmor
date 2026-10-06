# CodeArmor: Security-Aware Prompt Engineering Framework

Step-by-step guide to set up the environment, run experiments, and reproduce the paper's benchmark results.

---

## 1. Environment & Dependencies

- **Python**: 3.10+ recommended
- **Static Analysis Tools**: Bandit and Semgrep (included in `requirements.txt`)

```bash
# 1. Create and activate a virtual environment
python -m venv venv

# Linux / macOS:
source venv/bin/activate
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
```

### Unpack Semgrep Rules
Extract the bundled security rules into `semgrep-rule/`:
```bash
# Linux / macOS:
unzip semgrep-rule.zip -d .

# Windows (PowerShell):
Expand-Archive -Path semgrep-rule.zip -DestinationPath .
```
*(Verify that `semgrep-rule/python/` exists after extraction).*

---

## 2. Configuration Setup

### A. Environment Variables (`.env`)
Create a `.env` file in the `secure_code_generation` root (or adapt from `windows.env` / `linux.env`):

```ini
# Path to extracted Semgrep rules
SEMGREP_RULES_PATH=./semgrep-rule/python

# Embedding model for retrieval/one-shot (local path or HuggingFace identifier)
EMBEDDING_MODEL_PATH=sentence-transformers/all-MiniLM-L6-v2

# Path to retrieval examples dataset
RETRIEVAL_DATASET_PATH=./src/retrieval/data/retrieval_examples.jsonl
```

### B. LLM Profiles (`config.json`)
Copy `config.sample.json` to `config.json`:

```bash
# Linux / macOS:
cp config.sample.json config.json
# Windows:
copy config.sample.json config.json
```

Adjust the profile settings for your target model/endpoint:
```json
{
  "llm": {
    "provider": "api",
    "profiles": {
      "api": {
        "api_url": "https://api.openai.com/v1",
        "api_key_path": "api_key.txt",
        "model": "gpt-4o-mini",
        "timeout": 120,
        "seed": 1234
      },
      "local": {
        "api_url": "http://127.0.0.1:11434/v1",
        "api_key_path": "api_key_local.txt",
        "model": "llama3.1:8b",
        "timeout": 120,
        "seed": 1234
      }
    }
  }
}
```

Create the referenced API key file (e.g., `api_key.txt`) containing your raw API key string without extra spaces.

---

## 3. Running Experiments

Run experiments using `main.py`. Each run automatically generates code, validates syntax, runs Bandit and Semgrep scans, and computes security metrics.

```bash
python main.py --mode <MODE> --dataset <DATASET> --file <FILENAME> --provider <PROVIDER> [OPTIONS]
```

### Parameter Reference

| Parameter | Options | Description |
| :--- | :--- | :--- |
| `--mode` | `planning_zero_shot`, `direct`, `prefix`, `persona`, `zero_shot_cot`, `few_shot_cot`, `one_shot`, `rci`, `self_refine` | Prompt engineering technique |
| `--dataset` | `sallm`, `securityeval` | Benchmark dataset |
| `--file` | `SALLM.jsonl`, `SecurityEval.jsonl` | Dataset file located in `datasets/` |
| `--provider` | `api`, `local` | Active LLM profile in `config.json` |
| `--iterations` | `0`, `1`, `2` | Refinement iterations (`planning_zero_shot`, `rci`, `self_refine`) |
| `--limit` | `<N>` *(Optional)* | Limit to first N samples (useful for quick test runs) |

### Technique Mapping (Paper vs. Code)

| Paper Technique | `--mode` | `--iterations` |
| :--- | :--- | :--- |
| **CodeArmor (Base)** | `planning_zero_shot` | `0` (default) |
| **CodeArmor-1** | `planning_zero_shot` | `1` |
| **CodeArmor-2** | `planning_zero_shot` | `2` |
| Direct (Zero-Shot) | `direct` | - |
| Prefix Priming | `prefix` | - |
| Persona Priming | `persona` | - |
| Zero-shot CoT | `zero_shot_cot` | - |
| Few-shot CoT | `few_shot_cot` | - |
| One-shot (RAG) | `one_shot` | - |
| RCI (1 / 2 iterations) | `rci` | `1` or `2` |
| Self-Refine (1 / 2 iterations) | `self_refine` | `1` or `2` |

### Reproduction Examples

1. **CodeArmor Base on SALLM**:
   ```bash
   python main.py --mode planning_zero_shot --dataset sallm --file SALLM.jsonl --provider api --iterations 0
   ```

2. **CodeArmor-1 on SecurityEval**:
   ```bash
   python main.py --mode planning_zero_shot --dataset securityeval --file SecurityEval.jsonl --provider api --iterations 1
   ```

3. **Direct Baseline on SALLM**:
   ```bash
   python main.py --mode direct --dataset sallm --file SALLM.jsonl --provider api
   ```

4. **Smoke test (5 instances)**:
   ```bash
   python main.py --mode planning_zero_shot --dataset sallm --file SALLM.jsonl --provider api --limit 5
   ```

Outputs, scan logs, and run metrics are saved under:
```
outputs/<dataset>/<sanitized_model_name>/<technique>/
```

---

## 4. Results & Aggregation

### A. Comprehensive Metrics & Visualizations
Consolidate results across all runs into summaries, charts, and Excel workbooks:
```bash
python scripts/results.py --input-dir outputs --output-dir reports
```

### B. Summary Excel Sheet
Generate the multi-technique comparison spreadsheet (`summary.xlsx`):
```bash
python scripts/Summary_details_result.py
```

### C. Interactive HTML Inspection
Generate side-by-side prompt and completion HTML viewers for any run:
```bash
# CodeArmor runs:
python src/reports/planning_html_report.py --i <path_to_output_jsonl>

# Direct / Prefix / Persona:
python src/reports/html_report.py --i <path_to_output_jsonl>

# CoT:
python src/reports/cot_html_report.py --i <path_to_output_jsonl>

# RCI / Self-Refine:
python src/reports/RCI_html_report.py --i <path_to_output_jsonl>
```
