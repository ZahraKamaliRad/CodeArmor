# Dataset Description

This dataset was created by merging two public sources, **SALLM** and **SecurityEval**, to provide a retrieval‑based benchmark for security‑focused code generation.

The main goal of building this collection is to create a **standard dataset** for applying the **few‑shot prompt engineering** approach on the two source datasets. This helps provide diverse and reliable examples for guiding models to generate code with **reduced vulnerabilities** in LLM outputs.

**The dataset includes 209 samples covering 78 different CWE vulnerability types.**

---
## Dataset Structure

Each sample is stored in JSONL format and contains five fields:

- **id**: The original sample identifier from the source dataset  
- **benchmark**: The source dataset name (SALLM or SecurityEval)  
- **cwe**: The security weakness category extracted from the original identifier  
- **prompt**: The programming task or vulnerability‑related prompt  
- **secure_code**: A vulnerability‑free implementation for each prompt  
  All secure_code samples were checked using the static analysis tool **Bandit**, and only the samples with no detected issues were kept.

Merging these two sources increased the prompt diversity within each CWE category and reduced the number of unique CWEs present in the original datasets.

---

## Deduplication and Cleaning

To reduce redundancy, a semantic deduplication process was applied to the **prompt** field. Embeddings were generated using the **all‑MiniLM‑L6‑v2** model, and cosine similarity was calculated between prompts within each CWE.

Three levels of deduplication were applied:

- **Similarity ≥ 0.98**: Duplicate samples → removed  
- **0.96 to 0.98**: Very similar samples → removed  
- **0.95 to 0.96**: Removed after manual review to avoid losing useful variation  

This process helped maintain semantic diversity within each CWE while removing unnecessary repetitions.

## References

- **SALLM Dataset**  
  GitHub: https://github.com/s2e-lab/SALLM  

- **SecurityEval Dataset**  
  GitHub: https://github.com/s2e-lab/SecurityEval
