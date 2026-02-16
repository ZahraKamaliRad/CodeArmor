# JSONL → HTML Reports

Choose the appropriate script based on the prompt engineering method used to generate the JSONL file.

## Script Selection

### Direct / Persona / Naive Secure
```bash
python html_report.py --i <path_to_jsonl>
```

### Chain-of-Thought (CoT)
```bash
python cot_html_report.py --i <path_to_jsonl>
```

### Planning-based
```bash
python planning_html_report.py --i <path_to_jsonl>
```

## Output

- By default, the HTML file is generated next to the input JSONL file.
- The output file name matches the input file name with a `.html` extension.
