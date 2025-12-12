# Dataset

In this work, we use two datasets, **SALLM** and **SecurityEval**. In the following sections, we describe how each dataset was constructed, including the data collection process and the creation of their prompts.

## 1. SALLM

The SALLM dataset is a manually constructed collection of security-oriented prompts, created using real Python code examples gathered from various security-relevant sources. A significant portion of these examples was extracted from developer discussions on **StackOverflow**, where accepted answers often include unsafe or vulnerability-prone coding patterns. The dataset also incorporates Python-based examples from the official **CWE** repository. Since many CWE entries do not include Python implementations, additional vulnerable Python snippets were manually created based on the descriptions of the weaknesses. Furthermore, vulnerable examples provided in **CodeQL** and **SonarQube**, both of which include security-focused code samples, were used to expand the dataset.

After collecting these examples, they were manually transformed into structured prompts. These prompts include elements such as function signatures, docstrings, and import statements, while the core implementation was intentionally removed so that a code generation model must complete it. Each prompt was assigned a unique identifier mapped to its corresponding CWE entry, and an insecure reference implementation was included to illustrate how an incorrect completion could lead to a vulnerability. The dataset contains **100 programming prompts**.

## 2. SecurityEval

The SecurityEval dataset was created to evaluate code generation models from a software-security perspective. It was built by collecting insecure Python code examples from a variety of well-established security-related sources. These sources include vulnerable coding patterns from **CodeQL**, software-weakness examples listed in the official **CWE** repository, and non-compliant security examples found in **SonarQube** documentation. Additionally, several insecure Python scenarios were adapted from the work of **Pearce et al.**, which analyzed how code generation tools may produce vulnerable outputs. For CWE entries that lacked Python implementations, new Python examples were manually created based on the descriptions of the weaknesses.

The collected examples were then converted into structured prompts suitable for evaluating code generation models. These prompts consist of function signatures, docstrings, and import statements, while the main implementation was intentionally omitted. Each prompt was assigned a unique identifier encoding both its CWE category and original source. An insecure reference implementation was also added to demonstrate how an incorrect completion could introduce a vulnerability. The dataset contains **121 programming prompts**.

## References

- **SALLM Dataset**  
  GitHub: https://github.com/s2e-lab/SALLM  

- **SecurityEval Dataset**  
  GitHub: https://github.com/s2e-lab/SecurityEval
