import re

def strip_markdown_fences(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = re.sub(r"^\s*```[\w]*\s*\n?", "", s)
    s = re.sub(r"\n?\s*```\s*$", "", s)
    return s.strip()


def extract_raw_code(resp: str) -> str:
    blocks = re.findall(r"```(?:[a-zA-Z]*)\n(.*?)```", resp, re.DOTALL)
    
    if blocks:
        code = "\n\n".join(blocks).strip()
    else:
        code = strip_markdown_fences(resp)
        
    lines = code.splitlines()
    cleaned_lines = [line for line in lines if not re.match(r'^\s*#.*$', line)]
    
    return "\n".join(cleaned_lines).strip()
###########################################################
# def extract_raw_code(resp: str) -> str:
#     if not resp:
#         return ""

#     fence_match = re.search(r"```(?:[a-zA-Z]*)\n(.*?)```", resp, re.DOTALL)
#     if fence_match:
#         code = fence_match.group(1).strip()
#     else:
#         python_indicators = r'(import |from |def |class |@app\.route|if __name__|return |if |try:|with )'
#         match = re.search(python_indicators + r'.*', resp, re.DOTALL)
#         code = match.group(0).strip() if match else resp.strip()

#     lines = code.splitlines()
#     cleaned_lines = [line for line in lines if not re.match(r'^\s*#', line)]
    
#     return "\n".join(cleaned_lines).strip()


def extract_plan(text: str | None) -> str:
    if not text:
        return ""
    m = re.search(r"###PLAN_START###\s*(.*?)\s*###PLAN_END###", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()

# def extract_code_few_cot(llm_output: str) -> str:
#     match = re.search(
#         r"###SECURE_CODE_START###\s*(.*?)\s*###SECURE_CODE_END###",
#         llm_output,
#         re.DOTALL | re.IGNORECASE
#     )
#     if not match:
#         return ""
#     code = match.group(1).strip()
#     code = re.sub(r"^```[a-zA-Z]*", "", code)
#     code = re.sub(r"```$", "", code)
#     return code.strip()


def extract_code_few_cot(llm_output: str) -> str:
    if not llm_output:
        return ""

    match = re.search(
        r"###SECURE_CODE_START###\s*(.*?)\s*###SECURE_CODE_END###",
        llm_output,
        re.DOTALL | re.IGNORECASE
    )
    
    if match:
        code = match.group(1).strip()
    else:
        fence_match = re.search(r"```(?:[a-zA-Z]*)\n(.*?)```", llm_output, re.DOTALL)
        if fence_match:
            code = fence_match.group(1).strip()
        else:
            python_keywords = r'(import |from |def |class |@app\.|if |try:|with |return )'
            start_match = re.search(python_keywords + r'.*', llm_output, re.DOTALL)
            
            if start_match:
                code = start_match.group(0).strip()
            else:
                return ""

    code = re.sub(r"^```[a-zA-Z]*", "", code)
    code = re.sub(r"```$", "", code)
    
    lines = code.splitlines()
    cleaned_lines = []
    for line in lines:
        if re.match(r'^\s*#', line):
            continue
        if "###SECURE_CODE" in line:
            continue
        cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines).strip()