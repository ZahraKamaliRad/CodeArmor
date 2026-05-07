import re
def strip_markdown_fences(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = re.sub(r"^\s*```[a-zA-Z0-9]*\s*\n", "", s)
    s = re.sub(r"\n\s*```\s*$", "", s)
    return s.strip()

def extract_plan(plan_resp: str) -> dict:
    if not plan_resp:
        return {"plan": ""}

    text = re.sub(r"```[\s\S]*?```", "", plan_resp).strip()

    match_plan = re.search(r"(?:Plan:|### Plan:)\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    plan = match_plan.group(1).strip() if match_plan else text  
    return {"plan": plan}

def extract_raw_code(resp: str) -> str:
    if not resp:
        return ""
    m = re.search(r"```(?:[\w]+)?\s*(.*?)```", resp, re.DOTALL)
    code = m.group(1).strip() if m else strip_markdown_fences(resp).strip()
    lines = code.splitlines()
    cleaned_lines = [line for line in lines if not line.strip().startswith("#")]
    return "\n".join(cleaned_lines).strip()



def extract_code_cot(resp: str) -> str:
    if not isinstance(resp, str):
        return ""

    fenced = re.findall(
        r"```(?:[a-zA-Z0-9_+-]*)\s*(.*?)```",resp,flags=re.DOTALL)
    if fenced:
        return fenced[-1].strip()

    fallback = re.split(
        r"(Final code:|Final implementation:|Complete code:)",resp,flags=re.IGNORECASE)
    if len(fallback) > 2:
        return fallback[-1].strip()

    legacy = re.split(
        r"Therefore,.*implementation is:",resp,flags=re.IGNORECASE | re.DOTALL)
    if len(legacy) > 1:
        return legacy[-1].strip()

    code_like = []
    for line in resp.splitlines():
        if (line.strip().startswith(("def ", "class ", "import ", "#include", "{", "}", "for ", "while ", "if "))) or \
           (";" in line) or ("(" in line and ")" in line):
            code_like.append(line)

    if code_like:
        return "\n".join(code_like).strip()

    return ""
