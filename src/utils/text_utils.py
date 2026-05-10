import re

def strip_markdown_fences(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = re.sub(r"^\s*```[\w]*\s*\n?", "", s)
    s = re.sub(r"\n?\s*```\s*$", "", s)
    return s.strip()

def extract_raw_code(resp: str) -> str:
    if not resp:
        return ""

    blocks = re.findall(r"```(?:[\w]+)?\s*(.*?)```", resp, re.DOTALL)
    if blocks:
        code = "\n\n".join(blocks).strip()
    else:
        code = strip_markdown_fences(resp).strip()

    lines = code.splitlines()
    cleaned_lines = [line for line in lines if not re.match(r'^\s*#.*$', line)]
    return "\n".join(cleaned_lines).strip()


def extract_plan(plan_resp: str) -> dict:
    if not plan_resp:
        return {"plan": ""}

    text = re.sub(r"```[\s\S]*?```", "", plan_resp).strip()

    match_plan = re.search(r"(?:Plan:|### Plan:)\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    plan = match_plan.group(1).strip() if match_plan else text  
    return {"plan": plan} 



def extract_code_few_cot(llm_output: str) -> str:
    match = re.search(
        r"###SECURE_CODE_START###\s*```?[a-zA-Z]*\n?(.*?)\n?```?\s*###SECURE_CODE_END###",
        llm_output,re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    code = match.group(1).strip()
    if not code:
        return ""

    return code
