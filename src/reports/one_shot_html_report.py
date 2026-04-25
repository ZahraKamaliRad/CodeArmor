import json
import html
from pathlib import Path
from datetime import datetime
import argparse

SEV_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


def esc(s):
    return html.escape("" if s is None else str(s))


def load_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def issues_to_list(issues_field):
    if isinstance(issues_field, list):
        return issues_field
    if issues_field is None:
        return []
    if isinstance(issues_field, int):
        return [{}] * issues_field
    return []


def normalize_issues(issues, tool_name: str):
    out = []
    for iss in issues or []:
        tool = iss.get("tool") or tool_name

        severity = (iss.get("severity") or "").upper()
        if severity not in SEV_RANK:
            severity = "LOW"

        rule = (iss.get("rule_id") or iss.get("rule_key") or "") or ""

        line = iss.get("line") or iss.get("start_line") or iss.get("line_number") or 0
        msg = iss.get("message") or iss.get("issue_text") or iss.get("extra", "") or ""

        out.append(
            {
                "tool": tool,
                "severity": severity,
                "rule": str(rule),
                "line": int(line) if str(line).isdigit() else line,
                "message": str(msg),
            }
        )
    return out


def build_sev_options(issues):
    sevs = sorted({(i.get("severity") or "LOW").upper() for i in issues}, key=lambda s: SEV_RANK.get(s, 0))
    return "\n".join([f'<option value="{esc(s)}">{esc(s)}</option>' for s in sevs])


def build_tool_options(issues):
    tools = sorted({(i.get("tool") or "").lower() for i in issues if i.get("tool")})
    return "\n".join([f'<option value="{esc(t)}">{esc(t)}</option>' for t in tools])


def render_no_issues(tool_name: str) -> str:
    return f'''
<div class="chip" style="display:block;padding:12px 14px;border-radius:12px;">
  {esc(tool_name)}: no vulnerabilities reported.
</div>
'''.strip()


CSS = r"""
:root{--bg:#0b0e14;--fg:#e6e8ef;--muted:#a6adbb;--card:#121826;--line:#242b3a;--chip:#1b2436}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif;overflow-x:hidden}
a{color:inherit;text-decoration:none}
a:hover{text-decoration:underline}
.container{max-width:1200px;margin:24px auto;padding:0 16px}
.header{display:flex;gap:12px;align-items:flex-end;justify-content:space-between;margin-bottom:18px}
.h1{font-size:22px;font-weight:700}
.meta{color:var(--muted);font-size:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:14px;margin:12px 0;min-width:0}
.grid{display:grid;grid-template-columns:1fr;gap:12px}
@media(min-width:980px){.grid{grid-template-columns:1fr 1fr}}
.subhead{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;gap:10px;min-width:0}
.subttl{font-weight:700;min-width:0}
.chip{background:var(--chip);border:1px solid var(--line);padding:6px 10px;border-radius:999px;color:var(--muted);font-size:12px;display:inline-flex;gap:6px;align-items:center}
.controls{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0;align-items:center}
input,select{background:var(--bg);border:1px solid var(--line);color:var(--fg);border-radius:10px;padding:8px 10px}
input{min-width:280px}
.small{min-width:210px}

table{width:100%;border-collapse:separate;border-spacing:0}
th,td{padding:10px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600;text-align:left;font-size:12px}
td{font-size:13px}
.badge{padding:2px 8px;border-radius:999px;border:1px solid var(--line);background:var(--chip);color:var(--muted);font-size:11px;display:inline-block}
pre{margin:0;color:var(--fg)}

details{border:1px solid var(--line);border-radius:14px;padding:10px 12px;background:rgba(0,0,0,0.12);min-width:0}
summary{cursor:pointer;color:var(--muted);font-weight:600}
.kv{display:flex;gap:10px;flex-wrap:wrap}
hr{border:0;border-top:1px solid var(--line);margin:12px 0}
.hint{color:var(--muted);font-size:12px}

.toggle-row{display:flex;align-items:center;gap:12px;margin-left:8px}
.toggle-label{font-size:13px;color:var(--muted)}
.switch{position:relative;display:inline-block;width:46px;height:24px}
.switch input{opacity:0;width:0;height:0}
.slider{position:absolute;cursor:pointer;inset:0;background:var(--line);border-radius:999px;transition:0.2s}
.slider:before{position:absolute;content:"";height:18px;width:18px;left:3px;bottom:3px;background:white;border-radius:50%;transition:0.2s}
.switch input:checked + .slider{background:#4ade80}
.switch input:checked + .slider:before{transform:translateX(22px)}
.state-pill{padding:4px 10px;border-radius:999px;border:1px solid var(--line);background:var(--chip);font-size:12px;color:var(--muted)}
.state-on{color:#4ade80}
.state-off{color:var(--muted)}

.table-wrap{
  width:100%;
  max-width:100%;
  overflow-x:auto;
  overflow-y:hidden;
  border-radius:12px;
  border:1px solid var(--line);
}

.table-wrap table{
  width:max-content;
  min-width:860px;
  border-collapse:separate;
  border-spacing:0;
  table-layout:auto;
}

.t-msg{
  white-space:pre-wrap;
  overflow-wrap:normal;
  word-break:normal;
  line-height:1.35;
}

.t-one{
  white-space:nowrap;
  overflow:hidden;
  text-overflow:ellipsis;
  display:block;
  max-width:360px;
}

.t-line{white-space:nowrap}


th.col-tool, td.col-tool{width:90px}
th.col-sev,  td.col-sev{width:95px}
th.col-rule, td.col-rule{width:260px}
th.col-line, td.col-line{width:70px}
th.col-msg,  td.col-msg{width:520px}


.code-wrap{
  border:0;
  border-radius:0;
  overflow:visible;
  max-width:100%;
  background:transparent;
  margin-top:10px;
}

.code-pre{
  margin:0;
  padding:0;
  white-space:pre-wrap;
  overflow-wrap:normal;
  word-break:normal;
  line-height:1.2;
  font-family:inherit;
  font-size:12px;
  counter-reset: ln;
}

.code-line{
  display:block;
  padding-left:38px;
  position:relative;
}
.code-line::before{
  counter-increment: ln;
  content: counter(ln);
  position:absolute;
  left:0;
  width:28px;
  text-align:right;
  color:var(--muted);
}
"""


TASK_JS = r"""
function norm(s){return (s||"").toString().toLowerCase();}

function setToggleState(){
  const cb=document.getElementById("only_vuln");
  const pill=document.getElementById("only_vuln_state");
  if(!cb || !pill) return;
  if(cb.checked){
    pill.textContent="ON";
    pill.classList.add("state-on");
    pill.classList.remove("state-off");
  }else{
    pill.textContent="OFF";
    pill.classList.add("state-off");
    pill.classList.remove("state-on");
  }
}

function updateShown(){
  const items = Array.from(document.querySelectorAll("[data-item]"));
  const q = norm(document.getElementById("q")?.value);
  const lang = norm(document.getElementById("lang")?.value);
  const onlyVuln = document.getElementById("only_vuln")?.checked;

  let shown=0;
  for(const it of items){
    const text = norm(it.innerText);
    const itLang = norm(it.dataset.lang);
    const b = parseInt(it.dataset.bandit||"0",10);
    const s = parseInt(it.dataset.semgrep||"0",10);

    const okQ = !q || text.includes(q);
    const okL = (lang==="all"||!lang) ? true : itLang===lang;
    const okV = !onlyVuln ? true : (b+s)>0;

    const show = okQ && okL && okV;
    it.style.display = show ? "" : "none";
    if(show) shown++;
  }
  const el=document.getElementById("shown");
  if(el) el.textContent=shown.toString();
}

function applyToolFilter(itemId, tool){
  const root = document.getElementById(itemId);
  if(!root) return;

  const q = norm(root.querySelector(`[data-${tool}-q]`)?.value);
  const sev = norm(root.querySelector(`[data-${tool}-sev]`)?.value);
  const tval = norm(root.querySelector(`[data-${tool}-tool]`)?.value);

  const rows = Array.from(root.querySelectorAll(`table[data-${tool}-table] tbody tr`));
  let v=0;
  for(const tr of rows){
    const ttool = norm(tr.dataset.tool);
    const tsev = norm(tr.dataset.severity);
    const text = norm(tr.innerText);

    const okQ = !q || text.includes(q);
    const okS = (sev==="all"||!sev) ? true : tsev===sev;
    const okT = (tval==="all"||!tval) ? true : ttool===tval;

    const show = okQ && okS && okT;
    tr.style.display = show ? "" : "none";
    if(show) v++;
  }
  const cnt = root.querySelector(`[data-${tool}-count]`);
  if(cnt) cnt.textContent=v.toString();
}

function hookTool(itemId, tool){
  const root = document.getElementById(itemId);
  if(!root) return;
  const q = root.querySelector(`[data-${tool}-q]`);
  const s = root.querySelector(`[data-${tool}-sev]`);
  const t = root.querySelector(`[data-${tool}-tool]`);
  if(q) q.addEventListener("input", ()=>applyToolFilter(itemId, tool));
  if(s) s.addEventListener("change", ()=>applyToolFilter(itemId, tool));
  if(t) t.addEventListener("change", ()=>applyToolFilter(itemId, tool));
  applyToolFilter(itemId, tool);
}

document.addEventListener("DOMContentLoaded", ()=>{
  const q=document.getElementById("q");
  const lang=document.getElementById("lang");
  const only=document.getElementById("only_vuln");

  if(q) q.addEventListener("input", ()=>{updateShown();});
  if(lang) lang.addEventListener("change", ()=>{updateShown();});
  if(only) only.addEventListener("change", ()=>{setToggleState(); updateShown();});

  setToggleState();
  updateShown();

  const items = Array.from(document.querySelectorAll("[data-item]"));
  for(const it of items){
    const id = it.id;
    hookTool(id, "bandit");
    hookTool(id, "semgrep");
  }
});
"""


def issue_row(i):
    tool = esc(i.get("tool"))
    sev = esc(i.get("severity"))
    rule = esc(i.get("rule"))
    line = esc(i.get("line"))
    msg = esc(i.get("message"))

    return (
        f'<tr data-tool="{tool.lower()}" data-severity="{sev.lower()}">'
        f"<td class='col-tool'><span class='badge'>{tool}</span></td>"
        f"<td class='col-sev'>{sev}</td>"
        f"<td class='col-rule'><span class='t-one' title='{rule}'>{rule}</span></td>"
        f"<td class='col-line'><span class='t-line'>{line}</span></td>"
        f"<td class='col-msg'><pre class='t-msg'>{msg}</pre></td>"
        f"</tr>"
    )


def render_code_with_lines(code: str) -> str:
    lines = (code or "").splitlines()
    if not lines:
        lines = [""]
    inner = "\n".join(f"<span class='code-line'>{esc(l)}</span>" for l in lines)
    return f"<div class='code-wrap'><pre class='code-pre'>{inner}</pre></div>"


def render_fewshot_block(obj: dict) -> str:
    """Render the few-shot examples section (instead of Plan)."""
    mode = obj.get("fewshot_mode") or ""
    k = obj.get("fewshot_k")
    examples = obj.get("fewshot_examples") or []
    chosen = obj.get("fewshot_chosen_index")
    secure_count = obj.get("fewshot_secure_count")

    # هِدِر کوچک بالای لیست
    meta_chip_parts = []
    if mode:
        meta_chip_parts.append(f"mode: {esc(mode)}")
    if k is not None:
        meta_chip_parts.append(f"k: {esc(k)}")
    if secure_count is not None:
        meta_chip_parts.append(f"secure examples: {esc(secure_count)}")
    if chosen is not None and 0 <= int(chosen) < len(examples):
        meta_chip_parts.append(f"chosen index: {esc(chosen)}")

    meta_chip = ""
    if meta_chip_parts:
        meta_chip = (
            '<div class="chip" style="margin-top:10px;">'
            + " | ".join(meta_chip_parts) +
            "</div>"
        )

    if not examples:
        return f"""
{meta_chip}
<pre style="margin-top:10px;white-space:pre-wrap;overflow-wrap:normal;word-break:normal;line-height:1.35">
(no few-shot examples available for this task)
</pre>
""".strip()

    blocks = []
    for idx, ex in enumerate(examples):
        prompt = ex.get("example_prompt") or ""
        code = ex.get("example_code") or ""
        loc = ex.get("loc")
        ex_secure = ex.get("secure")

        b_res = ex.get("bandit_result") or {}
        s_res = ex.get("semgrep_result") or {}
        b_issues = issues_to_list(b_res.get("issues"))
        s_issues = issues_to_list(s_res.get("issues"))
        b_cnt = len(b_issues)
        s_cnt = len(s_issues)

        chips = []

        if ex_secure is not None:
            chips.append(f"secure: {esc(ex_secure)}")
        if loc is not None:
            chips.append(f"loc: {esc(loc)}")
        chips.append(f"bandit issues: {b_cnt}")
        chips.append(f"semgrep issues: {s_cnt}")

        chosen_label = ""
        if chosen is not None and int(chosen) == idx:
            chosen_label = '<span class="chip" style="background:#14532d;border-color:#16a34a;color:#bbf7d0;">chosen</span>'

        chips_html = ""
        if chips:
            chips_html = (
                "<div class='kv' style='margin-top:6px;'>"
                + "".join(f"<div class='chip'>{c}</div>" for c in chips) +
                "</div>"
            )

        block = f"""
<div class="card" style="margin:10px 0 4px 0;">
  <div class="subhead">
    <div class="subttl">Example #{idx}</div>
    {chosen_label}
  </div>
  {chips_html}
  <div style="height:8px"></div>
  <details>
    <summary>Example prompt</summary>
    <pre style="margin-top:10px;white-space:pre-wrap;overflow-wrap:normal;word-break:normal;line-height:1.35">{esc(prompt)}</pre>
  </details>
  <div style="height:8px"></div>
  <details>
    <summary>Example code</summary>
    {render_code_with_lines(code)}
  </details>
</div>
""".strip()
        blocks.append(block)

    return f"""
{meta_chip}
<div style="margin-top:10px;">
  {"".join(blocks)}
</div>
""".strip()


def render_item(i: int, obj: dict, langs: set):
    task = obj.get("task") or ""
    lang = obj.get("language") or ""
    langs.add((lang or "").strip().lower() or "unknown")
    loc = obj.get("loc") or 0
    intent = obj.get("intent") or ""
    code = obj.get("code") or ""

    bandit_block = obj.get("bandit_result") or {}
    semgrep_block = obj.get("semgrep_result") or {}

    bandit_issues = normalize_issues(issues_to_list(bandit_block.get("issues")), "bandit")
    semgrep_issues = normalize_issues(issues_to_list(semgrep_block.get("issues")), "semgrep")

    b_total = len(bandit_issues)
    s_total = len(semgrep_issues)

    b_rows = "\n".join(issue_row(x) for x in bandit_issues)
    s_rows = "\n".join(issue_row(x) for x in semgrep_issues)

    b_sev_opts = build_sev_options(bandit_issues)
    s_sev_opts = build_sev_options(semgrep_issues)

    b_tool_opts = build_tool_options(bandit_issues)
    s_tool_opts = build_tool_options(semgrep_issues)

    item_id = f"item_{i:04d}"

    if b_total == 0:
        bandit_html = f"""
<div>
  <div class="subhead">
    <div class="subttl">Bandit</div>
    <div class="chip">issues: <span data-bandit-count>0</span> / 0</div>
  </div>
  {render_no_issues("Bandit")}
</div>
""".strip()
    else:
        bandit_html = f"""
<div>
  <div class="subhead">
    <div class="subttl">Bandit</div>
    <div class="chip">issues: <span data-bandit-count>{b_total}</span> / {b_total}</div>
  </div>
  <div class="controls">
    <input data-bandit-q placeholder="Search bandit issues..."/>
    <select data-bandit-sev class="small">
      <option value="all">severity: all</option>
      {b_sev_opts}
    </select>
    <select data-bandit-tool class="small">
      <option value="all">tool: all</option>
      {b_tool_opts}
    </select>
  </div>
  <div class="table-wrap">
    <table data-bandit-table>
      <thead>
        <tr>
          <th class="col-tool">Tool</th>
          <th class="col-sev">Severity</th>
          <th class="col-rule">Rule</th>
          <th class="col-line">Line</th>
          <th class="col-msg">Message</th>
        </tr>
      </thead>
      <tbody>{b_rows}</tbody>
    </table>
  </div>
</div>
""".strip()

    if s_total == 0:
        semgrep_html = f"""
<div>
  <div class="subhead">
    <div class="subttl">Semgrep</div>
    <div class="chip">issues: <span data-semgrep-count>0</span> / 0</div>
  </div>
  {render_no_issues("Semgrep")}
</div>
""".strip()
    else:
        semgrep_html = f"""
<div>
  <div class="subhead">
    <div class="subttl">Semgrep</div>
    <div class="chip">issues: <span data-semgrep-count>{s_total}</span> / {s_total}</div>
  </div>
  <div class="controls">
    <input data-semgrep-q placeholder="Search semgrep issues..."/>
    <select data-semgrep-sev class="small">
      <option value="all">severity: all</option>
      {s_sev_opts}
    </select>
    <select data-semgrep-tool class="small">
      <option value="all">tool: all</option>
      {s_tool_opts}
    </select>
  </div>
  <div class="table-wrap">
    <table data-semgrep-table>
      <thead>
        <tr>
          <th class="col-tool">Tool</th>
          <th class="col-sev">Severity</th>
          <th class="col-rule">Rule</th>
          <th class="col-line">Line</th>
          <th class="col-msg">Message</th>
        </tr>
      </thead>
      <tbody>{s_rows}</tbody>
    </table>
  </div>
</div>
""".strip()

    return f"""
<div class="card" id="{item_id}" data-item data-lang="{esc((lang or '').strip().lower() or 'unknown')}" data-bandit="{b_total}" data-semgrep="{s_total}">
  <div class="subhead">
    <div class="subttl">{esc(task)}</div>
    <div class="chip">#{i:04d}</div>
  </div>

  <div class="kv">
    <div class="chip">language: {esc(lang)}</div>
    <div class="chip">loc: {esc(loc)}</div>
    <div class="chip">bandit: {b_total}</div>
    <div class="chip">semgrep: {s_total}</div>
  </div>

  <hr/>

  <details>
    <summary>Intent (input prompt)</summary>
    <pre style="margin-top:10px;white-space:pre-wrap;overflow-wrap:normal;word-break:normal;line-height:1.35">{esc(intent)}</pre>
  </details>

  <div style="height:10px"></div>

  <details>
    <summary>Few-shot examples</summary>
    {render_fewshot_block(obj)}
  </details>

  <div style="height:10px"></div>

  <details>
    <summary>Generated code</summary>
    {render_code_with_lines(code)}
  </details>

  <hr/>

  <div class="grid">
    <div class="card">{bandit_html}</div>
    <div class="card">{semgrep_html}</div>
  </div>
</div>
""".strip()


def write_html(path: Path, title: str, body: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{esc(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <div class="h1">{esc(title)}</div>
      <div class="meta">generated: {esc(now)}</div>
    </div>
  </div>

  {body}
</div>
<script>{TASK_JS}</script>
</body>
</html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(out, encoding="utf-8")


def jsonl_to_html(jsonl_path: str, html_path: str):
    records = load_jsonl(jsonl_path)
    langs = set()

    items_html = []
    for i, r in enumerate(records, start=1):
        items_html.append(render_item(i, r, langs))

    lang_opts = ['<option value="all">language: all</option>']
    for l in sorted(langs):
        lang_opts.append(f'<option value="{esc(l)}">{esc(l)}</option>')
    lang_opts = "\n".join(lang_opts)

    top = f"""
<div class="card">
  <div class="controls">
    <input id="q" placeholder="Search tasks / messages / code..."/>
    <select id="lang" class="small">
      {lang_opts}
    </select>

    <div class="toggle-row">
      <span class="toggle-label">only items with vulnerabilities</span>
      <label class="switch">
        <input id="only_vuln" type="checkbox">
        <span class="slider"></span>
      </label>
      <span id="only_vuln_state" class="state-pill state-off">OFF</span>
    </div>

    <div class="chip">shown: <span id="shown">0</span> / {len(records)}</div>
  </div>
  <div class="hint">Tip: Use search to filter across task, intent, code, and issue messages.</div>
</div>
""".strip()

    body = top + "\n" + "\n".join(items_html)
    write_html(Path(html_path), "One-shot Security Report", body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input jsonl file")
    ap.add_argument("-o", "--output", help="output html file (default: beside input)")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".html")

    jsonl_to_html(str(input_path), str(output_path))