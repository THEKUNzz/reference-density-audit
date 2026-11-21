#!/usr/bin/env python3
"""Regenerate quixbugs.jsonl with SHORT completions (the single-line fix).

Fill-in-the-blank setup: prompt = def line + docstring + function body with the
buggy line(s) replaced by a marker; certificate = the corrected line(s) only.
This makes QuixBugs the short-L anchor (~5-20 tokens) instead of the full body.
"""
import difflib
import json
import re
from pathlib import Path

SRC = Path("<path-to>/QuixBugs-master")
BUGGY = SRC / "python_programs"
CORRECT = SRC / "correct_python_programs"
OUT = Path("data")
OUT.mkdir(parents=True, exist_ok=True)


def split_code(code):
    """Return (def_line, indented_body_lines)."""
    lines = code.split("\n")
    def_line = None
    body = []
    for line in lines:
        if def_line is None and line.startswith("def "):
            def_line = line
            continue
        if def_line is not None:
            if line.strip() == "" and not body:
                continue  # skip blank right after def (unlikely)
            if line.startswith((" ", "\t")):
                body.append(line)
            elif line.strip() == "":
                body.append(line)  # keep in-function blank lines
            else:
                break  # non-indented line after body -> end of function
    return def_line, body


def extract_docstring(code):
    m = re.search(r'"""(.*?)"""', code, re.DOTALL)
    return m.group(0) if m else ""


items = []
skipped = []
for buggy_path in sorted(BUGGY.glob("*.py")):
    if buggy_path.name.endswith("_test.py"):
        continue
    task_id = buggy_path.stem
    correct_path = CORRECT / buggy_path.name
    if not correct_path.exists():
        skipped.append(task_id)
        continue
    buggy_code = buggy_path.read_text()
    correct_code = correct_path.read_text()

    buggy_def, buggy_body = split_code(buggy_code)
    correct_def, correct_body = split_code(correct_code)
    if buggy_def is None or correct_def is None:
        skipped.append(task_id + " (no def)")
        continue

    # diff the bodies to find the bug region (first replace)
    sm = difflib.SequenceMatcher(None, buggy_body, correct_body)
    bug_i1 = bug_i2 = fix_j1 = fix_j2 = None
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete") or (tag == "insert"):
            bug_i1, bug_i2, fix_j1, fix_j2 = i1, i2, j1, j2
            break
    if bug_i1 is None:
        skipped.append(task_id + " (no diff)")
        continue

    # prompt = def + docstring + body with bug region replaced by a marker
    marker = "    # FIX ME"
    prompt_body = buggy_body[:bug_i1] + [marker] + buggy_body[bug_i2:]
    doc = extract_docstring(buggy_code)
    prompt = buggy_def + "\n"
    if doc:
        prompt += "\n".join("    " + ln if ln.strip() else "" for ln in doc.split("\n")) + "\n"
    prompt += "\n".join(prompt_body) + "\n"

    # certificate = the corrected line(s)
    certificate = "\n".join(correct_body[fix_j1:fix_j2])

    items.append({"task_id": task_id, "prompt": prompt, "certificate": certificate})

with (OUT / "quixbugs.jsonl").open("w") as f:
    for item in items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"wrote {len(items)} tasks")
if skipped:
    print("skipped:", skipped)
lens = [len(i["certificate"].split()) for i in items]
print(f"certificate word counts: min={min(lens)} max={max(lens)} median={sorted(lens)[len(lens)//2]}")
