import os
import re
import subprocess
from sys import stderr
from typing import Optional, List

TOKS = r"(?:tok(?:en)?s?)/s"
NUM = r"([0-9][0-9,]*\.?[0-9]*)"

PATTERNS = {
    "codec": re.compile(rf"Decode\s*=\s*{NUM}\s*{TOKS}", re.IGNORECASE),
    "vllm": re.compile(rf"output:\s*{NUM}\s*{TOKS}", re.IGNORECASE),
}


def _pick_backend(name: str) -> Optional[str]:
    n = (name or "").lower()
    if "vllm" in n:
        return "vllm"
    if "codec" in n:
        return "codec"
    return None


def extract_all_speeds(text: str) -> List[float]:
    bk = _pick_backend(os.getenv("LLM_BACKEND"))
    candidates = [bk] if bk in PATTERNS else ["codec", "vllm"]

    vals: List[float] = []
    for k in candidates:
        for m in PATTERNS[k].finditer(text):
            try:
                vals.append(float(m.group(1).replace(",", "")))
            except ValueError:
                pass
    return vals


def extract_max_speed(text: str) -> Optional[float]:
    vals = extract_all_speeds(text)
    return max(vals) if vals else None


cmd = ["python", "benchmark/e2e.py", "--scenario", os.getenv("SCENE", ""), "--index", os.getenv("INDEX", "")]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

stderr_output = ""
for line in proc.stderr:
    stderr_output += line

proc.wait()

vals = extract_all_speeds(stderr_output)
max_speed = max(vals) if vals else None

if max_speed is not None:
    print(f"Max speed = {max_speed:.2f} toks/s")
else:
    print("Error")
    print(stderr_output)
