# generate_testcases_v4.py
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import tempfile
import py_compile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import urllib.request
import urllib.error
import http.client

BASE_URL = "https://api.deepseek.com"
CHAT_PATH = "/chat/completions"

# ---------------------------
# 1) Robust extraction helpers
# ---------------------------

_FENCE_LINE_RE = re.compile(r"^\s*```.*?$", re.MULTILINE)
_JSON_KEY_RE = re.compile(r'"(python_code|code)"\s*:\s*', re.IGNORECASE)

def strip_markdown_fences(text: str) -> str:
    # Remove any ```... lines. (Not just ```python)
    return _FENCE_LINE_RE.sub("", text).strip()

def _try_parse_json_blob(text: str) -> Optional[Any]:
    """
    Try to parse text as JSON.
    Also try extracting the first {...} blob if text has extra junk around it.
    """
    t = text.strip()
    # direct parse
    try:
        return json.loads(t)
    except Exception:
        pass

    # Heuristic: extract first '{' .. last '}' and try again
    l = t.find("{")
    r = t.rfind("}")
    if l != -1 and r != -1 and r > l:
        blob = t[l : r + 1]
        try:
            return json.loads(blob)
        except Exception:
            pass

    # Heuristic: extract first '[' .. last ']' and try again
    l = t.find("[")
    r = t.rfind("]")
    if l != -1 and r != -1 and r > l:
        blob = t[l : r + 1]
        try:
            return json.loads(blob)
        except Exception:
            pass

    return None

def extract_python_code(model_text: str) -> str:
    """
    Accepts either:
      - raw python code
      - JSON like {"python_code": "..."} or {"code": "..."}
    Returns python code as string.
    """
    t = strip_markdown_fences(model_text)

    # If it's JSON (or contains JSON), parse and extract.
    obj = _try_parse_json_blob(t)
    if isinstance(obj, dict):
        for k in ("python_code", "code"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return strip_markdown_fences(v).strip() + "\n"

        # Some models return {"result": {"code": "..."}}
        for k in ("result", "data", "output"):
            v = obj.get(k)
            if isinstance(v, dict):
                for kk in ("python_code", "code"):
                    vv = v.get(kk)
                    if isinstance(vv, str) and vv.strip():
                        return strip_markdown_fences(vv).strip() + "\n"

    # If it looks like it tried to return JSON but failed, salvage from first "import"
    if _JSON_KEY_RE.search(t) and "import " in t:
        idx = t.find("import ")
        return t[idx:].strip() + "\n"

    # Otherwise assume it's raw python
    return t.strip() + "\n"

# ---------------------------
# 2) Hard sanitization (kills common failure patterns)
# ---------------------------

TRIPLE_QUOTE_ASSIGN_START_RE = re.compile(r"^(\s*[A-Za-z_]\w*\s*=\s*)([\"']{3})")
TRIPLE_QUOTE_RE = re.compile(r"([\"']{3})")

def remove_triple_quoted_assignments(code: str) -> str:
    """
    Removes blocks like:
      test_script = \"\"\" ... \"\"\"
      script = ''' ... '''
    These are the #1 source of unterminated-string failures in generated tests.
    """
    lines = code.splitlines()
    out: List[str] = []
    i = 0
    while i < len(lines):
        m = TRIPLE_QUOTE_ASSIGN_START_RE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue

        prefix = m.group(1)
        delim = m.group(2)
        # Replace assignment with safe empty string + note
        out.append(prefix + "''  # [REMOVED embedded script string to prevent syntax/truncation issues]")
        i += 1

        # Skip until we find the closing delimiter occurrence
        while i < len(lines):
            if delim in lines[i]:
                i += 1
                break
            i += 1
        continue

    return "\n".join(out) + "\n"

def hard_sanitize_python(code: str) -> str:
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = strip_markdown_fences(code)

    # Remove embedded script assignments (most frequent reason for TC10-like failures)
    code = remove_triple_quoted_assignments(code)

    # If model still left a JSON wrapper inside python, salvage from first import
    if '"code":' in code and "import " in code and not code.lstrip().startswith(("import", "from")):
        idx = code.find("import ")
        code = code[idx:]

    # If still contains ``` fences, kill again
    code = strip_markdown_fences(code)

    return code.strip() + "\n"

def syntax_check(code: str) -> Tuple[bool, str]:
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py", encoding="utf-8") as f:
            f.write(code)
            tmp = f.name
        py_compile.compile(tmp, doraise=True)
        return True, ""
    except Exception as e:
        return False, str(e)
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except Exception:
                pass

# ---------------------------
# 3) LLM call (DeepSeek) - optional
# ---------------------------

def key_fingerprint(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]

def deepseek_chat(api_key: str, model: str, messages: List[Dict[str, str]],
                 timeout_sec: int, max_retries: int = 6) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 2500,
    }
    data = json.dumps(payload).encode("utf-8")

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                BASE_URL + CHAT_PATH,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                raw_bytes = resp.read()

            raw = raw_bytes.decode("utf-8", errors="replace")
            obj = json.loads(raw)
            return obj["choices"][0]["message"]["content"]

        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            if e.code in (401, 403):
                raise RuntimeError(f"AUTH_ERROR {e.code}: {body}") from e
            last_err = f"HTTPError {e.code}: {body}"

        except (http.client.IncompleteRead, TimeoutError, urllib.error.URLError) as e:
            last_err = f"NETWORK_ERROR: {repr(e)}"

        except Exception as e:
            last_err = f"UNKNOWN_ERROR: {repr(e)}"

        # backoff
        sleep = min(10.0, 0.8 * (2 ** (attempt - 1))) + random.random() * 0.25
        time.sleep(sleep)

    raise RuntimeError(last_err or "deepseek_chat failed")

# ---------------------------
# 4) Prompts (force JSON output to avoid your exact failures)
# ---------------------------

def build_messages(test_obj: Dict[str, Any]) -> List[Dict[str, str]]:
    # The critical change: require JSON with python_code (NOT raw python).
    # This makes extraction deterministic and prevents ``` fences breaking syntax checks.
    system = (
        "You are generating a SINGLE Python test file.\n"
        "Return STRICT JSON ONLY, in this exact schema:\n"
        "{\n"
        '  "python_code": "<full python file as a string>"\n'
        "}\n\n"
        "CRITICAL RULES:\n"
        "1) Output MUST be valid JSON. No markdown. No ``` fences. No extra keys.\n"
        "2) The python_code MUST be valid Python.\n"
        "3) Do NOT embed another python script inside a triple-quoted string.\n"
        "   (No test_script='''...''', no TEST_TEMPLATE='''...''', etc.)\n"
        "4) If multi-process is needed, make THIS SAME FILE self-launching using subprocess + __file__.\n"
        "5) Include dependency guards; if missing packages or GPU requirements, print SKIP_ENV and exit(0).\n"
        "6) Keep runtime small. Tiny tensors. No downloads.\n"
        "7) Implement oracle using assertions; on bug signal raise AssertionError.\n"
        "8) Keep code size reasonable (<250 lines).\n"
    )

    user = (
        "Convert this test specification into a runnable Python test file.\n"
        "Return JSON only.\n\n"
        f"TEST_SPEC:\n{json.dumps(test_obj, indent=2, ensure_ascii=False)}\n"
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def build_repair_messages(test_obj: Dict[str, Any], bad_code: str, err: str) -> List[Dict[str, str]]:
    system = (
        "You repair a Python test file.\n"
        "Return STRICT JSON ONLY: {\"python_code\": \"...\"}\n"
        "No markdown. No extra keys.\n"
        "Do NOT embed other scripts in triple-quoted strings.\n"
    )
    user = (
        "The previous python_code failed syntax check.\n"
        f"SYNTAX_ERROR:\n{err}\n\n"
        "BAD_PYTHON_CODE:\n"
        "-----BEGIN-----\n"
        f"{bad_code}\n"
        "-----END-----\n\n"
        "Now return corrected JSON with python_code implementing the same TEST_SPEC:\n"
        f"{json.dumps(test_obj, indent=2, ensure_ascii=False)}\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

# ---------------------------
# 5) Fallback template (guaranteed valid .py)
# ---------------------------

def fallback_template(test_obj: Dict[str, Any], reason: str) -> str:
    spec = json.dumps(test_obj, ensure_ascii=False, indent=2)
    return f"""# AUTO-FALLBACK TEST (LLM generation failed)
# Reason: {reason}
# This file is syntactically valid and will SKIP so your pipeline never blocks.

import sys
import json

TEST_SPEC = {spec}

def main():
    print("SKIP_GEN: LLM failed to produce valid code for this spec.")
    print("GCFL_SPEC:", json.dumps({{"test_id": TEST_SPEC.get("test_id"), "target_library": TEST_SPEC.get("target_library")}}, ensure_ascii=False))
    sys.exit(0)

if __name__ == "__main__":
    main()
"""

# ---------------------------
# 6) Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--overwrite", type=int, default=0)
    ap.add_argument("--max_per_file", type=int, default=0)
    ap.add_argument("--sleep_sec", type=float, default=0.4)
    ap.add_argument("--timeout_sec", type=int, default=120)
    ap.add_argument("--max_repairs", type=int, default=6)
    ap.add_argument("--use_llm", type=int, default=1, help="1=use DeepSeek, 0=write fallback templates only")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    failures_path = out_dir / "failures.jsonl"

    json_files = sorted(in_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in: {in_dir}")

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if args.use_llm and not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set but --use_llm=1")

    if args.use_llm:
        print(f"[INFO] DeepSeek key fingerprint (sha1 first10) = {key_fingerprint(api_key)}")
        print(f"[INFO] Model={args.model} base_url={BASE_URL}")
    else:
        print("[INFO] --use_llm=0 (writing fallback templates only)")

    print(f"[INFO] Found {len(json_files)} JSON files in {in_dir}")
    print(f"[INFO] Output dir: {out_dir}")

    generated = skipped = failed = 0

    with manifest_path.open("a", encoding="utf-8") as mf, failures_path.open("a", encoding="utf-8") as ff:
        for jf in json_files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except Exception as e:
                ff.write(json.dumps({"status": "failed_file", "source_file": str(jf), "error": repr(e)}) + "\n")
                continue

            if not isinstance(data, list):
                continue

            objs = data if args.max_per_file <= 0 else data[: args.max_per_file]

            for obj in objs:
                test_id = obj.get("test_id") or "UNKNOWN_TEST_ID"
                safe_name = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", test_id)
                out_file = out_dir / f"{safe_name}.py"

                if out_file.exists() and not bool(args.overwrite):
                    mf.write(json.dumps({"status": "skipped", "test_id": test_id, "reason": "exists",
                                         "out_file": str(out_file), "source_file": str(jf)}) + "\n")
                    skipped += 1
                    continue

                try:
                    if not args.use_llm:
                        code = fallback_template(obj, "use_llm=0")
                        out_file.write_text(code, encoding="utf-8")
                        mf.write(json.dumps({"status": "generated_fallback", "test_id": test_id,
                                             "out_file": str(out_file), "source_file": str(jf)}) + "\n")
                        generated += 1
                        print(f"[OK] {test_id} -> {out_file.name} (fallback)")
                        continue

                    # LLM generation + multiple repair rounds
                    messages = build_messages(obj)
                    raw = deepseek_chat(api_key, args.model, messages, timeout_sec=args.timeout_sec)
                    code = hard_sanitize_python(extract_python_code(raw))

                    ok, err = syntax_check(code)
                    repairs_used = 0

                    while not ok and repairs_used < args.max_repairs:
                        repairs_used += 1
                        rep_msgs = build_repair_messages(obj, code, err)
                        raw2 = deepseek_chat(api_key, args.model, rep_msgs, timeout_sec=args.timeout_sec)
                        code = hard_sanitize_python(extract_python_code(raw2))
                        ok, err = syntax_check(code)

                    if not ok:
                        # Final: guaranteed-valid fallback file so you never block at scale
                        reason = f"SYNTAX_STILL_BAD after {repairs_used} repairs: {err}"
                        code = fallback_template(obj, reason)
                        out_file.write_text(code, encoding="utf-8")
                        mf.write(json.dumps({"status": "generated_fallback", "test_id": test_id,
                                             "out_file": str(out_file), "source_file": str(jf),
                                             "reason": reason}) + "\n")
                        generated += 1
                        print(f"[WARN] {test_id} -> {out_file.name} (fallback due to syntax)")
                    else:
                        out_file.write_text(code, encoding="utf-8")
                        mf.write(json.dumps({"status": "generated", "test_id": test_id,
                                             "out_file": str(out_file), "source_file": str(jf),
                                             "repairs_used": repairs_used}) + "\n")
                        generated += 1
                        print(f"[OK] {test_id} -> {out_file.name} (repairs_used={repairs_used})")

                except Exception as e:
                    failed += 1
                    ff.write(json.dumps({
                        "status": "failed",
                        "test_id": test_id,
                        "source_file": str(jf),
                        "out_file": str(out_file),
                        "error": repr(e),
                    }) + "\n")
                    print(f"[FAILED] {test_id} -> {repr(e)}")

                time.sleep(max(0.0, args.sleep_sec))

    print("\n====================")
    print(f"Done. generated={generated}, skipped={skipped}, failed={failed}")
    print(f"manifest: {manifest_path}")
    print(f"failures: {failures_path}")
    print("====================\n")

if __name__ == "__main__":
    main()
