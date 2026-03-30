import json
import sys
from typing import List, Dict, Any, Tuple, Set

# ==========================================
# Heuristic configuration
# ==========================================

ERROR_WORDS = [
    "error", "exception", "traceback", "crash", "segmentation fault",
    "segfault", "fails", "failed", "doesn't work", "does not work",
    "not working", "bug", "incorrect", "wrong result", "wrong output",
    "bad result", "nan", "inf", "overflow", "underflow",
    "abort", "aborted", "assertion failed", "stack trace",
]

BEHAVIOR_MISMATCH_WORDS = [
    "expected", "should be", "should return", "should give",
    "but got", "but i got", "but i'm getting",
    "wrong result", "incorrect result",
    "mismatch", "not equal", "not the same",
]

# Fix phrases that explicitly talk about *code* being correct / working
CODE_FIX_PHRASES = [
    "this is the correct code",
    "this is correct code",
    "correct code",
    "here is the correct version",
    "here's the correct version",
    "here is the working code",
    "here's the working code",
    "working code",
    "this code works",
    "it will work",
    "it works now",
    "this version works",
    "this example works",
    "this minimal example works",
    "this snippet works",
    "no longer fails",
    "no longer throws",
    "does not crash",
    "runs without error",
    "fixed code below",
    "fixed version below",
    "you should write",
    "you should use",
]

# More generic “fixed” phrases – still require code & no workaround / version-only stuff
STRONG_FIX_PHRASES = [
    "has been fixed",
    "was fixed",
    "is now fixed",
    "issue is fixed",
    "this issue is fixed",
    "should be fixed now",
    "this should be fixed now",
    "closing the issue as fixed",
    "the fix is",
    "here is the fix",
    "here's the fix",
    "here is the patch",
    "here's the patch",
    "this change fixes",
    "this patch fixes",
    "fixed at head",
    "fixed in head",
    "fixed in",
    "fixed by",
    "landed in",
    "merged in",
    "merged into",
]

STRONG_FIX_VERB_PATTERNS = [
    "we fixed",
    "i fixed",
    "fix committed",
    "fix is committed",
    "fix has landed",
    "fix landed",
]

WORKAROUND_WORDS = [
    "workaround",
    "a workaround is",
    "as a workaround",
    "temporary fix",
    "temp fix",
    "monkey patch",
    "monkey-patch",
    "hacky fix",
    "for now you can",
    "until this is fixed you can",
    "until then you can",
]

# Version / environment “fixes” – we *do not* treat these as code-level fixes
VERSION_FIX_WORDS = [
    "install tf-nightly",
    "pip install tf-nightly",
    "pip install tensorflow==",
    "pip3 install tensorflow==",
    "conda install",
    "conda create",
    "docker pull",
    "upgrade to tf",
    "upgrade to tensorflow",
    "downgrade to tf",
    "downgrade to tensorflow",
    "use tf ",
    "use tensorflow ",
    "use version",
    "tf-nightly",
    "nightly build",
    "nightly package",
]

TF_CODE_MARKERS = [
    "import tensorflow as tf",
    "import tensorflow",
    "tf.",
    "tensorflow.",
    "from tensorflow",
    "tf.keras",
    "keras.",
    "tf_keras.",
]

TRUSTED_AUTHOR_ROLES = {
    "CONTRIBUTOR",
    "MEMBER",
    "OWNER",
    "COLLABORATOR",
}

# We tighten these compared to earlier versions
MIN_SYMBOL_OVERLAP = 3          # previously 2
MIN_TOKEN_OVERLAP_RATIO = 0.6   # previously 0.5

# Require some minimum “substance” in a code block
MIN_CODE_TOKENS = 8


# ==========================================
# Utilities
# ==========================================

def extract_blocks(bug: Dict[str, Any]) -> List[Tuple[str, str, str, str, str]]:
    """
    Return time-ordered list of (kind, createdAt, text, role, login) blocks.
    kind: 'body' or 'comment'
    role: authorAssociation (for comments) or 'REPORTER' for body
    login: author.login
    """
    blocks: List[Tuple[str, str, str, str, str]] = []

    body_text = bug.get("body") or ""
    reporter_login = (bug.get("author") or {}).get("login") or ""
    if body_text.strip():
        blocks.append(("body", bug.get("createdAt", ""), body_text, "REPORTER", reporter_login))

    for c in bug.get("comments", []):
        text = c.get("body") or ""
        if not text.strip():
            continue
        created = c.get("createdAt", "")
        role = c.get("authorAssociation") or ""
        login = (c.get("author") or {}).get("login") or ""
        blocks.append(("comment", created, text, role, login))

    blocks.sort(key=lambda x: x[1] or "")
    return blocks


def split_code_and_text(text: str) -> Tuple[List[str], str]:
    """
    Split markdown-ish text into code blocks and plain text.
    Code blocks are delimited by ``` ... ```.
    """
    parts = text.split("```")
    code_blocks: List[str] = []
    plain_parts: List[str] = []

    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            code_blocks.append(part.strip())
        else:
            plain_parts.append(part)

    plain_text = "\n".join(plain_parts).strip()
    return code_blocks, plain_text


def contains_any(text: str, words: List[str]) -> bool:
    lower = text.lower()
    for w in words:
        if w.lower() in lower:
            return True
    return False


def has_tf_code_marker(text: str) -> bool:
    lower = text.lower()
    for m in TF_CODE_MARKERS:
        if m.lower() in lower:
            return True
    return False


def has_error_language(text: str) -> bool:
    return contains_any(text, ERROR_WORDS)


def has_behavior_mismatch(text: str) -> bool:
    return contains_any(text, BEHAVIOR_MISMATCH_WORDS)


def classify_fix_block(plain_text: str) -> str:
    """
    Classify a plain-text block as:
    - 'workaround'       – workaround / hack, not a proper fix
    - 'version_fix'      – upgrade/downgrade/install instructions only
    - 'code_fix'         – explicitly talks about correct / working code
    - 'generic_fix'      – generic 'fixed' language, no explicit version workaround
    - 'none'             – not a fix block
    """
    lower = plain_text.lower()

    if not lower.strip():
        return "none"

    if contains_any(lower, WORKAROUND_WORDS):
        return "workaround"

    if contains_any(lower, VERSION_FIX_WORDS):
        # version upgrade / downgrade, not a code-level fix
        return "version_fix"

    if contains_any(lower, CODE_FIX_PHRASES):
        return "code_fix"

    if contains_any(lower, STRONG_FIX_PHRASES) or contains_any(lower, STRONG_FIX_VERB_PATTERNS):
        return "generic_fix"

    return "none"


def extract_symbols_from_code(code: str) -> Set[str]:
    """
    Extract path / symbol-like tokens from code: dotted names, paths, etc.
    No regex, only string ops.
    """
    separators = ["(", ")", ",", ";", ":", "[", "]", "{", "}", "\t"]
    norm = code
    for s in separators:
        norm = norm.replace(s, " ")

    tokens = norm.replace("\n", " ").split()
    symbols: Set[str] = set()

    for tok in tokens:
        t = tok.strip("()[]{}.,;:'\"")
        if not t:
            continue
        if len(t) < 3:
            continue
        if "." in t or "/" in t or "_" in t:
            if any(c.isalpha() for c in t):
                symbols.add(t)

    return symbols


def code_token_set(code: str) -> Set[str]:
    """
    Coarse token set for overlap comparison.
    """
    tokens = code.replace("\n", " ").split()
    out: Set[str] = set()
    for tok in tokens:
        t = tok.strip("()[]{}.,;:'\"")
        if len(t) >= 3:
            out.add(t)
    return out


def tokens_similar_enough(c1: str, c2: str) -> bool:
    """
    Require high overlap between token sets to consider two snippets
    the 'same location' with minor changes.
    """
    s1 = code_token_set(c1)
    s2 = code_token_set(c2)
    if not s1 or not s2:
        return False
    inter = s1.intersection(s2)
    ratio = len(inter) / float(min(len(s1), len(s2)))
    return ratio >= MIN_TOKEN_OVERLAP_RATIO


def tf_symbol_subset(symbols: Set[str]) -> Set[str]:
    """
    Keep only tokens that look like TF / Keras symbols.
    """
    out: Set[str] = set()
    for s in symbols:
        ls = s.lower()
        if "tf." in ls or "tensorflow." in ls or "keras." in ls or "tf_keras." in ls:
            out.add(s)
    return out


def code_block_has_enough_tokens(code: str) -> bool:
    tokens = code_token_set(code)
    return len(tokens) >= MIN_CODE_TOKENS


# ==========================================
# Classification helpers
# ==========================================

def find_buggy_snippets(code_blocks: List[str], plain_text: str) -> List[Tuple[str, Set[str]]]:
    """
    Buggy snippets:
    - TF markers in surrounding text/code.
    - Error/mismatch language present.
    - Code block has enough “substance”.
    """
    buggy: List[Tuple[str, Set[str]]] = []
    if not code_blocks:
        return buggy

    text_for_bug = plain_text + "\n" + "\n".join(code_blocks)
    if not has_tf_code_marker(text_for_bug):
        return buggy
    if not (has_error_language(text_for_bug) or has_behavior_mismatch(text_for_bug)):
        return buggy

    for code in code_blocks:
        if not code.strip():
            continue
        if not code_block_has_enough_tokens(code):
            continue
        syms = extract_symbols_from_code(code)
        if syms:
            buggy.append((code, syms))

    return buggy


def find_fixed_snippets(
    code_blocks: List[str],
    plain_text: str,
    role: str,
    login: str,
    reporter_login: str
) -> List[Tuple[str, Set[str]]]:
    """
    Fixed snippets:
    - Block is classified as 'code_fix' OR (trusted author & 'generic_fix').
    - Not a workaround or version-only fix.
    - Contains TF code markers overall.
    - Contains code blocks with enough tokens and TF/keras symbols.
    """
    fixed: List[Tuple[str, Set[str]]] = []
    if not code_blocks:
        return fixed

    block_kind = classify_fix_block(plain_text)

    if block_kind in ("workaround", "version_fix", "none"):
        return fixed

    # For generic "fixed" language, require TF person or the original reporter.
    # For explicit CODE_FIX_PHRASES, accept any author.
    is_trusted_author = (role in TRUSTED_AUTHOR_ROLES) or (login == reporter_login)
    if block_kind == "generic_fix" and not is_trusted_author:
        return fixed

    text_for_fix = plain_text + "\n" + "\n".join(code_blocks)
    if not has_tf_code_marker(text_for_fix):
        return fixed

    for code in code_blocks:
        if not code.strip():
            continue
        if not code_block_has_enough_tokens(code):
            continue
        syms = extract_symbols_from_code(code)
        if not syms:
            continue
        # Require at least one TF-like symbol in the fix code
        if not tf_symbol_subset(syms):
            continue
        fixed.append((code, syms))

    return fixed


def buggy_and_fixed_pair_exists(bug: Dict[str, Any]) -> bool:
    """
    New core logic:
    - Find buggy snippets in earlier blocks.
    - Find fixed snippets in later blocks.
    - Require:
      * shared symbols,
      * at least one shared TF/keras symbol,
      * high token overlap,
      * code not identical.
    """
    blocks = extract_blocks(bug)
    if not blocks:
        return False

    reporter_login = (bug.get("author") or {}).get("login") or ""

    parsed: List[Tuple[List[str], str, str, str, str]] = []
    for kind, created_at, text, role, login in blocks:
        code_blocks, plain_text = split_code_and_text(text)
        parsed.append((code_blocks, plain_text, created_at, role, login))

    for i in range(len(parsed)):
        code_i, plain_i, _, role_i, login_i = parsed[i]
        buggy_snips = find_buggy_snippets(code_i, plain_i)
        if not buggy_snips:
            continue

        # Look forward for strong fix blocks
        for j in range(i + 1, len(parsed)):
            code_j, plain_j, _, role_j, login_j = parsed[j]
            fixed_snips = find_fixed_snippets(code_j, plain_j, role_j, login_j, reporter_login)
            if not fixed_snips:
                continue

            for buggy_code, buggy_syms in buggy_snips:
                for fixed_code, fixed_syms in fixed_snips:
                    if buggy_code.strip() == fixed_code.strip():
                        continue

                    shared_syms = buggy_syms.intersection(fixed_syms)
                    if len(shared_syms) < MIN_SYMBOL_OVERLAP:
                        continue

                    shared_tf_syms = tf_symbol_subset(shared_syms)
                    if not shared_tf_syms:
                        continue

                    if not tokens_similar_enough(buggy_code, fixed_code):
                        continue

                    combined = buggy_code + "\n" + fixed_code
                    if not has_tf_code_marker(combined):
                        continue

                    return True

    return False


def filter_bugs(bugs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for bug in bugs:
        try:
            if buggy_and_fixed_pair_exists(bug):
                result.append(bug)
        except Exception:
            # Skip corrupt entries, don't crash the whole run
            continue
    return result


# ==========================================
# CLI
# ==========================================

def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSON (list of bug objects)")
    ap.add_argument("--output", required=True, help="Output JSON file path")
    args = ap.parse_args()

    in_path = args.input
    out_path = args.output

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected input JSON to be a list of bug objects")

    filtered = filter_bugs(data)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"Input bugs: {len(data)}")
    print(f"Filtered (strict buggy + fixed code): {len(filtered)}")



if __name__ == "__main__":
    main()
