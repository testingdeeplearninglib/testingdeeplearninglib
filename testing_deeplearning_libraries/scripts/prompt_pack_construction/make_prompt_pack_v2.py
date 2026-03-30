# make_prompt_pack_v2.py
# Build execution-ready prompt packs from GCFL clusters.
# Adds: tiering, environment spec skeleton, strict output contract, gating, knobs,
#       oracle policy, allowed imports, launch command templates.

import json
import collections
import sys
import re

# ---------- helpers ----------

def top_counts(list_of_lists, k=12):
    c = collections.Counter()
    for xs in list_of_lists:
        for x in (xs or []):
            c[str(x)] += 1
    return [x for x, _ in c.most_common(k)]

def norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def choose_target_library(top_lib: str, members):
    # Use the dominant library unless it's NA; fallback to first member library.
    if top_lib and top_lib != "NA":
        return top_lib
    for m in members:
        lib = m.get("library")
        if lib:
            return lib
    return "unknown"

def default_version_spec(lib: str):
    # This is a *skeleton*; you will pin real versions in your execution matrix.
    # Keep it explicit so your pipeline can fill it later.
    libn = norm(lib)
    if "tensorflow" == libn:
        return {"package": "tensorflow", "versions": ["==2.20.0"], "python": ["3.10", "3.11"]}
    if libn in ("keras", "keras3"):
        return {"package": "keras", "versions": ["==3.12.*"], "python": ["3.10", "3.11"]}
    if libn in ("torch", "pytorch"):
        return {"package": "torch", "versions": ["==2.7.*"], "python": ["3.10", "3.11"]}
    if "deepspeed" in libn:
        return {"package": "deepspeed", "versions": ["==0.16.*"], "python": ["3.10", "3.11"], "requires": [{"package":"torch","versions":["==2.7.*"]}]}
    if "tvm" in libn:
        return {"package": "tvm", "versions": ["source_or_wheel"], "python": ["3.10", "3.11"]}
    return {"package": lib, "versions": ["unknown"], "python": ["3.10", "3.11"]}

def derive_tier(scenario: str, oracle_union, kw_top):
    scn = (scenario or "OTHER").upper()
    kws = set(norm(x) for x in (kw_top or []))
    orc = set(norm(x) for x in (oracle_union or []))

    if scn == "DISTRIBUTED":
        return "L"
    if "multi_gpu" in kws or "distributed_training" in kws or "nccl" in kws:
        return "L"
    if "cuda" in kws or "gpu" in kws or "tensorrt" in kws:
        # not necessarily multi-gpu
        return "M"
    if "crash" in orc and ("cuda" in kws or "gpu" in kws):
        return "M"
    return "S"

def derive_hw_requirements(tier: str, scenario: str, kw_top):
    kws = set(norm(x) for x in (kw_top or []))
    scn = (scenario or "OTHER").upper()

    needs_multi = (tier == "L") or (scn == "DISTRIBUTED")
    needs_gpu = needs_multi or (tier == "M") or ("gpu" in kws) or ("cuda" in kws)

    # conservative defaults
    if needs_multi:
        return {"needs_gpu": True, "needs_multi_gpu": True, "min_gpu_mem_mb": 6000, "world_size": 2}
    if needs_gpu:
        return {"needs_gpu": True, "needs_multi_gpu": False, "min_gpu_mem_mb": 3500, "world_size": 1}
    return {"needs_gpu": False, "needs_multi_gpu": False, "min_gpu_mem_mb": 0, "world_size": 1}

def derive_timeout_s(tier: str):
    return {"S": 60, "M": 180, "L": 300}.get(tier, 120)

def allowed_imports_for(lib: str, tier: str):
    base = [
        "os","sys","json","time","math","random","tempfile","pathlib","traceback"
    ]
    # allow numpy always (helps determinism and small utilities)
    base += ["numpy"]

    libn = norm(lib)
    if "deepspeed" in libn:
        # deepspeed requires torch
        base += ["torch","torch.distributed","torch.nn","torch.nn.functional","deepspeed"]
    elif libn in ("torch","pytorch"):
        base += ["torch","torch.nn","torch.nn.functional"]
        if tier == "L":
            base += ["torch.distributed"]
    elif libn == "tensorflow":
        base += ["tensorflow"]
    elif libn == "keras":
        base += ["keras"]
    elif "tvm" in libn:
        base += ["tvm"]
    else:
        # unknown target; keep minimal
        pass
    # de-dupe
    return sorted(set(base))

def derive_oracle_policy(scenario: str, oracle_union):
    orc = [norm(x) for x in (oracle_union or [])]
    orc_set = set(orc)

    # Hard oracle first
    if "crash" in orc_set:
        return {
            "oracle_type": "crash",
            "oracle_rule": "Any segfault/illegal memory access/assertion/abort -> Test Passed ✅ (bug triggered). Normal completion -> Test Failed ❌.",
            "repeat_policy": {"enabled": False}
        }
    if "exception" in orc_set:
        return {
            "oracle_type": "exception",
            "oracle_rule": "Expected bug signal is a specific exception (type/message). If triggered -> Test Passed ✅ else Test Failed ❌.",
            "repeat_policy": {"enabled": False}
        }

    # Differential or numeric oracles need repeats
    if "mode_mismatch" in orc_set:
        return {
            "oracle_type": "differential_mode",
            "oracle_rule": "Compare eager vs graph/compile outputs or behavior. If mismatch OR graph crashes but eager works -> Test Passed ✅.",
            "repeat_policy": {"enabled": True, "runs": 3, "require_consistent": True}
        }
    if "grad_mismatch" in orc_set:
        return {
            "oracle_type": "grad_mismatch",
            "oracle_rule": "Compare gradients across modes/devices/backends. If mismatch beyond tolerance consistently -> Test Passed ✅.",
            "repeat_policy": {"enabled": True, "runs": 3, "require_consistent": True}
        }
    if "output_mismatch" in orc_set:
        return {
            "oracle_type": "output_mismatch",
            "oracle_rule": "Numeric mismatch beyond tolerance consistently -> Test Passed ✅.",
            "repeat_policy": {"enabled": True, "runs": 3, "require_consistent": True}
        }

    return {
        "oracle_type": "exception",
        "oracle_rule": "Fallback: treat unexpected exception as bug trigger -> Test Passed ✅. Otherwise Test Failed ❌.",
        "repeat_policy": {"enabled": False}
    }

def launch_cmd_template(lib: str, hw: dict):
    libn = norm(lib)
    if hw["needs_multi_gpu"]:
        # torchrun default
        return "torchrun --nproc_per_node={world_size} {script} 2>&1 | tee {log}"
    return "python {script} 2>&1 | tee {log}"

def gating_policy(hw: dict):
    # This is what the code MUST do at runtime.
    if hw["needs_multi_gpu"]:
        return {
            "rules": [
                "If torch.distributed not available -> SKIP_ENV",
                "If CUDA not available -> SKIP_ENV",
                "If world_size < 2 -> SKIP_ENV",
                f"If min GPU memory < {hw['min_gpu_mem_mb']}MB -> SKIP_ENV",
            ]
        }
    if hw["needs_gpu"]:
        return {
            "rules": [
                "If CUDA/GPU not available -> SKIP_ENV",
                f"If min GPU memory < {hw['min_gpu_mem_mb']}MB -> SKIP_ENV",
            ]
        }
    return {"rules": ["If required package missing -> SKIP_ENV"]}

def knobs_for(tier: str, hw: dict):
    # knobs are env vars; defaults must be fast.
    if tier == "L":
        return {
            "ITERS": 50,
            "BATCH": 1,
            "SEQ": 8,
            "D_MODEL": 256,
            "VOCAB": 4096,
            "WORLD_SIZE": hw.get("world_size", 2),
            "SEED": 2026,
        }
    if tier == "M":
        return {"ITERS": 30, "BATCH": 1, "SEQ": 8, "D_MODEL": 256, "VOCAB": 4096, "SEED": 2026}
    return {"ITERS": 10, "BATCH": 1, "SEQ": 4, "D_MODEL": 128, "VOCAB": 2048, "SEED": 2026}

# ---------- main ----------

def main():
    if len(sys.argv) != 3:
        print("USAGE: python make_prompt_pack_v2.py <GCFL_JSON> <PROMPT_PACK_JSON>")
        raise SystemExit(2)

    p = sys.argv[1]
    outp = sys.argv[2]
    gcfl = json.load(open(p, "r", encoding="utf-8"))

    packs = []
    for c in gcfl:
        members = c.get("members", [])
        if not members:
            continue

        scenario = c.get("scenario", "OTHER")
        size = int(c.get("size", len(members)))

        libs = collections.Counter((m.get("library") or "NA") for m in members)
        top_lib, top_cnt = libs.most_common(1)[0]
        purity = (top_cnt / size) if size else 0.0

        oracle_union = sorted({t for m in members for t in (m.get("oracle_types") or [])})
        fam_top = top_counts([m.get("families") for m in members], k=15)
        kw_top  = top_counts([m.get("keywords") for m in members], k=25)

        target_lib = choose_target_library(top_lib, members)
        tier = derive_tier(scenario, oracle_union, kw_top)
        hw = derive_hw_requirements(tier, scenario, kw_top)

        pack = {
            "gcfl_id": c.get("gcfl_id"),
            "scenario_type": scenario,
            "cluster_size": size,
            "library_purity": round(purity, 3),
            "top_library": top_lib,
            "target": {
                "library": target_lib,
                "version_spec": default_version_spec(target_lib),
            },
            "tier": tier,
            "hardware": hw,
            "timeout_s": derive_timeout_s(tier),
            "allowed_imports": allowed_imports_for(target_lib, tier),
            "launch_cmd_template": launch_cmd_template(target_lib, hw),
            "output_contract": {
                "required_lines": ["ENV:", "Test Passed ✅", "Test Failed ❌", "SKIP_ENV:"],
                "semantics": {
                    "Test Passed ✅": "BUG/SUSPICIOUS TRIGGERED",
                    "Test Failed ❌": "NO BUG OBSERVED",
                    "SKIP_ENV: ...": "NOT RUNNABLE HERE",
                }
            },
            "gating_policy": gating_policy(hw),
            "knobs": knobs_for(tier, hw),
            "oracle_policy": derive_oracle_policy(scenario, oracle_union),
            "cluster_signals": {
                "oracle_types": oracle_union,
                "top_families": fam_top,
                "top_keywords": kw_top,
            },
            "evidence_members": [
                {
                    "library": m.get("library"),
                    "bug_no": m.get("bug_no"),
                    "title": m.get("title"),
                    "url": m.get("url"),
                }
                for m in members[:12]
            ],
        }

        packs.append(pack)

    json.dump(packs, open(outp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("WROTE", outp, "clusters=", len(packs))

if __name__ == "__main__":
    main()
