# GCFL-OTHER-0023


import os
import sys
import time
import subprocess
import traceback

def _env_str(name: str, default: str) -> str:
    v = os.getenv(name, default)
    return str(v).strip()

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, None)
    if v is None:
        return default
    v = str(v).strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, None)
    if v is None:
        return default
    v = str(v).strip()
    try:
        return int(v)
    except Exception:
        raise ValueError(f"Invalid int for {name}={v!r}")

def _pass():
    print("Test Passed ✅")
    sys.exit(0)

def _fail():
    print("Test Failed ❌")
    sys.exit(0)

def _skip(reason: str):
    print("SKIP_ENV:", reason)
    sys.exit(0)

def _print_env(torch, deepspeed):
    print("\n=== ENVIRONMENT ===")
    print("PYTHON:", sys.version)
    print("TORCH:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    try:
        print("GPU:", torch.cuda.get_device_name(0))
    except Exception:
        print("GPU: <unknown>")
    try:
        import transformers
        print("TRANSFORMERS:", transformers.__version__)
    except Exception:
        print("TRANSFORMERS: <not importable>")
    print("DEEPSPEED:", getattr(deepspeed, "__version__", "<unknown>"))
    print("===================\n")

# ---------------- CONFIG ----------------
SEED = 1337

CFG_MODE = _env_str("DS_MODE", "scan")  # scan | child | single
USE_KERNEL = _env_bool("DS_KERNEL_INJECT", False)

# decode parameters
BATCH = _env_int("DS_BATCH", 2)
PROMPT_LEN = _env_int("DS_PROMPT_LEN", 33)
VOCAB = _env_int("DS_VOCAB", 256)
STEPS = _env_int("DS_STEPS", 32)  # used in child/single
MAX_STEPS = _env_int("DS_SCAN_RANGE_MAX", 64)  # scan upper bound
SCAN_MIN = _env_int("DS_SCAN_RANGE_MIN", 4)
SCAN_TIMEOUT = float(_env_str("DS_SCAN_TIMEOUT", "180"))
CHILD_TIMEOUT = float(_env_str("DS_CHILD_TIMEOUT", "180"))

COMPILE_WARMUP = _env_bool("DS_COMPILE_WARMUP", True)
LAUNCH_BLOCKING = _env_bool("DS_LAUNCH_BLOCKING", False)
FAIL_FAST = _env_bool("DS_FAIL_FAST", True)

print(f"CFG_DS_KERNEL_INJECT: {USE_KERNEL}")
print(f"CFG_MODE: {CFG_MODE}")
print(f"CFG_STEPS: {STEPS}")
print(f"CFG_BATCH: {BATCH}")
print(f"CFG_PROMPT_LEN: {PROMPT_LEN}")
print(f"CFG_VOCAB: {VOCAB}")
print(f"CFG_SCAN_RANGE: {SCAN_MIN}..{MAX_STEPS}")
print(f"CFG_SCAN_TIMEOUT: {SCAN_TIMEOUT:.1f}s")
print(f"CFG_CHILD_TIMEOUT: {CHILD_TIMEOUT:.1f}s")
print(f"CFG_COMPILE_WARMUP: {COMPILE_WARMUP}")
print(f"CFG_LAUNCH_BLOCKING: {LAUNCH_BLOCKING}")
print(f"CFG_FAIL_FAST: {FAIL_FAST}")

if LAUNCH_BLOCKING:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ---------------- MODEL/DECODE ----------------
def _build_model(torch, deepspeed):
    from transformers import GPT2Config, GPT2LMHeadModel

    device = "cuda"
    dtype = torch.float16

    cfg = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=64,
        n_positions=1024,
        use_cache=True,
        vocab_size=VOCAB,
    )
    model = GPT2LMHeadModel(cfg).eval().to(device, dtype=dtype)

    if USE_KERNEL:
        if not hasattr(deepspeed, "init_inference"):
            _skip("deepspeed.init_inference not available")

        ds_cfg = {
            "dtype": dtype,
            "tensor_parallel": {"tp_size": 1},
            "replace_with_kernel_inject": True,
        }
        model = deepspeed.init_inference(model, config=ds_cfg)

    return model

def _looks_like_cuda_assert(msg: str) -> bool:
    m = msg.lower()
    return (
        "device-side assert" in m
        or "indexing.cu" in m
        or "indexselectsmallindex" in m
        or "cuda error" in m and "assert" in m
    )

def child_run(steps: int):
    # local imports only in child to keep parent fast
    try:
        import torch
        import deepspeed
    except Exception as e:
        _skip(f"import failed: {e}")

    if not torch.cuda.is_available():
        _skip("CUDA not available")

    _print_env(torch, deepspeed)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model = _build_model(torch, deepspeed)

    device = "cuda"

    # deterministic-ish inputs
    input_ids = torch.randint(0, VOCAB, (BATCH, PROMPT_LEN), device=device)
    past = None

    for step in range(int(steps)):
        try:
            out = model(input_ids=input_ids, past_key_values=past, use_cache=True)
            # Force sync to surface async CUDA errors at the true step
            torch.cuda.synchronize()

            past = getattr(out, "past_key_values", None)
            input_ids = torch.randint(0, VOCAB, (BATCH, 1), device=device)

        except RuntimeError as e:
            msg = str(e)
            if _looks_like_cuda_assert(msg):
                print(f"CHILD_CRASH_STEP: {step}")
                print(msg)
                sys.exit(2)
            raise

    print("CHILD_OK: no crash")
    sys.exit(0)

# ---------------- SUBPROCESS HELPERS ----------------
def _run_child_subprocess(steps: int, timeout_s: float):
    env = os.environ.copy()
    env["DS_MODE"] = "child"
    env["DS_STEPS"] = str(int(steps))

    t0 = time.time()
    try:
        r = subprocess.run(
            [sys.executable, "-u", __file__],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
        )
        dt = time.time() - t0
        out = r.stdout or ""
        crashed = ("CHILD_CRASH_STEP" in out) or (r.returncode == 2)
        return r.returncode, dt, False, crashed, out
    except subprocess.TimeoutExpired as e:
        dt = time.time() - t0
        out = (e.stdout or "") + "\n<TIMEOUT>\n"
        return 124, dt, True, False, out

def warmup_once():
    # Compile/extension warmup to match your earlier behavior (toggleable)
    if not COMPILE_WARMUP:
        return
    rc, dt, to, crashed, out = _run_child_subprocess(steps=1, timeout_s=max(60.0, CHILD_TIMEOUT))
    print(f"WARMUP: rc={rc} secs={dt:.2f} timed_out={to} crashed={crashed}")
    if crashed and FAIL_FAST:
        print(out)
        _pass()

def scan_min_step():
    warmup_once()
    any_crash = False

    for s in range(SCAN_MIN, MAX_STEPS + 1):
        rc, dt, to, crashed, out = _run_child_subprocess(steps=s, timeout_s=SCAN_TIMEOUT)
        print(f"SCAN_TRY: steps={s} rc={rc} secs={dt:.2f} timed_out={to} crashed={crashed}")
        if crashed:
            any_crash = True
            print(f"REPRO: kernel_inject={int(USE_KERNEL)} batch={BATCH} prompt={PROMPT_LEN} min_crash_steps={s}")
            print(out)
            _pass()
        if to and FAIL_FAST:
            print("NOTE: timed out; stopping early due to CFG_FAIL_FAST")
            _fail()

    if not any_crash:
        print("No crash found in scan range")
        _fail()

def main():
    # mode router
    mode = CFG_MODE.lower()
    if mode == "child":
        # child reads DS_STEPS
        child_run(STEPS)
        return

    if mode == "single":
        # single run in-proc (not recommended for CUDA asserts; use child)
        rc, dt, to, crashed, out = _run_child_subprocess(steps=STEPS, timeout_s=CHILD_TIMEOUT)
        print(out)
        if crashed:
            _pass()
        _fail()
        return

    # default: scan
    scan_min_step()

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print("HARNESS_ERROR:", type(e).__name__, str(e))
        traceback.print_exc()
        _fail()



# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Deepspeed


# Commands
# *****************
# unset DS_CHILD_STEPS DS_STEPS DS_FIND_MIN_STEP DS_SUBPROC_MODE
# export DS_KERNEL_INJECT=1
# export DS_MODE=scan
# export DS_SCAN_RANGE_MIN=4
# export DS_SCAN_RANGE_MAX=256
# export DS_SCAN_TIMEOUT=240
# export DS_COMPILE_WARMUP=0

# python -u testcases/deepspeed_testcase_2.py | tee scan_256.log




# Output:
# *****************
# CFG_DS_KERNEL_INJECT: True
# CFG_MODE: scan
# CFG_STEPS: 32
# CFG_BATCH: 2
# CFG_PROMPT_LEN: 33
# CFG_VOCAB: 256
# CFG_SCAN_RANGE: 4..256
# CFG_SCAN_TIMEOUT: 240.0s
# CFG_CHILD_TIMEOUT: 180.0s
# CFG_COMPILE_WARMUP: False
# CFG_LAUNCH_BLOCKING: False
# CFG_FAIL_FAST: True
# SCAN_TRY: steps=4 rc=0 secs=4.52 timed_out=False crashed=False
# SCAN_TRY: steps=5 rc=0 secs=4.39 timed_out=False crashed=False
# SCAN_TRY: steps=6 rc=0 secs=4.56 timed_out=False crashed=False
# SCAN_TRY: steps=7 rc=0 secs=4.37 timed_out=False crashed=False
# SCAN_TRY: steps=8 rc=0 secs=4.50 timed_out=False crashed=False
# SCAN_TRY: steps=9 rc=0 secs=4.61 timed_out=False crashed=False
# SCAN_TRY: steps=10 rc=0 secs=4.63 timed_out=False crashed=False
# SCAN_TRY: steps=11 rc=0 secs=4.62 timed_out=False crashed=False
# SCAN_TRY: steps=12 rc=0 secs=4.58 timed_out=False crashed=False
# SCAN_TRY: steps=13 rc=0 secs=4.62 timed_out=False crashed=False
# SCAN_TRY: steps=14 rc=0 secs=4.40 timed_out=False crashed=False
# SCAN_TRY: steps=15 rc=0 secs=4.32 timed_out=False crashed=False
# SCAN_TRY: steps=16 rc=0 secs=4.60 timed_out=False crashed=False
# SCAN_TRY: steps=17 rc=0 secs=4.56 timed_out=False crashed=False
# SCAN_TRY: steps=18 rc=0 secs=4.54 timed_out=False crashed=False
# SCAN_TRY: steps=19 rc=0 secs=4.53 timed_out=False crashed=False
# SCAN_TRY: steps=20 rc=0 secs=4.57 timed_out=False crashed=False
# SCAN_TRY: steps=21 rc=0 secs=4.60 timed_out=False crashed=False
# SCAN_TRY: steps=22 rc=0 secs=4.63 timed_out=False crashed=False
# SCAN_TRY: steps=23 rc=0 secs=4.54 timed_out=False crashed=False
# SCAN_TRY: steps=24 rc=0 secs=4.62 timed_out=False crashed=False
# SCAN_TRY: steps=25 rc=0 secs=4.62 timed_out=False crashed=False
# SCAN_TRY: steps=26 rc=0 secs=4.59 timed_out=False crashed=False
# SCAN_TRY: steps=27 rc=0 secs=4.63 timed_out=False crashed=False
# SCAN_TRY: steps=28 rc=0 secs=4.59 timed_out=False crashed=False
# SCAN_TRY: steps=29 rc=0 secs=4.63 timed_out=False crashed=False
# SCAN_TRY: steps=30 rc=0 secs=4.66 timed_out=False crashed=False
# SCAN_TRY: steps=31 rc=0 secs=4.58 timed_out=False crashed=False
# SCAN_TRY: steps=32 rc=0 secs=4.61 timed_out=False crashed=False
# SCAN_TRY: steps=33 rc=0 secs=4.59 timed_out=False crashed=False
# SCAN_TRY: steps=34 rc=0 secs=4.44 timed_out=False crashed=False
# SCAN_TRY: steps=35 rc=0 secs=4.68 timed_out=False crashed=False
# SCAN_TRY: steps=36 rc=0 secs=4.68 timed_out=False crashed=False
# SCAN_TRY: steps=37 rc=0 secs=4.60 timed_out=False crashed=False
# SCAN_TRY: steps=38 rc=0 secs=4.59 timed_out=False crashed=False
# SCAN_TRY: steps=39 rc=0 secs=4.59 timed_out=False crashed=False
# SCAN_TRY: steps=40 rc=0 secs=4.68 timed_out=False crashed=False
# SCAN_TRY: steps=41 rc=0 secs=4.67 timed_out=False crashed=False
# SCAN_TRY: steps=42 rc=0 secs=4.63 timed_out=False crashed=False
# SCAN_TRY: steps=43 rc=0 secs=4.71 timed_out=False crashed=False
# SCAN_TRY: steps=44 rc=0 secs=4.66 timed_out=False crashed=False
# SCAN_TRY: steps=45 rc=0 secs=4.52 timed_out=False crashed=False
# SCAN_TRY: steps=46 rc=0 secs=4.67 timed_out=False crashed=False
# SCAN_TRY: steps=47 rc=0 secs=4.74 timed_out=False crashed=False
# SCAN_TRY: steps=48 rc=0 secs=4.46 timed_out=False crashed=False
# SCAN_TRY: steps=49 rc=0 secs=4.60 timed_out=False crashed=False
# SCAN_TRY: steps=50 rc=0 secs=4.68 timed_out=False crashed=False
# SCAN_TRY: steps=51 rc=0 secs=4.63 timed_out=False crashed=False
# SCAN_TRY: steps=52 rc=0 secs=4.42 timed_out=False crashed=False
# SCAN_TRY: steps=53 rc=0 secs=4.35 timed_out=False crashed=False
# SCAN_TRY: steps=54 rc=0 secs=4.62 timed_out=False crashed=False
# SCAN_TRY: steps=55 rc=0 secs=4.45 timed_out=False crashed=False
# SCAN_TRY: steps=56 rc=0 secs=4.66 timed_out=False crashed=False
# SCAN_TRY: steps=57 rc=0 secs=4.67 timed_out=False crashed=False
# SCAN_TRY: steps=58 rc=0 secs=4.63 timed_out=False crashed=False
# SCAN_TRY: steps=59 rc=0 secs=4.47 timed_out=False crashed=False
# SCAN_TRY: steps=60 rc=0 secs=4.66 timed_out=False crashed=False
# SCAN_TRY: steps=61 rc=0 secs=4.60 timed_out=False crashed=False
# SCAN_TRY: steps=62 rc=0 secs=4.67 timed_out=False crashed=False
# SCAN_TRY: steps=63 rc=0 secs=4.43 timed_out=False crashed=False
# SCAN_TRY: steps=64 rc=0 secs=4.73 timed_out=False crashed=False
# SCAN_TRY: steps=65 rc=0 secs=4.50 timed_out=False crashed=False
# SCAN_TRY: steps=66 rc=0 secs=4.83 timed_out=False crashed=False
# SCAN_TRY: steps=67 rc=0 secs=4.46 timed_out=False crashed=False
# SCAN_TRY: steps=68 rc=0 secs=4.67 timed_out=False crashed=False
# SCAN_TRY: steps=69 rc=0 secs=4.61 timed_out=False crashed=False
# SCAN_TRY: steps=70 rc=0 secs=4.50 timed_out=False crashed=False
# SCAN_TRY: steps=71 rc=0 secs=4.63 timed_out=False crashed=False
# SCAN_TRY: steps=72 rc=0 secs=4.66 timed_out=False crashed=False
# SCAN_TRY: steps=73 rc=0 secs=4.55 timed_out=False crashed=False
# SCAN_TRY: steps=74 rc=0 secs=4.67 timed_out=False crashed=False
# SCAN_TRY: steps=75 rc=0 secs=4.45 timed_out=False crashed=False
# SCAN_TRY: steps=76 rc=0 secs=4.65 timed_out=False crashed=False
# SCAN_TRY: steps=77 rc=0 secs=4.71 timed_out=False crashed=False
# SCAN_TRY: steps=78 rc=0 secs=4.58 timed_out=False crashed=False
# SCAN_TRY: steps=79 rc=0 secs=4.72 timed_out=False crashed=False
# SCAN_TRY: steps=80 rc=0 secs=4.74 timed_out=False crashed=False
# SCAN_TRY: steps=81 rc=0 secs=4.39 timed_out=False crashed=False
# SCAN_TRY: steps=82 rc=0 secs=4.72 timed_out=False crashed=False
# SCAN_TRY: steps=83 rc=0 secs=4.76 timed_out=False crashed=False
# SCAN_TRY: steps=84 rc=0 secs=4.41 timed_out=False crashed=False
# SCAN_TRY: steps=85 rc=0 secs=4.67 timed_out=False crashed=False
# SCAN_TRY: steps=86 rc=0 secs=4.65 timed_out=False crashed=False
# SCAN_TRY: steps=87 rc=0 secs=4.72 timed_out=False crashed=False
# SCAN_TRY: steps=88 rc=0 secs=4.55 timed_out=False crashed=False
# SCAN_TRY: steps=89 rc=0 secs=4.77 timed_out=False crashed=False
# SCAN_TRY: steps=90 rc=0 secs=4.60 timed_out=False crashed=False
# SCAN_TRY: steps=91 rc=0 secs=4.46 timed_out=False crashed=False
# SCAN_TRY: steps=92 rc=0 secs=4.71 timed_out=False crashed=False
# SCAN_TRY: steps=93 rc=0 secs=4.47 timed_out=False crashed=False
# SCAN_TRY: steps=94 rc=0 secs=4.60 timed_out=False crashed=False
# SCAN_TRY: steps=95 rc=0 secs=4.69 timed_out=False crashed=False
# SCAN_TRY: steps=96 rc=0 secs=4.69 timed_out=False crashed=False
# SCAN_TRY: steps=97 rc=0 secs=4.68 timed_out=False crashed=False
# SCAN_TRY: steps=98 rc=0 secs=4.73 timed_out=False crashed=False
# SCAN_TRY: steps=99 rc=0 secs=4.73 timed_out=False crashed=False
# SCAN_TRY: steps=100 rc=0 secs=4.75 timed_out=False crashed=False
# SCAN_TRY: steps=101 rc=0 secs=4.51 timed_out=False crashed=False
# SCAN_TRY: steps=102 rc=0 secs=4.65 timed_out=False crashed=False
# SCAN_TRY: steps=103 rc=0 secs=4.71 timed_out=False crashed=False
# SCAN_TRY: steps=104 rc=0 secs=4.67 timed_out=False crashed=False
# SCAN_TRY: steps=105 rc=0 secs=4.76 timed_out=False crashed=False
# SCAN_TRY: steps=106 rc=0 secs=4.72 timed_out=False crashed=False
# SCAN_TRY: steps=107 rc=0 secs=4.58 timed_out=False crashed=False
# SCAN_TRY: steps=108 rc=0 secs=4.64 timed_out=False crashed=False
# SCAN_TRY: steps=109 rc=0 secs=4.75 timed_out=False crashed=False
# SCAN_TRY: steps=110 rc=0 secs=4.67 timed_out=False crashed=False
# SCAN_TRY: steps=111 rc=0 secs=4.57 timed_out=False crashed=False
# SCAN_TRY: steps=112 rc=0 secs=4.70 timed_out=False crashed=False
# SCAN_TRY: steps=113 rc=0 secs=4.55 timed_out=False crashed=False
# SCAN_TRY: steps=114 rc=0 secs=4.74 timed_out=False crashed=False
# SCAN_TRY: steps=115 rc=0 secs=4.49 timed_out=False crashed=False
# SCAN_TRY: steps=116 rc=0 secs=4.62 timed_out=False crashed=False
# SCAN_TRY: steps=117 rc=0 secs=4.54 timed_out=False crashed=False
# SCAN_TRY: steps=118 rc=0 secs=4.70 timed_out=False crashed=False
# SCAN_TRY: steps=119 rc=0 secs=4.73 timed_out=False crashed=False
# SCAN_TRY: steps=120 rc=0 secs=4.74 timed_out=False crashed=False
# SCAN_TRY: steps=121 rc=0 secs=4.72 timed_out=False crashed=False
# SCAN_TRY: steps=122 rc=0 secs=4.68 timed_out=False crashed=False
# SCAN_TRY: steps=123 rc=0 secs=4.49 timed_out=False crashed=False
# SCAN_TRY: steps=124 rc=0 secs=4.75 timed_out=False crashed=False
# SCAN_TRY: steps=125 rc=0 secs=4.77 timed_out=False crashed=False
# SCAN_TRY: steps=126 rc=0 secs=4.66 timed_out=False crashed=False
# SCAN_TRY: steps=127 rc=0 secs=4.70 timed_out=False crashed=False
# SCAN_TRY: steps=128 rc=0 secs=4.80 timed_out=False crashed=False
# SCAN_TRY: steps=129 rc=0 secs=4.57 timed_out=False crashed=False
# SCAN_TRY: steps=130 rc=0 secs=4.63 timed_out=False crashed=False
# SCAN_TRY: steps=131 rc=0 secs=4.72 timed_out=False crashed=False
# SCAN_TRY: steps=132 rc=0 secs=4.73 timed_out=False crashed=False
# SCAN_TRY: steps=133 rc=0 secs=4.77 timed_out=False crashed=False
# SCAN_TRY: steps=134 rc=0 secs=4.78 timed_out=False crashed=False
# SCAN_TRY: steps=135 rc=0 secs=4.69 timed_out=False crashed=False
# SCAN_TRY: steps=136 rc=0 secs=4.57 timed_out=False crashed=False
# SCAN_TRY: steps=137 rc=0 secs=4.69 timed_out=False crashed=False
# SCAN_TRY: steps=138 rc=0 secs=4.74 timed_out=False crashed=False
# SCAN_TRY: steps=139 rc=0 secs=4.79 timed_out=False crashed=False
# SCAN_TRY: steps=140 rc=0 secs=4.79 timed_out=False crashed=False
# SCAN_TRY: steps=141 rc=0 secs=4.70 timed_out=False crashed=False
# SCAN_TRY: steps=142 rc=0 secs=4.70 timed_out=False crashed=False
# SCAN_TRY: steps=143 rc=0 secs=4.50 timed_out=False crashed=False
# SCAN_TRY: steps=144 rc=0 secs=4.71 timed_out=False crashed=False
# SCAN_TRY: steps=145 rc=0 secs=4.80 timed_out=False crashed=False
# SCAN_TRY: steps=146 rc=0 secs=4.51 timed_out=False crashed=False
# SCAN_TRY: steps=147 rc=0 secs=4.75 timed_out=False crashed=False
# SCAN_TRY: steps=148 rc=0 secs=4.67 timed_out=False crashed=False
# SCAN_TRY: steps=149 rc=0 secs=4.59 timed_out=False crashed=False
# SCAN_TRY: steps=150 rc=0 secs=4.77 timed_out=False crashed=False
# SCAN_TRY: steps=151 rc=0 secs=4.51 timed_out=False crashed=False
# SCAN_TRY: steps=152 rc=0 secs=4.82 timed_out=False crashed=False
# SCAN_TRY: steps=153 rc=0 secs=4.75 timed_out=False crashed=False
# SCAN_TRY: steps=154 rc=0 secs=4.66 timed_out=False crashed=False
# SCAN_TRY: steps=155 rc=0 secs=4.78 timed_out=False crashed=False
# SCAN_TRY: steps=156 rc=0 secs=4.75 timed_out=False crashed=False
# SCAN_TRY: steps=157 rc=0 secs=4.76 timed_out=False crashed=False
# SCAN_TRY: steps=158 rc=0 secs=4.49 timed_out=False crashed=False
# SCAN_TRY: steps=159 rc=0 secs=4.77 timed_out=False crashed=False
# SCAN_TRY: steps=160 rc=0 secs=4.81 timed_out=False crashed=False
# SCAN_TRY: steps=161 rc=0 secs=4.60 timed_out=False crashed=False
# SCAN_TRY: steps=162 rc=0 secs=4.61 timed_out=False crashed=False
# SCAN_TRY: steps=163 rc=0 secs=4.83 timed_out=False crashed=False
# SCAN_TRY: steps=164 rc=0 secs=4.46 timed_out=False crashed=False
# SCAN_TRY: steps=165 rc=0 secs=4.78 timed_out=False crashed=False
# SCAN_TRY: steps=166 rc=0 secs=4.79 timed_out=False crashed=False
# SCAN_TRY: steps=167 rc=0 secs=4.50 timed_out=False crashed=False
# SCAN_TRY: steps=168 rc=0 secs=4.80 timed_out=False crashed=False
# SCAN_TRY: steps=169 rc=0 secs=4.63 timed_out=False crashed=False
# SCAN_TRY: steps=170 rc=0 secs=4.80 timed_out=False crashed=False
# SCAN_TRY: steps=171 rc=0 secs=4.71 timed_out=False crashed=False
# SCAN_TRY: steps=172 rc=0 secs=4.68 timed_out=False crashed=False
# SCAN_TRY: steps=173 rc=0 secs=4.82 timed_out=False crashed=False
# SCAN_TRY: steps=174 rc=0 secs=4.62 timed_out=False crashed=False
# SCAN_TRY: steps=175 rc=0 secs=4.72 timed_out=False crashed=False
# SCAN_TRY: steps=176 rc=0 secs=4.76 timed_out=False crashed=False
# SCAN_TRY: steps=177 rc=0 secs=4.75 timed_out=False crashed=False
# SCAN_TRY: steps=178 rc=0 secs=4.86 timed_out=False crashed=False
# SCAN_TRY: steps=179 rc=0 secs=4.71 timed_out=False crashed=False
# SCAN_TRY: steps=180 rc=0 secs=4.87 timed_out=False crashed=False
# SCAN_TRY: steps=181 rc=0 secs=4.71 timed_out=False crashed=False
# SCAN_TRY: steps=182 rc=0 secs=4.64 timed_out=False crashed=False
# SCAN_TRY: steps=183 rc=0 secs=4.79 timed_out=False crashed=False
# SCAN_TRY: steps=184 rc=0 secs=4.82 timed_out=False crashed=False
# SCAN_TRY: steps=185 rc=0 secs=4.73 timed_out=False crashed=False
# SCAN_TRY: steps=186 rc=0 secs=4.74 timed_out=False crashed=False
# SCAN_TRY: steps=187 rc=0 secs=4.80 timed_out=False crashed=False
# SCAN_TRY: steps=188 rc=0 secs=4.75 timed_out=False crashed=False
# SCAN_TRY: steps=189 rc=0 secs=4.76 timed_out=False crashed=False
# SCAN_TRY: steps=190 rc=0 secs=4.60 timed_out=False crashed=False
# SCAN_TRY: steps=191 rc=0 secs=4.73 timed_out=False crashed=False
# SCAN_TRY: steps=192 rc=0 secs=4.77 timed_out=False crashed=False
# SCAN_TRY: steps=193 rc=0 secs=4.88 timed_out=False crashed=False
# SCAN_TRY: steps=194 rc=0 secs=4.66 timed_out=False crashed=False
# SCAN_TRY: steps=195 rc=0 secs=4.69 timed_out=False crashed=False
# SCAN_TRY: steps=196 rc=0 secs=4.66 timed_out=False crashed=False
# SCAN_TRY: steps=197 rc=0 secs=4.86 timed_out=False crashed=False
# SCAN_TRY: steps=198 rc=0 secs=4.69 timed_out=False crashed=False
# SCAN_TRY: steps=199 rc=0 secs=4.88 timed_out=False crashed=False
# SCAN_TRY: steps=200 rc=0 secs=4.84 timed_out=False crashed=False
# SCAN_TRY: steps=201 rc=0 secs=4.86 timed_out=False crashed=False
# SCAN_TRY: steps=202 rc=0 secs=4.85 timed_out=False crashed=False
# SCAN_TRY: steps=203 rc=0 secs=4.61 timed_out=False crashed=False
# SCAN_TRY: steps=204 rc=0 secs=4.75 timed_out=False crashed=False
# SCAN_TRY: steps=205 rc=0 secs=4.73 timed_out=False crashed=False
# SCAN_TRY: steps=206 rc=0 secs=4.64 timed_out=False crashed=False
# SCAN_TRY: steps=207 rc=0 secs=4.80 timed_out=False crashed=False
# SCAN_TRY: steps=208 rc=0 secs=4.77 timed_out=False crashed=False
# SCAN_TRY: steps=209 rc=0 secs=4.82 timed_out=False crashed=False
# SCAN_TRY: steps=210 rc=0 secs=4.72 timed_out=False crashed=False
# SCAN_TRY: steps=211 rc=0 secs=4.78 timed_out=False crashed=False
# SCAN_TRY: steps=212 rc=0 secs=4.73 timed_out=False crashed=False
# SCAN_TRY: steps=213 rc=0 secs=4.73 timed_out=False crashed=False
# SCAN_TRY: steps=214 rc=0 secs=4.67 timed_out=False crashed=False
# SCAN_TRY: steps=215 rc=0 secs=4.89 timed_out=False crashed=False
# SCAN_TRY: steps=216 rc=0 secs=4.78 timed_out=False crashed=False
# SCAN_TRY: steps=217 rc=0 secs=4.81 timed_out=False crashed=False
# SCAN_TRY: steps=218 rc=0 secs=4.84 timed_out=False crashed=False
# SCAN_TRY: steps=219 rc=0 secs=4.83 timed_out=False crashed=False
# SCAN_TRY: steps=220 rc=0 secs=4.67 timed_out=False crashed=False
# SCAN_TRY: steps=221 rc=0 secs=4.86 timed_out=False crashed=False
# SCAN_TRY: steps=222 rc=0 secs=4.76 timed_out=False crashed=False
# SCAN_TRY: steps=223 rc=0 secs=4.91 timed_out=False crashed=False
# SCAN_TRY: steps=224 rc=0 secs=4.87 timed_out=False crashed=False
# SCAN_TRY: steps=225 rc=0 secs=4.83 timed_out=False crashed=False
# SCAN_TRY: steps=226 rc=0 secs=4.75 timed_out=False crashed=False
# SCAN_TRY: steps=227 rc=0 secs=4.56 timed_out=False crashed=False
# SCAN_TRY: steps=228 rc=0 secs=4.83 timed_out=False crashed=False
# SCAN_TRY: steps=229 rc=0 secs=4.73 timed_out=False crashed=False
# SCAN_TRY: steps=230 rc=0 secs=4.61 timed_out=False crashed=False
# SCAN_TRY: steps=231 rc=0 secs=4.91 timed_out=False crashed=False
# SCAN_TRY: steps=232 rc=0 secs=4.86 timed_out=False crashed=False
# SCAN_TRY: steps=233 rc=0 secs=4.96 timed_out=False crashed=False
# SCAN_TRY: steps=234 rc=0 secs=4.94 timed_out=False crashed=False
# SCAN_TRY: steps=235 rc=0 secs=4.85 timed_out=False crashed=False
# SCAN_TRY: steps=236 rc=0 secs=4.84 timed_out=False crashed=False
# SCAN_TRY: steps=237 rc=0 secs=4.85 timed_out=False crashed=False
# SCAN_TRY: steps=238 rc=0 secs=4.74 timed_out=False crashed=False
# SCAN_TRY: steps=239 rc=0 secs=4.89 timed_out=False crashed=False
# SCAN_TRY: steps=240 rc=0 secs=4.63 timed_out=False crashed=False
# SCAN_TRY: steps=241 rc=0 secs=4.76 timed_out=False crashed=False
# SCAN_TRY: steps=242 rc=0 secs=4.97 timed_out=False crashed=False
# SCAN_TRY: steps=243 rc=0 secs=4.69 timed_out=False crashed=False
# SCAN_TRY: steps=244 rc=0 secs=4.85 timed_out=False crashed=False
# SCAN_TRY: steps=245 rc=0 secs=4.85 timed_out=False crashed=False
# SCAN_TRY: steps=246 rc=0 secs=4.81 timed_out=False crashed=False
# SCAN_TRY: steps=247 rc=0 secs=4.82 timed_out=False crashed=False
# SCAN_TRY: steps=248 rc=0 secs=4.77 timed_out=False crashed=False
# SCAN_TRY: steps=249 rc=0 secs=4.82 timed_out=False crashed=False
# SCAN_TRY: steps=250 rc=0 secs=4.66 timed_out=False crashed=False
# SCAN_TRY: steps=251 rc=0 secs=4.78 timed_out=False crashed=False
# SCAN_TRY: steps=252 rc=0 secs=4.93 timed_out=False crashed=False
# SCAN_TRY: steps=253 rc=0 secs=4.91 timed_out=False crashed=False
# SCAN_TRY: steps=254 rc=0 secs=4.86 timed_out=False crashed=False
# SCAN_TRY: steps=255 rc=0 secs=4.84 timed_out=False crashed=False
# SCAN_TRY: steps=256 rc=0 secs=4.87 timed_out=False crashed=False
# No crash found in scan range
# Test Failed ❌

