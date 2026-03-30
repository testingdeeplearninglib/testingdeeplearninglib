# GCFL-DISTRIBUTE-0018

import os
import sys
import json
import signal
import random
from contextlib import contextmanager

def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, default)).strip())
    except Exception:
        return default

GCFL_STRESS_ITERS = _env_int("GCFL_STRESS_ITERS", 50)
GCFL_SEQ_LEN      = _env_int("GCFL_SEQ_LEN", 2048)
GCFL_BATCH        = _env_int("GCFL_BATCH", 2)
GCFL_NEW_TOKENS   = _env_int("GCFL_NEW_TOKENS", 32)

def _skip(reason: str):
    print(f"SKIP_ENV: {reason}")
    sys.exit(0)

def _pass():
    print("Test Passed ✅")
    sys.exit(0)

def _fail():
    print("Test Failed ❌")
    sys.exit(0)

def _harness_error(e: BaseException):
    print(f"HARNESS_ERROR: {type(e).__name__}: {e}")
    sys.exit(1)

GCFL_HIDDEN = _env_int("GCFL_HIDDEN", 2048)
GCFL_LAYERS = _env_int("GCFL_LAYERS", 24)
GCFL_VOCAB  = _env_int("GCFL_VOCAB", 32000)
SEED = 1337

def set_deterministic_seeds():
    random.seed(SEED)
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))
    try:
        import numpy as np
        np.random.seed(SEED)
    except Exception:
        pass

class TimeoutError(RuntimeError):
    pass

from contextlib import contextmanager
@contextmanager
def time_limit(seconds: int, label: str):
    if seconds <= 0:
        yield
        return

    def _handle(signum, frame):
        raise TimeoutError(f"TIMEOUT({label}): exceeded {seconds}s")

    old_handler = None
    try:
        old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _handle)
        signal.alarm(int(seconds))
        yield
    except AttributeError:
        yield
    finally:
        try:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        except Exception:
            pass

def init_distributed(torch):
    if not (torch.distributed.is_available() and not torch.distributed.is_initialized()):
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    requested = os.environ.get("DS_DIST_BACKEND", "").strip().lower()
    if requested:
        backends = [requested]
    else:
        backends = ["nccl" if torch.cuda.is_available() else "gloo", "gloo"]

    last = None
    for backend in backends:
        try:
            torch.distributed.init_process_group(backend=backend, init_method="env://")
            return
        except Exception as e:
            last = (backend, e)
    if last:
        backend, e = last
        _skip(f"distributed init failed ({backend}): {type(e).__name__}: {e}")

def cleanup_distributed(torch):
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        pass

def looks_like_inflight_params_error(e: BaseException) -> bool:
    s = (str(e) or "").lower()
    return ("inflight" in s and "param" in s) or ("inflight parameters" in s)

def looks_like_timeout(e: BaseException) -> bool:
    return isinstance(e, TimeoutError) or ("timeout(" in (str(e) or "").lower())

def build_tiny_lm(torch):
    import torch.nn as nn

    class TinyCausalLM(nn.Module):
        def __init__(self, vocab=256, d=1024, n_layers=16, n_heads=16, max_len=None):
            if max_len is None:
                try:
                    max_len = int(os.environ.get('GCFL_MAX_LEN', '512').strip())
                except Exception:
                    max_len = 512
            super().__init__()
            self.vocab = vocab
            self.max_len = max_len
            self.tok = nn.Embedding(vocab, d)
            print("DBG: d_model =", d, "n_heads =", n_heads, "d%n_heads =", (d % n_heads))
            self.pos = nn.Embedding(max_len, d)

            enc_layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=n_heads,
                dim_feedforward=4 * d,
                dropout=0.0,
                batch_first=True,
            )
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.lm_head = nn.Linear(d, vocab, bias=False)

        def forward(self, input_ids):
            B, T = input_ids.shape
            if T > self.max_len:
                raise RuntimeError(f"input too long: T={T} > max_len={self.max_len}")
            pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
            x = self.tok(input_ids) + self.pos(pos)

            mask = torch.triu(torch.ones(T, T, device=input_ids.device, dtype=torch.bool), diagonal=1)
            mask = None  # DEBUG: disable attention mask to isolate crash
            x = self.enc(x, mask=mask)
            logits = self.lm_head(x)
            return logits

        @torch.no_grad()
        def generate_like(self, input_ids, max_new_tokens=20):
            for _ in range(max_new_tokens):
                logits = self.forward(input_ids)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_tok], dim=1)
            return input_ids

    return TinyCausalLM()

def _deepcopy(obj):
    return json.loads(json.dumps(obj))

def get_model_from_engine(engine):
    if hasattr(engine, "module") and engine.module is not None:
        return engine.module
    return engine

def try_init_deepspeed(deepspeed, model, ds_config):
    errs = []
    try:
        eng, _, _, _ = deepspeed.initialize(model=model, model_parameters=None, config=ds_config)
        return eng, None
    except Exception as e:
        errs.append(("initialize", e))
    msg = "; ".join([f"{name}:{type(e).__name__}:{e}" for name, e in errs])
    return None, msg

def main():
    set_deterministic_seeds()
    MAX_RUNTIME_SEC = 600
    RUN_PREFETCH0_VARIANT = os.environ.get("DS_PREFETCH0", "0").strip() == "1"
    REQUIRE_BF16 = os.environ.get("REQUIRE_BF16", "1").strip() != "0"

    try:
        try:
            import torch
        except Exception as e:
            _skip(f"missing torch: {type(e).__name__}: {e}")

        try:
            import deepspeed
        except Exception as e:
            _skip(f"missing deepspeed: {type(e).__name__}: {e}")

        if not torch.cuda.is_available():
            _skip("CUDA GPU not available (spec assumes GPU ZeRO-3 inference).")

        if REQUIRE_BF16:
            try:
                if not bool(torch.cuda.is_bf16_supported()):
                    _skip("bf16 not supported by this CUDA runtime/hardware (REQUIRE_BF16=1).")
            except Exception:
                _skip("bf16 support check failed (REQUIRE_BF16=1).")

        init_distributed(torch)

        base_ds_config = {
            "train_batch_size": 2,
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "bf16": {"enabled": True} if REQUIRE_BF16 else {"enabled": False},
            "fp16": {"enabled": False},
            "zero_optimization": {
                "stage": 3,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 2e7,
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_param_persistence_threshold": 1e5,
                "stage3_gather_16bit_weights_on_model_save": False,
                "stage3_prefetch_bucket_size": 5e7,
            },
            "wall_clock_breakdown": False,
        }

        def run_once(prefetch0: bool) -> str:
            ds_config = _deepcopy(base_ds_config)
            if prefetch0:
                ds_config["zero_optimization"]["stage3_prefetch_bucket_size"] = 0

            device = torch.device("cuda")
            dtype = torch.float32 if os.environ.get('FORCE_FP32','0').strip()=='1' else (torch.bfloat16 if REQUIRE_BF16 else torch.float16)

            model = build_tiny_lm(torch).to(device=device, dtype=dtype)
            model.eval()

            engine, err = try_init_deepspeed(deepspeed, model, ds_config)
            if engine is None:
                _skip(f"unable to init DeepSpeed engine: {err}")

            m = get_model_from_engine(engine)
            ids = torch.randint(0, 256, (GCFL_BATCH, GCFL_SEQ_LEN), device=device, dtype=torch.long)

            try:
                with time_limit(180, "generate"):
                    with torch.no_grad():
                        for _i in range(GCFL_STRESS_ITERS):
                            _ = m.generate_like(ids, max_new_tokens=GCFL_NEW_TOKENS)
                return "fail"
            except BaseException as e:
                import traceback
                print('TRACEBACK_IN_RUN_ONCE:')
                traceback.print_exc()
                if looks_like_inflight_params_error(e) or looks_like_timeout(e):
                    return "pass"
                if "inflight" in (str(e) or "").lower():
                    return "pass"
                print(f"EXC_IN_RUN_ONCE: {type(e).__name__}: {e}")
                return "fail"

        with time_limit(MAX_RUNTIME_SEC, "overall"):
            r1 = run_once(prefetch0=False)
            if r1 == "pass":
                _pass()
            if RUN_PREFETCH0_VARIANT:
                r2 = run_once(prefetch0=True)
                if r2 == "pass":
                    _pass()
            _fail()

    except SystemExit:
        raise
    except BaseException as e:
        _harness_error(e)
    finally:
        try:
            import torch
            cleanup_distributed(torch)
        except Exception:
            pass

if __name__ == "__main__":
    main()



# ******************************************************************************
#                                    Result 
# ******************************************************************************

# DeepSpeed


# Trigeering Commands
# *****************

# mkdir -p evidence_gcfl_0018

# cp -f gcfl_distribute_0018.py evidence_gcfl_0018/gcfl_distribute_0018.py
# nl -ba gcfl_distribute_0018.py > evidence_gcfl_0018/gcfl_distribute_0018_lineno.txt

# source ~/.venvs/dl_testing/bin/activate
# python - <<'PY' > evidence_gcfl_0018/versions.txt
# import sys, torch
# print("python =", sys.version.replace("\n"," "))
# print("torch  =", torch.__version__)
# try:
#     import deepspeed
#     print("deepspeed =", deepspeed.__version__)
# except Exception as e:
#     print("deepspeed import FAILED:", repr(e))
# PY

# export CUDA_VISIBLE_DEVICES=0
# export GCFL_SEQ_LEN=1024
# export GCFL_BATCH=2
# export GCFL_STRESS_ITERS=1
# export GCFL_NEW_TOKENS=1

# deepspeed --num_gpus=1 gcfl_distribute_0018.py 2>&1 | tee evidence_gcfl_0018/run_output.txt


# Output:
# *****************
# [2026-01-19 17:16:08,763] [WARNING] [runner.py:232:fetch_hostfile] Unable to find hostfile, will proceed with training with local resources only.
# Detected VISIBLE_DEVICES=0 but
# ...
# DBG: d_model = 1024 n_heads = 16 d%n_heads = 0
# DeepSpeedZeRoOffload initialize [begin]
# ...
# TRACEBACK_IN_RUN_ONCE:
# Traceback (most recent call last):
#   File "/home/talha/dl_testing/testcases/gcfl_distribute_0018.py", line 299, in run_once
#     _ = m.generate_like(ids, max_new_tokens=GCFL_NEW_TOKENS)
#   File "/home/talha/dl_testing/testcases/gcfl_distribute_0018.py", line 188, in generate_like
#     logits = self.forward(input_ids)
#   File "/home/talha/dl_testing/testcases/gcfl_distribute_0018.py", line 180, in forward
#     x = self.enc(x, mask=mask)
#   ...
#   File "/home/talha/.venvs/dl_testing/lib/python3.12/site-packages/torch/nn/functional.py", line 6417, in multi_head_attention_forward
#     attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
# RuntimeError: mat2 must be a matrix, got 1-D tensor
# EXC_IN_RUN_ONCE: RuntimeError: mat2 must be a matrix, got 1-D tensor
# Test Failed ❌
# [2026-01-19 17:16:27,455] [INFO] [launch.py:367:main] Process 1955419 exits successfully.

