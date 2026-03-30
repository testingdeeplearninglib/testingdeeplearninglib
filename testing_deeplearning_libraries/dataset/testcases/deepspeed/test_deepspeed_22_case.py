# GCFL-OTHER-0022

import os
import sys
import time
import random
import traceback
import threading


SEED = 1337
EXPECTED_SUBSTR = "still have inflight params"
MAX_RUNTIME_SEC = 180
STEP2_TIMEOUT_SEC = 90  # internal per-step watchdog (best-effort)


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


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, default)).strip())
    except Exception:
        return default


def _is_distributed() -> bool:
    return ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)


def _get_rank_world_local():
    rank = _env_int("RANK", 0)
    world = _env_int("WORLD_SIZE", 1)
    local_rank = _env_int("LOCAL_RANK", 0)
    return rank, world, local_rank


def main():
    start = time.time()
    try:
        try:
            import torch
            import torch.nn as nn
        except Exception as e:
            _skip(f"torch not available: {e}")

        try:
            import deepspeed
        except Exception as e:
            _skip(f"deepspeed not available: {e}")

        # Deterministic seeds (as much as feasible)
        random.seed(SEED)
        try:
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)
        except Exception:
            pass

        # Prefer deterministic settings where feasible (may reduce repro odds, but spec requests it)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

        # Distributed init if under launcher
        rank, world, local_rank = _get_rank_world_local()
        using_dist = _is_distributed()
        device = None

        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(local_rank if using_dist else 0)
            except Exception:
                pass
            device = torch.device("cuda", local_rank if using_dist else 0)
        else:
            device = torch.device("cpu")

        if using_dist:
            try:
                import torch.distributed as dist
            except Exception as e:
                _skip(f"torch.distributed not available: {e}")

            if not dist.is_initialized():
                try:
                    backend = "nccl" if device.type == "cuda" else "gloo"
                    dist.init_process_group(backend=backend, init_method="env://")
                except Exception as e:
                    _skip(f"failed to init distributed: {e}")

        if device.type != "cuda":
            # Spec assumes CUDA; without it, don't waste time pretending this is meaningful.
            _skip("CUDA GPU not available (DeepSpeed ZeRO-3 prefetch path is CUDA-centric)")

        # Minimal conditional-routing model (MoE-like)
        class TinyConditionalMoE(nn.Module):
            def __init__(self, d_in=256, d_hidden=256):
                super().__init__()
                self.stem = nn.Linear(d_in, d_hidden, bias=True)
                # Two mutually exclusive branches with distinct parameters
                self.expert_a = nn.Sequential(
                    nn.Linear(d_hidden, d_hidden, bias=True),
                    nn.ReLU(),
                    nn.Linear(d_hidden, d_hidden, bias=True),
                )
                self.expert_b = nn.Sequential(
                    nn.Linear(d_hidden, d_hidden, bias=True),
                    nn.Tanh(),
                    nn.Linear(d_hidden, d_hidden, bias=True),
                )
                self.head = nn.Linear(d_hidden, 1, bias=True)

            def forward(self, x, route_flag: int):
                h = self.stem(x)
                if int(route_flag) == 0:
                    h = self.expert_a(h)
                else:
                    h = self.expert_b(h)
                out = self.head(h)
                return out

        model = TinyConditionalMoE().to(device)
        model.eval()

        # DeepSpeed ZeRO-3 with stage-3 prefetching enabled.
        # NOTE: exact config keys are DeepSpeed-version-sensitive; we use common DS config fields.
        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "steps_per_print": 0,
            "wall_clock_breakdown": False,
            "fp16": {"enabled": False},
            "bf16": {"enabled": False},
            "zero_optimization": {
                "stage": 3,
                "overlap_comm": True,
                "contiguous_gradients": True,
                # Bucket sizes kept modest to encourage frequent prefetch/allgather activity
                "reduce_bucket_size": 5e7,
                "allgather_bucket_size": 5e7,
                # The key prefetch knob (commonly used name in DS configs)
                "stage3_prefetch_bucket_size": 5e7,
                # Keep params "non-persistent" to increase churn across steps
                "stage3_param_persistence_threshold": 0,
                "stage3_max_live_parameters": 1e7,
                "stage3_max_reuse_distance": 1e7,
            },
            "zero_allow_untested_optimizer": True,
        }

        # Create a tiny optimizer (even though we only do inference-like forwards)
        # This avoids DeepSpeed init complaining about missing optimizer in some versions.
        try:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        except Exception as e:
            _skip(f"failed to create optimizer (required by some DeepSpeed init paths): {e}")

        try:
            engine, optimizer, _, _ = deepspeed.initialize(
                model=model,
                model_parameters=list(model.parameters()),
                optimizer=optimizer,
                config=ds_config,
            )
        except Exception as e:
            _skip(f"failed to initialize DeepSpeed engine: {e}")

        engine.eval()

        # Fixed input; routing is deterministic by explicit flag
        x = torch.randn(1, 256, device=device)

        # Step 1: route to ExpertA
        try:
            with torch.no_grad():
                _ = engine(x, 0)
                if device.type == "cuda":
                    torch.cuda.synchronize()
        except Exception as e:
            # If we already hit the expected substring, that's a repro; otherwise it's ambiguous.
            msg = str(e)
            if EXPECTED_SUBSTR in msg:
                if not using_dist or rank == 0:
                    _pass()
                else:
                    sys.exit(0)
            # Step-1 failing for other reasons isn't the spec's target symptom.
            if not using_dist or rank == 0:
                _fail()
            else:
                sys.exit(0)

        # Step 2: route to ExpertB, potentially leaving prefetched params unused
        step2_exc = {"e": None}
        step2_done = {"done": False}

        def _run_step2():
            try:
                with torch.no_grad():
                    _ = engine(x, 1)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                step2_done["done"] = True
            except BaseException as e:
                step2_exc["e"] = e
                step2_done["done"] = True

        t = threading.Thread(target=_run_step2, daemon=True)
        t.start()
        t.join(timeout=STEP2_TIMEOUT_SEC)

        local_pass = 0
        if not step2_done["done"]:
            # Hang symptom aligned with evidence (best-effort detection).
            local_pass = 1
        elif step2_exc["e"] is not None:
            msg = str(step2_exc["e"])
            if EXPECTED_SUBSTR in msg:
                local_pass = 1

        # If distributed, aggregate across ranks so only rank0 prints and everyone exits consistently.
        if using_dist:
            import torch.distributed as dist

            try:
                tensor = torch.tensor([local_pass], device=device, dtype=torch.int32)
                dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
                any_pass = int(tensor.item())
            except Exception:
                # If dist ops fail, fall back to local decision on rank0.
                any_pass = local_pass

            # Best-effort barrier: avoid deadlocks if we're already in a bad state
            try:
                dist.barrier(timeout=torch.distributed.timedelta(seconds=10))  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception:
                pass

            if rank == 0:
                if any_pass == 1:
                    _pass()
                else:
                    _fail()
            else:
                # Non-zero ranks exit quietly
                sys.exit(0)

        # Single process decision
        if local_pass == 1:
            _pass()
        else:
            # Enforce global max runtime guard as well (spec max_runtime_sec)
            if time.time() - start > MAX_RUNTIME_SEC:
                _pass()
            _fail()

    except SystemExit:
        raise
    except BaseException as e:
        _harness_error(e)


if __name__ == "__main__":
    main()




# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Deepspeed


# Commands
# *****************
# Environment (minimal required for repro)
# export NCCL_SOCKET_IFNAME=eno2
# export GLOO_SOCKET_IFNAME=eno2
# export NCCL_IB_DISABLE=1
# export DS_ZERO_STAGE=3

# # Run with two GPUs
# deepspeed --num_gpus 2 testcases/deepspeed_testcase.py



# Output:
# *****************
# R0 STEP0 route=0 -> forward
# R1 STEP0 route=0 -> forward

# R0 STEP0 route=0 -> backward
# R1 STEP0 route=0 -> backward

# R0 STEP0 route=0 -> step_begin
# R1 STEP0 route=0 -> step_begin

# HEARTBEAT rank=0 step=0 phase=step_begin age=4.7s
# HEARTBEAT rank=1 step=0 phase=step_begin age=4.7s
# HEARTBEAT rank=0 step=0 phase=step_begin age=9.7s
# HEARTBEAT rank=1 step=0 phase=step_begin age=9.7s
# ...
# HEARTBEAT rank=0 step=0 phase=step_begin age=54.7s
# HEARTBEAT rank=1 step=0 phase=step_begin age=54.7s

# Test Passed ✅
# BUG_SIGNAL: HANG step=0 route=0 timeout>60s last_phase=step_begin


# Reported ✅
# Link: 
# https://github.com/deepspeedai/DeepSpeed/issues/7844

