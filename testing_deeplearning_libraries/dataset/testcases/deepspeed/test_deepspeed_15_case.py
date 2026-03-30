# GCFL-OTHER-0015

#!/usr/bin/env python3
# Reproducer: DeepSpeed ZeroDivisionError in engine.step() when steps_per_print == 0
#
# Environment (from your run):
#   Python 3.12.3
#   deepspeed 0.18.5
#   torch 2.5.1+cu121
#   2 GPUs
#
# Symptom:
#   Rank0 crashes with:
#     ZeroDivisionError: integer modulo by zero
#   in:
#     deepspeed/runtime/engine.py  (_take_model_step) modulo by steps_per_print()
#
# Notes:
#   This file supports:
#     - controller mode: runs deepspeed launcher
#     - worker mode: actual distributed training step that triggers the crash

import argparse
import json
import os
import subprocess
import sys
import time
import traceback


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, default)).strip())
    except Exception:
        return default


def _print_env_banner(args):
    print(f"PY_EXE: {sys.executable}")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME')}")
    print(f"CUDA_PATH: {os.environ.get('CUDA_PATH')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
    # show only a "head" of PATH to avoid huge noise
    path = os.environ.get("PATH", "")
    print(f"PATH_HEAD: {':'.join(path.split(':')[:5])}")

    try:
        import torch  # noqa
        print(f"TORCH_VER: {getattr(torch, '__version__', 'unknown')}")
        print(f"CUDA_AVAILABLE: {bool(torch.cuda.is_available())}")
    except Exception as e:
        print(f"TORCH_IMPORT_ERROR: {type(e).__name__}: {e}")

    try:
        import deepspeed  # noqa
        print(f"DEEPSPEED_VER: {getattr(deepspeed, '__version__', 'unknown')}")
    except Exception as e:
        print(f"DEEPSPEED_IMPORT_ERROR: {type(e).__name__}: {e}")

    print(f"GCFL_NUM_GPUS: {args.num_gpus}")
    print(f"GCFL_REPEATS: {args.repeats}")
    print(f"GCFL_TIMEOUT_SEC: {args.timeout_sec}")
    print(f"GCFL_SEED: {args.seed}")
    print(f"GCFL_MICRO_STEPS: {args.micro_steps}")
    print(f"GCFL_HIDDEN: {args.hidden}")


def _write_worker_json(ok: bool, rank: int, error: str = "", tb: str = ""):
    payload = {"ok": bool(ok), "rank": int(rank)}
    if not ok:
        payload["error"] = str(error)
        payload["traceback"] = str(tb)
    print("WORKER_JSON:", json.dumps(payload))


def worker_main(args) -> int:
    """
    Runs inside each distributed rank.
    Expected trigger: DeepSpeed Engine step() does modulo by steps_per_print(),
    but steps_per_print() becomes 0 -> ZeroDivisionError.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.distributed as dist
        import deepspeed
    except Exception as e:
        print(f"WORKER_IMPORT_ERROR: {type(e).__name__}: {e}")
        return 2

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank is not None else 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(f"LOCAL_RANK: {local_rank}")
    print(f"WORLD_SIZE: {world_size}")
    print(f"overlap_comm: {args.overlap_comm}")
    print(f"seed: {args.seed}")
    print(f"micro_steps: {args.micro_steps}")
    print(f"hidden: {args.hidden}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(local_rank)

    # Minimal model + loss
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    model = nn.Sequential(
        nn.Linear(args.hidden, args.hidden, bias=False),
        nn.ReLU(),
        nn.Linear(args.hidden, args.hidden, bias=False),
    ).to(device)

    # ---- KEY CONFIG: steps_per_print = 0 triggers DS modulo-by-zero path ----
    # This reproduces the exact stack you saw:
    #   engine.step() -> _take_model_step -> (global_steps+1) % steps_per_print()
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "fp16": {"enabled": False},
        "bf16": {"enabled": False},
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": bool(args.overlap_comm),
            "contiguous_gradients": True,
        },
        # Intentionally invalid / edge config:
        # DeepSpeed should validate this, but currently it can reach a modulo by zero.
        "steps_per_print": 0,
        "wall_clock_breakdown": False,
    }

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    try:
        engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=ds_config,
        )

        # One forward/backward/step
        x = torch.randn(1, args.hidden, device=device)
        y = torch.randn(1, args.hidden, device=device)

        loss = F.mse_loss(engine(x), y)
        engine.backward(loss)

        # Crash occurs here on rank0:
        engine.step()

        _write_worker_json(ok=True, rank=local_rank)
        # clean exit
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        return 0

    except Exception as e:
        tb = traceback.format_exc()
        # Label rank exception clearly in logs
        print(f"WORKER_RANK: {local_rank} EXCEPTION: {type(e).__name__}: {e}")
        print("WORKER_TRACEBACK_BEGIN")
        print(tb.rstrip())
        print("WORKER_TRACEBACK_END")
        _write_worker_json(ok=False, rank=local_rank, error=f"{type(e).__name__}: {e}", tb=tb)

        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        return 1


def controller_main(args) -> int:
    """
    Controller launches deepspeed. Captures log to file, prints summary result.
    """
    os.makedirs(args.logs_dir, exist_ok=True)

    script_path = os.path.abspath(__file__)
    log_file = os.path.join(args.logs_dir, f"ds_run01_overlap{int(args.overlap_comm)}.log")

    cmd = [
        args.deepspeed_bin,
        f"--num_gpus={args.num_gpus}",
        script_path,
        "--worker",
        "--overlap_comm",
        str(int(args.overlap_comm)),
        "--seed",
        str(args.seed),
        "--micro_steps",
        str(args.micro_steps),
        "--hidden",
        str(args.hidden),
        "--dist_timeout_sec",
        str(args.dist_timeout_sec),
    ]

    # Make controller deterministic and avoid “random” env surprises.
    env = os.environ.copy()
    env.setdefault("NCCL_IB_DISABLE", "1")        # socket-only
    env.setdefault("NCCL_SOCKET_IFNAME", "eno2")  # your NIC
    env.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("NCCL_DEBUG", "INFO")

    t0 = time.time()
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("ENV: " + " ".join(
            f"{k}={env.get(k)}" for k in [
                "TORCH_NCCL_BLOCKING_WAIT",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING",
                "NCCL_DEBUG",
                "NCCL_IB_DISABLE",
                "NCCL_SOCKET_IFNAME",
            ]
        ) + "\n")
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()

        try:
            p = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=args.timeout_sec,
                text=True,
            )
            rc = int(p.returncode)
        except subprocess.TimeoutExpired:
            rc = 124
            f.write("\nCONTROLLER_TIMEOUT\n")
        except Exception as e:
            rc = 125
            f.write(f"\nCONTROLLER_EXCEPTION: {type(e).__name__}: {e}\n")

    dt = time.time() - t0
    print(f"LOG_FILE: {log_file}")
    print(f"LAUNCH_RC: {rc}")
    print(f"RUN_TIME_SEC: {dt:.3f}")

    if rc != 0:
        print(f"FAIL: deepspeed returned rc={rc}. See log: {log_file}")
        print("Test Failed ❌")
        return 1

    print("Test Passed ✅")
    return 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true", help="Run as deepspeed worker rank")
    p.add_argument("--local_rank", type=int, default=None)
    p.add_argument("--overlap_comm", type=int, default=_env_int("GCFL_OVERLAP_COMM", 1))
    p.add_argument("--seed", type=int, default=_env_int("GCFL_SEED", 2021))
    p.add_argument("--micro_steps", type=int, default=_env_int("GCFL_MICRO_STEPS", 30))
    p.add_argument("--hidden", type=int, default=_env_int("GCFL_HIDDEN", 4096))

    # controller settings
    p.add_argument("--num_gpus", type=int, default=_env_int("GCFL_NUM_GPUS", 2))
    p.add_argument("--repeats", type=int, default=_env_int("GCFL_REPEATS", 1))
    p.add_argument("--timeout_sec", type=int, default=_env_int("GCFL_TIMEOUT_SEC", 300))
    p.add_argument("--dist_timeout_sec", type=int, default=_env_int("GCFL_DIST_TIMEOUT_SEC", 120))
    p.add_argument("--logs_dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "logs"))

    # explicit deepspeed bin to avoid PATH surprises
    p.add_argument("--deepspeed_bin", type=str, default=os.path.join(os.path.dirname(sys.executable), "deepspeed"))

    return p.parse_args()


def main():
    args = parse_args()

    if args.worker:
        return worker_main(args)

    _print_env_banner(args)

    # Repeat loop (default 1 for docs)
    overall_ok = True
    for i in range(args.repeats):
        print(f"\nRUN {i+1}/{args.repeats}: {args.deepspeed_bin} --num_gpus={args.num_gpus} {os.path.abspath(__file__)} --worker --overlap_comm {int(args.overlap_comm)} --seed {args.seed} --micro_steps {args.micro_steps} --hidden {args.hidden} --dist_timeout_sec {args.dist_timeout_sec}")
        rc = controller_main(args)
        if rc != 0:
            overall_ok = False
            break

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())





# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Deepspeed


# Commands
# *****************
# source ~/.venvs/ds_py312_clean/bin/activate
# cd ~/dl_testing
# unset CUDA_VISIBLE_DEVICES
# unset LD_LIBRARY_PATH   # baseline
# export GCFL_REPEATS=1
# export GCFL_TIMEOUT_SEC=300



# Output:
# *****************
# PY_EXE: /home/talha/.venvs/ds_py312_clean/bin/python
# CUDA_HOME: None
# CUDA_PATH: None
# LD_LIBRARY_PATH: None
# TORCH_VER: 2.5.1+cu121
# CUDA_AVAILABLE: True
# DEEPSPEED_VER: 0.18.5
# GCFL_NUM_GPUS: 2
# GCFL_REPEATS: 1
# GCFL_TIMEOUT_SEC: 300
# GCFL_SEED: 2021
# GCFL_MICRO_STEPS: 30
# GCFL_HIDDEN: 4096

# RUN 1/1: /home/talha/.venvs/ds_py312_clean/bin/deepspeed --num_gpus=2 ... --worker --overlap_comm 1 ...
# LOG_FILE: /home/talha/dl_testing/logs/ds_run01_overlap1.log
# LAUNCH_RC: 1
# RUN_TIME_SEC: 19.096
# FAIL: deepspeed returned rc=1. See log: /home/talha/dl_testing/logs/ds_run01_overlap1.log
# Test Failed ❌
