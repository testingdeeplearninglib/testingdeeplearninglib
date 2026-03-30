# GCFL-DISTRIBUTE-0016

import os
import sys
import random


def _p(msg: str):
    print(msg, flush=True)


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    try:
        return int(v) if v != "" else default
    except Exception:
        return default


def _msg_is_expected_dtype_mismatch(msg: str) -> bool:
    m = msg.lower()
    return (
        "must have the same type" in m
        or "same dtype" in m
        or "expected scalar type" in m
    )


def _cleanup_dist():
    try:
        import torch
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            try:
                if dist.get_backend() == "nccl" and torch.cuda.is_available():
                    lr = _env_int("LOCAL_RANK", 0)
                    dist.barrier(device_ids=[lr])
                else:
                    dist.barrier()
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception:
                pass
    except Exception:
        pass


def main() -> int:
    random.seed(1234)

    try:
        import torch
        import torch.distributed as dist
    except Exception as e:
        _p(f"SKIP_ENV: torch import failed: {e}")
        return 0

    world_size = _env_int("WORLD_SIZE", 1)
    if world_size < 2:
        _p("SKIP_ENV: WORLD_SIZE < 2")
        return 0

    backend = "nccl" if torch.cuda.is_available() else "gloo"

    try:
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
    except Exception as e:
        _p(f"SKIP_ENV: init_process_group failed: {e}")
        return 0

    rank = dist.get_rank()

    if backend == "nccl":
        try:
            torch.cuda.set_device(_env_int("LOCAL_RANK", 0))
        except Exception as e:
            _p(f"SKIP_ENV: set_device failed: {e}")
            return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        _ = torch.empty(1, device=device, dtype=torch.bfloat16)
    except Exception as e:
        _p(f"SKIP_ENV: bf16 unsupported: {e}")
        return 0

    inp = torch.arange(16, device=device, dtype=torch.uint8).reshape(1, 16)
    out = torch.empty((world_size, 1, 16), device=device, dtype=torch.bfloat16)

    try:
        dist.all_gather_into_tensor(out, inp)
        _p(f"RANK {rank}: SUSPICIOUS — dtype mismatch accepted")
        _p("Test Passed ✅")
        return 0

    except (TypeError, RuntimeError) as e:
        _p(f"RANK {rank}: error -> {type(e).__name__}: {e}")
        if _msg_is_expected_dtype_mismatch(str(e)):
            _p("Test Failed ❌")
        else:
            _p("Test Passed ✅")
        return 0

    except Exception as e:
        _p(f"RANK {rank}: SUSPICIOUS — unknown exception -> {e}")
        _p("Test Passed ✅")
        return 0


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        _cleanup_dist()
    sys.exit(rc)


# ******************************************************************************
#                                    Result 
# ******************************************************************************

# DeepSpeed


# Commands
# *****************
# source ~/.venvs/dl_testing/bin/activate
# cd ~/dl_testing/testcases

# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=INFO

# deepspeed --master_port 29620 --num_gpus=2 deepspeed_testcase.py


# Output:
# *****************
# RANK 1: error -> TypeError: output tensor must have the same type as input tensor
# Test Failed ❌
# RANK 0: error -> TypeError: output tensor must have the same type as input tensor
# Test Failed ❌

