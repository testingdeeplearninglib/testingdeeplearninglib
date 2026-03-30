# GCFL-DISTRIBUTE-0014_tc_01

import os
import sys
import time


def _skip(reason: str):
    print(f"SKIP_ENV: {reason}", flush=True)
    sys.exit(0)


def _pass():
    print("Test Passed ✅", flush=True)
    sys.exit(0)


def _fail():
    print("Test Failed ❌", flush=True)
    sys.exit(0)


def _harness_error(e: BaseException):
    print(f"HARNESS_ERROR: {type(e).__name__}: {e}", flush=True)
    sys.exit(1)


def _get_int_env(name: str, default=None):
    v = os.environ.get(name, None)
    if v is None:
        return default
    try:
        return int(str(v).strip())
    except Exception:
        return default


def _barrier_with_timeout(group, timeout_s: float = 20.0):
    import torch
    import torch.distributed as dist

    start = time.time()
    x = torch.ones(1, device=torch.device("cuda"))
    while True:
        try:
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
            return True
        except Exception:
            return False
        if time.time() - start > timeout_s:
            return False


def main():
    try:
        try:
            import torch
        except Exception as e:
            _skip(f"missing torch: {type(e).__name__}: {e}")

        try:
            import deepspeed
            from deepspeed.sequence.layer import DistributedAttention, _SeqAllToAll
        except Exception as e:
            _skip(f"missing deepspeed/DistributedAttention: {type(e).__name__}: {e}")

        import torch.distributed as dist

        rank = _get_int_env("RANK", None)
        world_size = _get_int_env("WORLD_SIZE", None)
        local_rank = _get_int_env("LOCAL_RANK", None)

        if rank is None or world_size is None:
            _skip("not launched with distributed env (missing RANK/WORLD_SIZE); use torchrun --nproc_per_node=2")

        if world_size < 2:
            _skip(f"world_size < 2 (world_size={world_size}); need >=2 ranks")

        if not torch.cuda.is_available():
            _skip("CUDA not available (torch.cuda.is_available() is False)")
        ndev = torch.cuda.device_count()
        if ndev < 2:
            _skip(f"need >=2 CUDA GPUs; found {ndev}")
        if not dist.is_nccl_available():
            _skip("NCCL backend not available (dist.is_nccl_available() is False)")

        if local_rank is None:
            local_rank = rank
        if local_rank < 0 or local_rank >= ndev:
            _skip(f"LOCAL_RANK out of range (LOCAL_RANK={local_rank}, device_count={ndev})")

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        if rank == 0:
            print("STAGE: before_deepspeed_init", flush=True)
            print(f"ENV: rank={rank} world_size={world_size} local_rank={local_rank} ndev={ndev}", flush=True)

        try:
            deepspeed.init_distributed("nccl")
        except Exception as e:
            if rank == 0:
                print(f"STAGE_FAIL: deepspeed_init: {type(e).__name__}: {e}", flush=True)
            _fail()

        if not dist.is_initialized():
            _skip("torch.distributed not initialized after deepspeed.init_distributed")

        group = dist.group.WORLD

        if rank == 0:
            print("REACHED_AFTER_INIT", flush=True)

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        # Parameters
        B = 8
        H = 8
        S = world_size * 2 + 1
        D = 64
        dtype = torch.float32

        if rank == 0:
            print("STAGE: make_inputs", flush=True)

        # Sequence-first layout: [S, B, H, D]
        q_sbh = torch.rand((S, B, H, D), device=device, dtype=dtype)
        k_sbh = torch.rand((S, B, H, D), device=device, dtype=dtype)
        v_sbh = torch.rand((S, B, H, D), device=device, dtype=dtype)

        if rank == 0:
            print("STAGE: broadcast_inputs", flush=True)
        try:
            dist.broadcast(q_sbh, src=0, group=group)
            dist.broadcast(k_sbh, src=0, group=group)
            dist.broadcast(v_sbh, src=0, group=group)
        except Exception as e:
            if rank == 0:
                print(f"STAGE_FAIL: broadcast: {type(e).__name__}: {e}", flush=True)
            _fail()

        # Build DS distributed attention
        if rank == 0:
            print("STAGE: build_distributed_attention", flush=True)

        scatter_idx = 2
        gather_idx = 0

        try:
            dist_attn = DistributedAttention(
                torch.nn.functional.scaled_dot_product_attention,
                group,
                scatter_idx=scatter_idx,
                gather_idx=gather_idx,
            )
        except Exception as e:
            if rank == 0:
                print(f"STAGE_FAIL: ctor: {type(e).__name__}: {e}", flush=True)
            _fail()

        # Run DS
        if rank == 0:
            print("STAGE: run_distributed_attention", flush=True)
        try:
            ds_out_sbh = dist_attn(q_sbh, k_sbh, v_sbh, None, 0.0, False)
        except Exception as e:
            if rank == 0:
                print(f"STAGE_FAIL: ds_call: {type(e).__name__}: {e}", flush=True)
            _fail()

        # DS-equivalent baseline (IMPORTANT)
        # Simulate what DistributedAttention.forward does internally:
        # query_layer = all2all(q, scatter_idx, gather_idx)
        # context_layer = sdpa(query_layer, key_layer, value_layer, *args)
        # output = all2all(context_layer, gather_idx, scatter_idx)
        if rank == 0:
            print("STAGE: baseline_ds_equivalent", flush=True)
        try:
            q_layer = _SeqAllToAll.apply(group, q_sbh, scatter_idx, gather_idx)
            k_layer = _SeqAllToAll.apply(group, k_sbh, scatter_idx, gather_idx)
            v_layer = _SeqAllToAll.apply(group, v_sbh, scatter_idx, gather_idx)

            # local SDPA expects [B, H, S, D], so convert whatever layout q_layer currently is
            # We keep the same permutation logic DS users commonly rely on: treat axis0 as S-like after all2all.
            q_bhsd = q_layer.permute(1, 2, 0, 3).contiguous()
            k_bhsd = k_layer.permute(1, 2, 0, 3).contiguous()
            v_bhsd = v_layer.permute(1, 2, 0, 3).contiguous()

            ctx_bhsd = torch.nn.functional.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, None, 0.0, False
            )
            ctx_layer = ctx_bhsd.permute(2, 0, 1, 3).contiguous()

            ref_out_sbh = _SeqAllToAll.apply(group, ctx_layer, gather_idx, scatter_idx)
        except Exception as e:
            if rank == 0:
                print(f"STAGE_FAIL: baseline_equivalent: {type(e).__name__}: {e}", flush=True)
            _fail()

        ok_bar = _barrier_with_timeout(group, timeout_s=20.0)
        if not ok_bar:
            if rank == 0:
                print("STAGE_FAIL: barrier_timeout", flush=True)
            _fail()

        if dist.get_rank(group=group) == 0:
            print("REF out:", tuple(ref_out_sbh.shape), ref_out_sbh.dtype, ref_out_sbh.device, flush=True)
            print("DS  out:", tuple(ds_out_sbh.shape), ds_out_sbh.dtype, ds_out_sbh.device, flush=True)

            diff = (ds_out_sbh - ref_out_sbh).abs()
            print("MAX_ABS_DIFF:", float(diff.max().item()), flush=True)
            print("MEAN_ABS_DIFF:", float(diff.mean().item()), flush=True)

            ok = bool(torch.allclose(ds_out_sbh, ref_out_sbh, rtol=1e-4, atol=1e-4))
            print("ALLCLOSE:", ok, flush=True)

            # Oracle: mismatch => bug reproduced
            if not ok:
                _pass()
            else:
                _fail()
        else:
            sys.exit(0)

    except SystemExit:
        raise
    except Exception as e:
        _harness_error(e)


if __name__ == "__main__":
    main()




# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Deepspeed


# Commands
# *****************
# conda activate ds_testcase
# PYTHONUNBUFFERED=1 torchrun --nproc_per_node=2 deepspeed_testcase.py



# Output:
# *****************
# (ds_testcase) talha@bitse-SYS-7048GR-TR:~/dl_testing$ conda activate ds_testcase
# PYTHONUNBUFFERED=1 torchrun --nproc_per_node=2 /home/talha/dl_testing/testcases/deepspeed_testcase.py
# [2026-02-10 17:33:52,415] torch.distributed.run: [WARNING] 
# [2026-02-10 17:33:52,415] torch.distributed.run: [WARNING] *****************************************
# [2026-02-10 17:33:52,415] torch.distributed.run: [WARNING] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
# [2026-02-10 17:33:52,415] torch.distributed.run: [WARNING] *****************************************
# [2026-02-10 17:33:54,653] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
# [2026-02-10 17:33:54,750] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
#  [WARNING]  async_io requires the dev libaio .so object and headers but these were not found.
#  [WARNING]  async_io: please install the libaio-dev package with apt
#  [WARNING]  If libaio is already installed (perhaps from source), try setting the CFLAGS and LDFLAGS environment variables to where it can be found.
#  [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
#  [WARNING]  async_io requires the dev libaio .so object and headers but these were not found.
#  [WARNING]  async_io: please install the libaio-dev package with apt
#  [WARNING]  If libaio is already installed (perhaps from source), try setting the CFLAGS and LDFLAGS environment variables to where it can be found.
#  [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
#  [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.2
#  [WARNING]  using untested triton version (2.2.0), only 1.0.0 is known to be compatible
#  [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.2
#  [WARNING]  using untested triton version (2.2.0), only 1.0.0 is known to be compatible
# STAGE: before_deepspeed_init
# ENV: rank=0 world_size=2 local_rank=0 ndev=4
# [2026-02-10 17:33:55,462] [INFO] [comm.py:637:init_distributed] cdb=None
# [2026-02-10 17:33:55,462] [INFO] [comm.py:668:init_distributed] Initializing TorchBackend in DeepSpeed with backend nccl
# REACHED_AFTER_INIT
# STAGE: make_inputs
# STAGE: broadcast_inputs
# [2026-02-10 17:33:55,764] [INFO] [comm.py:637:init_distributed] cdb=None
# STAGE: build_distributed_attention
# STAGE: run_distributed_attention
# STAGE: baseline_ds_equivalent
# REF out: (5, 8, 8, 64) torch.float32 cuda:0
# DS  out: (5, 8, 8, 64) torch.float32 cuda:0
# MAX_ABS_DIFF: 0.6803791522979736
# MEAN_ABS_DIFF: 0.13978055119514465
# ALLCLOSE: False
# Test Passed ✅


# ******************************************************************************

# Reported ✅
# Link: 
# https://github.com/deepspeedai/DeepSpeed/issues/7842