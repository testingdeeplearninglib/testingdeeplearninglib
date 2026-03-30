# GCFL-DISTRIBUTE-0017


import os
import sys
import json
import traceback
import random


def _skip(msg: str):
    print(f"SKIP_ENV: {msg}", flush=True)
    sys.exit(0)


def env_int(k: str, d: int) -> int:
    v = os.environ.get(k, "").strip()
    try:
        return int(v) if v else d
    except Exception:
        return d


def env_bool(k: str, d: bool) -> bool:
    v = os.environ.get(k, "").strip().lower()
    if not v:
        return d
    return v in ("1", "true", "yes", "y", "on")


def main():
    # --- imports ---
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.distributed as dist
    except Exception as e:
        _skip(f"missing torch/dist: {type(e).__name__}: {e}")

    try:
        import deepspeed
        from deepspeed import zero
    except Exception as e:
        _skip(f"missing deepspeed: {type(e).__name__}: {e}")

    # --- distributed init ---
    if not dist.is_available():
        _skip("torch.distributed not available")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world = dist.get_world_size()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    def barrier():
        # monitored_barrier is GLOO-only; DO NOT use it on NCCL
        dist.barrier()

    # --- config / knobs ---
    ds_config = os.environ.get("DEEPSPEED_CONFIG", "ds_config_zero3_stress.json")

    VOCAB = env_int("VOCAB", 32768)
    D_MODEL = env_int("D_MODEL", 2048)
    ITERS = env_int("ITERS", 200)
    TILE = env_int("TILE", 8)
    DO_BWD = env_bool("DO_BWD", True)
    FORCE_EDIT = env_bool("FORCE_GATHER_EDIT", True)

    # DS batch rule sanity
    try:
        cfg = json.load(open(ds_config, "r", encoding="utf-8"))
    except Exception as e:
        _skip(f"cannot load {ds_config}: {type(e).__name__}: {e}")

    micro = int(cfg.get("train_micro_batch_size_per_gpu", 1))
    gas = int(cfg.get("gradient_accumulation_steps", 1))
    tbs = int(cfg.get("train_batch_size", micro * gas * world))
    if tbs != micro * gas * world:
        _skip(f"bad DS batch params: train_batch_size {tbs} != micro {micro} * gas {gas} * world {world}")

    # --- tiny model: emb + two 2D Linear weights ---
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(VOCAB, D_MODEL)
            self.l1 = nn.Linear(D_MODEL, 4 * D_MODEL, bias=False)
            self.l2 = nn.Linear(4 * D_MODEL, D_MODEL, bias=False)

        def forward(self, x):
            h = self.emb(x)  # [B,S,D]
            h = self.l1(h)
            h = F.gelu(h)
            h = self.l2(h)
            return h.sum()

    model = M().to(device).train()

    # --- deepspeed init ---
    try:
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=[p for p in model.parameters() if p.requires_grad],
            config=ds_config,
        )
    except Exception as e:
        _skip(f"deepspeed.initialize failed: {type(e).__name__}: {e}")

    wrapped = getattr(engine, "module", engine)

    # gather multiple 2D params together (this resembles the stress trigger)
    p_emb = wrapped.emb.weight
    p_l1 = wrapped.l1.weight
    p_l2 = wrapped.l2.weight
    params = [p_emb, p_l1, p_l2]

    if rank == 0:
        print(
            f"[rank0] CONFIG world={world} device={device} vocab={VOCAB} d_model={D_MODEL} "
            f"iters={ITERS} tile={TILE} do_bwd={DO_BWD} force_edit={FORCE_EDIT}",
            flush=True,
        )

    rnd = random.Random(2026 + rank)
    any_exc = None

    try:
        for i in range(1, ITERS + 1):
            x = torch.randint(0, VOCAB, (1, 8), device=device)

            # IMPORTANT: gather multiple params together
            with zero.GatheredParameters(params, modifier_rank=None):
                # in-place slice edits while gathered
                for p in params:
                    t = p.data
                    r = rnd.randrange(0, t.shape[0])
                    if t.ndim >= 2:
                        cmax = max(int(t.shape[1]) - TILE, 1)
                        c0 = rnd.randrange(0, cmax)
                        _ = t[r, c0 : c0 + TILE].sum().item()
                        if FORCE_EDIT:
                            # no-op edit; still "writes"
                            t[r, c0 : c0 + TILE].add_(0.0)

            # tick fwd/bwd path
            loss = engine(x)
            if DO_BWD:
                engine.backward(loss)
                engine.step()
            else:
                engine.zero_grad()

            if rank == 0 and (i % 25 == 0 or i == ITERS):
                print(f"[rank0] step={i} ok", flush=True)

    except Exception as e:
        any_exc = f"{type(e).__name__}: {e}"
        if rank == 0:
            print("[rank0] EXC TRACEBACK:", flush=True)
            traceback.print_exc()

    # --- synchronize decision across ranks ---
    flag = torch.tensor([1 if any_exc else 0], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    hit = bool(int(flag.item()) == 1)

    if rank == 0:
        print(f"[rank0] RESULT hit={hit} exc={any_exc}", flush=True)

    decision = torch.tensor([1 if hit else 0], device=device, dtype=torch.int32)
    dist.broadcast(decision, src=0)
    barrier()

    # === DOCUMENTATION SEMANTICS ===
    # hit==True -> BUG/SUSPICIOUS TRIGGERED -> Test Passed ✅
    # hit==False -> NO BUG -> Test Failed ❌
    if int(decision.item()) == 1:
        if rank == 0:
            print("Test Passed ✅", flush=True)
    else:
        if rank == 0:
            print("Test Failed ❌", flush=True)

    barrier()
    try:
        dist.destroy_process_group()
    except Exception:
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()


# ******************************************************************************
#                                    Result 
# ******************************************************************************

# DeepSpeed


# Commands
# *****************
# cd ~/dl_testing/testcases

# python -m py_compile ds_zero3_gather_assert_tri.py && echo "OK: py_compile" || echo "BAD: py_compile failed"

# truly suspicious shell junk tokens
# grep -nE '(^|[[:space:]])(echo|PYint\(|WROTE|<<|>>>|cat >|tee |BEGIN:VEVENT)([[:space:]]|$)' ds_zero3_gather_assert_tri.py \
#   && echo "BAD: shell junk detected" || echo "OK: no shell junk"


# trigger command
# *****************
# export CUDA_VISIBLE_DEVICES=0,1
# export NCCL_DEBUG=WARN
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export DEEPSPEED_CONFIG=ds_config_zero3_stress.json

# export VOCAB=32768
# export D_MODEL=2048
# export ITERS=200
# export TILE=8
# export DO_BWD=1
# export FORCE_GATHER_EDIT=1

# torchrun --nproc_per_node=2 ds_zero3_gather_assert_tri.py 2>&1 | tee FINAL_out_gather_tri.log

# # quick proof lines
# grep -n "AssertionError" FINAL_out_gather_tri.log | head -n 5
# grep -n "\[rank0\] RESULT" FINAL_out_gather_tri.log | tail -n 5
# grep -n "Test Passed ✅\|Test Failed ❌" FINAL_out_gather_tri.log | tail -n 3




# Output:
# *****************

# [rank0] CONFIG world=2 device=cuda:0 vocab=32768 d_model=2048 iters=200 tile=8 do_bwd=True force_edit=True
# [rank0] EXC TRACEBACK:
# Traceback (most recent call last):
#   File "/home/talha/dl_testing/testcases/ds_zero3_gather_assert_tri.py", line 125, in main
#     with zero.GatheredParameters(params, modifier_rank=None):
#   File "/home/talha/.venvs/dl_testing/lib/python3.12/site-packages/deepspeed/runtime/zero/partition_parameters.py", line 2344, in __exit__
#     self.params[0].partition(param_list=self.params, has_been_updated=False)
#   File "/home/talha/.venvs/dl_testing/lib/python3.12/site-packages/deepspeed/runtime/zero/partition_parameters.py", line 1487, in partition
#     self._partition(param_list, has_been_updated=has_been_updated, free_data=True)
#   File "/home/talha/.venvs/dl_testing/lib/python3.12/site-packages/deepspeed/runtime/zero/partition_parameters.py", line 1636, in _partition
#     self._partition_param(param, has_been_updated=has_been_updated, free_data=True)
#   File "/home/talha/.venvs/dl_testing/lib/python3.12/site-packages/deepspeed/runtime/zero/partition_parameters.py", line 1670, in _partition_param
#     free_param(param)
#   File "/home/talha/.venvs/dl_testing/lib/python3.12/site-packages/deepspeed/runtime/zero/partition_parameters.py", line 302, in free_param
#     assert not param.ds_active_sub_modules, param.ds_summary()
# AssertionError: {'id': 0, 'status': 'AVAILABLE', 'numel': 67108864, 'ds_numel': 67108864, 'shape': (32768, 2048), 'ds_shape': (32768, 2048), 'requires_grad': True, 'grad_shape': None, 'persist': False, 'active_sub_modules': {1}, 'ds_tensor.shape': torch.Size([33554432])}

# [rank0] RESULT hit=True exc=AssertionError: {'id': 0, 'status': 'AVAILABLE', 'numel': 67108864, 'ds_numel': 67108864, 'shape': (32768, 2048), 'ds_shape': (32768, 2048), 'requires_grad': True, 'grad_shape': None, 'persist': False, 'active_sub_modules': {1}, 'ds_tensor.shape': torch.Size([33554432])}

# Test Passed ✅




# ******************************************************************************

# Reported ✅
# Link: 
# https://github.com/deepspeedai/DeepSpeed/issues/7811
