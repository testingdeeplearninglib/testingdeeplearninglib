# GCFL-AUTOGRADBA-0021

import os
import sys
import time
import random

SEED = 2021
MAX_RUNTIME_SEC = 180


def _print_skip(reason: str) -> None:
    print(f"SKIP_ENV: {reason}")
    sys.exit(0)


def _print_pass() -> None:
    print("Test Passed ✅")
    sys.exit(0)


def _print_fail() -> None:
    print("Test Failed ❌")
    sys.exit(0)


def _harness_error(exc: BaseException) -> None:
    msg = f"{type(exc).__name__}: {exc}"
    print(f"HARNESS_ERROR: {msg}")
    sys.exit(1)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        pass


def _get_int_env(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default

def init_dist_if_needed() -> None:
    """
    Fix:
    - Do NOT overwrite RANK/WORLD_SIZE/LOCAL_RANK when launched by DeepSpeed.
    - Initialize DeepSpeed comm backend explicitly via deepspeed.init_distributed().
    - For single-process runs, init a tiny local process group.
    """
    import socket
    import torch  # type: ignore
    import torch.distributed as dist  # type: ignore
    import deepspeed  # type: ignore

    if not dist.is_available():
        raise RuntimeError("torch.distributed not available")

    # If already initialized, nothing to do
    if dist.is_initialized():
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    # Ensure addr/port exist; DeepSpeed launcher sets these normally.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")

    if "MASTER_PORT" not in os.environ:
        # pick a free port for single-proc fallback
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        os.environ["MASTER_PORT"] = str(s.getsockname()[1])
        s.close()

    backend = "nccl" if torch.cuda.is_available() else "gloo"

    # IMPORTANT: initialize via DeepSpeed (this sets DS comm backend properly)
    try:
        deepspeed.init_distributed(dist_backend=backend)
    except Exception as e:
        raise RuntimeError(f"deepspeed.init_distributed failed: {e}") from e

    # Safety: if DS didn’t initialize torch.distributed for some reason, do it ourselves
    if not dist.is_initialized():
        try:
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        except Exception as e:
            raise RuntimeError(f"torch.distributed init failed: {e}") from e


def build_pipeline_model():
    """
    Build a PipelineModule with:
      stage0: embedding
      stage1: small compute -> logits -> swapaxes
      stage2: loss
    """
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    from deepspeed.pipe import PipelineModule, LayerSpec  # type: ignore

    vocab = 128
    hidden = 32
    seq_len = 8
    batch = 2
    num_classes = 16

    class EmbedStage(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)

        def forward(self, tokens):
            return self.embed(tokens)

    class ComputeStage(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(hidden, num_classes)

        def forward(self, h):
            logits = self.proj(h)               # [B,S,C]
            logits = torch.swapaxes(logits, 1, 2)  # [B,C,S]
            return logits

    class LossStage(nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_fn = nn.CrossEntropyLoss()

        def forward(self, logits_and_labels):
            logits, labels = logits_and_labels
            return self.loss_fn(logits, labels)

    layers = [
        LayerSpec(EmbedStage),
        LayerSpec(ComputeStage),
        LayerSpec(LossStage),
    ]

    model = PipelineModule(
        layers=layers,
        loss_fn=None,  # loss is an explicit stage
        num_stages=3,
        partition_method="parameters",
        activation_checkpoint_interval=0,
    )

    # IMPORTANT:
    # In pipeline parallelism, inputs should be on the first stage device.
    # Safest for this synthetic case: keep on CPU and let DS handle transfers
    # (DS pipeline can move microbatches internally). If your DS version
    # requires GPU input, we handle that in run_one_step().
    tokens = torch.randint(0, vocab, (batch, seq_len), dtype=torch.long)
    labels = torch.randint(0, num_classes, (batch, seq_len), dtype=torch.long)

    return model, (tokens, labels)


def run_one_step(pipe_model, tokens, labels, activation_checkpoint_interval: int):
    import torch  # type: ignore
    import deepspeed  # type: ignore

    local_rank = _get_int_env("LOCAL_RANK", 0)

    # Rank-aware device (important)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        first_stage_device = torch.device("cuda:0")
    else:
        first_stage_device = torch.device("cpu")

    # Put the *input* on first stage device (cuda:0 typically).
    # This avoids per-rank device mismatch issues.
    tokens = tokens.to(first_stage_device, non_blocking=True)
    labels = labels.to(first_stage_device, non_blocking=True)

    ds_config = {
        "train_batch_size": 1,
        "train_micro_batch_size_per_gpu": 1,
        "steps_per_print": 999999,
        "zero_optimization": {"stage": 0},
        "fp16": {"enabled": False},
        "bf16": {"enabled": False},
        "pipeline": {
            "seed_layers": True,
            "activation_checkpoint_interval": int(activation_checkpoint_interval),
        },
        "optimizer": {"type": "Adam", "params": {"lr": 1e-3}},
    }

    class _Args:
        pass

    args = _Args()
    args.local_rank = local_rank

    try:
        engine, _, _, _ = deepspeed.initialize(
            args=_Args,
            model=pipe_model,
            model_parameters=[p for p in pipe_model.parameters() if p.requires_grad],
            config=ds_config,
        )

        def data_iter():
            # In pipeline mode, train_batch expects an iterator yielding batches.
            yield (tokens, labels)

        engine.train()
        engine.train_batch(data_iter=data_iter)
        return True, None

    except Exception as e:
        return False, e


def main():
    start = time.time()

    try:
        # Dependency guards
        try:
            import torch  # type: ignore
        except Exception as e:
            _print_skip(f"torch not installed ({e})")

        try:
            import deepspeed  # type: ignore
        except Exception as e:
            _print_skip(f"deepspeed not installed ({e})")

        set_determinism(SEED)

        # Correct distributed init
        try:
            init_dist_if_needed()
        except Exception as e:
            _print_skip(str(e))

        # Build pipeline model + data
        pipe_model, (tokens, labels) = build_pipeline_model()

        # Case A (control): activation_checkpoint_interval=0
        ok_a, exc_a = run_one_step(pipe_model, tokens, labels, activation_checkpoint_interval=0)

        if time.time() - start > MAX_RUNTIME_SEC:
            _print_fail()

        # Rebuild fresh for Case B
        set_determinism(SEED)
        pipe_model_b, (tokens_b, labels_b) = build_pipeline_model()

        # Case B (trigger): activation_checkpoint_interval=1
        ok_b, exc_b = run_one_step(pipe_model_b, tokens_b, labels_b, activation_checkpoint_interval=1)

        expected_substr = "element 0 of tensors does not require grad and does not have a grad_fn"

        a_has_expected = (exc_a is not None) and (expected_substr in str(exc_a))
        b_has_expected = (exc_b is not None) and (expected_substr in str(exc_b))

        if ok_a and (not a_has_expected) and (not ok_b) and b_has_expected:
            _print_pass()
        else:
            _print_fail()

    except SystemExit:
        raise
    except Exception as e:
        _harness_error(e)
    finally:
        # Clean shutdown (prevents NCCL resource leak warnings)
        try:
            import torch.distributed as dist  # type: ignore
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    main()



# ******************************************************************************
#                                    Result 
# ******************************************************************************

# DeepSpeed

# Output:
# [2026-01-14 17:05:14,004] [WARNING] [runner.py:232:fetch_hostfile] Unable to find hostfile, will proceed with training with local resources only.
# [2026-01-14 17:05:14,005] [INFO] [runner.py:630:main] cmd = /home/talha/.venvs/dl_testing/bin/python -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMCwgMSwgMl19 --master_addr=127.0.0.1 --master_port=29620 --enable_each_rank_log=None --log_level=info testing.py
# [2026-01-14 17:05:18,698] [INFO] [launch.py:162:main] WORLD INFO DICT: {'localhost': [0, 1, 2]}
# [2026-01-14 17:05:18,698] [INFO] [launch.py:168:main] nnodes=1, num_local_procs=3, node_rank=0
# [2026-01-14 17:05:18,699] [INFO] [launch.py:179:main] global_rank_mapping=defaultdict(<class 'list'>, {'localhost': [0, 1, 2]})
# [2026-01-14 17:05:18,699] [INFO] [launch.py:180:main] dist_world_size=3
# [2026-01-14 17:05:18,699] [INFO] [launch.py:184:main] Setting CUDA_VISIBLE_DEVICES=0,1,2
# [2026-01-14 17:05:18,700] [INFO] [launch.py:272:main] process 3050839 spawned with command: ['/home/talha/.venvs/dl_testing/bin/python', '-u', 'testing.py', '--local_rank=0']
# [2026-01-14 17:05:18,700] [INFO] [launch.py:272:main] process 3050840 spawned with command: ['/home/talha/.venvs/dl_testing/bin/python', '-u', 'testing.py', '--local_rank=1']
# [2026-01-14 17:05:18,701] [INFO] [launch.py:272:main] process 3050841 spawned with command: ['/home/talha/.venvs/dl_testing/bin/python', '-u', 'testing.py', '--local_rank=2']
# SEED_LAYERS=False BASE_SEED=1234 SEED_FN=None
# Using topology: {ProcessCoord(pipe=0, data=0): 0, ProcessCoord(pipe=1, data=0): 1, ProcessCoord(pipe=2, data=0): 2}
# stage=0 layers=1
#      0: EmbedStage
# stage=1 layers=1
#      1: ComputeStage
# stage=2 layers=1
#      2: LossStage
# Test Failed ❌
# /home/talha/.venvs/dl_testing/lib/python3.12/site-packages/torch/utils/cpp_extension.py:2059: UserWarning: TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation. 
# If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'].
#   warnings.warn(
# [2026-01-14 17:05:26,702] [INFO] [launch.py:367:main] Process 3050841 exits successfully.