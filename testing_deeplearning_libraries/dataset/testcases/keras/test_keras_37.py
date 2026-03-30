# GCFL-OTHER-0037 --- pytorch version
# legitimate PyTorch/Inductor issue

import os
import sys
import random
import hashlib
import traceback
from pathlib import Path


def _skip(reason: str) -> None:
    print(f"SKIP_ENV: {reason}")
    sys.exit(0)


def _pass() -> None:
    print("Test Passed ✅")
    sys.exit(0)


def _fail() -> None:
    print("Test Failed ❌")
    sys.exit(0)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "")
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _print_script_fingerprint() -> None:
    try:
        p = Path(__file__).resolve()
        print(f"SCRIPT_PATH: {p}")
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"SCRIPT_SHA256: {h}")
    except Exception:
        pass


def _safe_imports():
    try:
        import numpy as np
    except Exception as e:
        _skip(f"missing numpy ({e})")

    try:
        import torch
    except Exception as e:
        _skip(f"missing torch ({e})")

    try:
        import keras
    except Exception as e:
        _skip(f"missing keras ({e})")

    return np, torch, keras


def _resolve_keras_backend(keras):
    try:
        return keras.backend.backend()
    except Exception:
        try:
            return keras.config.backend()
        except Exception:
            return None


def main() -> None:
    os.environ.setdefault("KERAS_BACKEND", "torch")
    _print_script_fingerprint()

    np, torch, keras = _safe_imports()

    # Disable traceback filtering to show real failure sites.
    try:
        if hasattr(keras, "config") and hasattr(keras.config, "disable_traceback_filtering"):
            keras.config.disable_traceback_filtering()
            print("DEBUG_KERAS_TRACEBACK_FILTERING: disabled")
    except Exception as e:
        print(f"DEBUG_KERAS_TRACEBACK_FILTERING_FAILED: {type(e).__name__}: {e}")

    backend = _resolve_keras_backend(keras)

    USE_DATALOADER = _env_bool("USE_DATALOADER", False)
    FORCE_META_INIT = _env_bool("FORCE_META_INIT", False)
    # Optional knobs (default: don't hide errors)
    DYNAMO_SUPPRESS_ERRORS = _env_bool("DYNAMO_SUPPRESS_ERRORS", False)
    DYNAMO_DYNAMIC_SHAPES = _env_bool("DYNAMO_DYNAMIC_SHAPES", False)
    CAPTURE_SCALAR_OUTPUTS = _env_bool("DYNAMO_CAPTURE_SCALARS", True)

    print(f"PYTHON: {sys.version.split()[0]}")
    print(f"KERAS_VERSION: {getattr(keras, '__version__', 'unknown')}")
    print(f"TORCH_VERSION: {getattr(torch, '__version__', 'unknown')}")
    print(f"KERAS_BACKEND_RESOLVED: {backend}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"USE_DATALOADER: {USE_DATALOADER}")
    print(f"FORCE_META_INIT: {FORCE_META_INIT}")
    print(f"DYNAMO_SUPPRESS_ERRORS: {DYNAMO_SUPPRESS_ERRORS}")
    print(f"DYNAMO_DYNAMIC_SHAPES: {DYNAMO_DYNAMIC_SHAPES}")
    print(f"DYNAMO_CAPTURE_SCALARS: {CAPTURE_SCALAR_OUTPUTS}")

    if str(backend).lower() != "torch":
        _skip(f"keras backend is not torch (backend={backend!r})")

    if not torch.cuda.is_available():
        _skip("CUDA not available")

    try:
        print(f"CUDA_DEVICE_0: {torch.cuda.get_device_name(0)}")
    except Exception:
        print("CUDA_DEVICE_0: <unknown>")

    # Determinism
    SEED = 1337
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    MeanIoU = getattr(getattr(keras, "metrics", None), "MeanIoU", None)
    if MeanIoU is None:
        _skip("keras.metrics.MeanIoU not available")

    # IMPORTANT: keep this metric simple & deterministic.
    # We do argmax on y_pred then feed int labels to MeanIoU.
    class ArgmaxMeanIoU(keras.metrics.Metric):
        def __init__(self, num_classes: int, name: str = "mean_iou", **kwargs):
            super().__init__(name=name, **kwargs)
            self._inner = MeanIoU(num_classes=num_classes)

        def update_state(self, y_true, y_pred, sample_weight=None):
            ops = keras.ops
            y_true = ops.cast(y_true, "int64")
            y_pred_ids = ops.argmax(y_pred, axis=-1)
            y_pred_ids = ops.cast(y_pred_ids, "int64")
            return self._inner.update_state(y_true, y_pred_ids, sample_weight=sample_weight)

        def result(self):
            return self._inner.result()

        def reset_state(self):
            return self._inner.reset_state()

    # Avoid one_hot (it hit Dynamo FakeTensor issues for you).
    # Use sparse NLL gather-based loss.
    def sparse_nll_loss(y_true, y_pred):
        y_true = y_true.to(dtype=torch.int64)             # (B,H,W)
        y_pred = torch.clamp(y_pred, 1e-7, 1.0)           # (B,H,W,K)
        logp = torch.log(y_pred)
        idx = y_true.unsqueeze(-1)                        # (B,H,W,1)
        picked = torch.gather(logp, dim=-1, index=idx).squeeze(-1)
        return -picked.mean()

    # Fixed shapes
    B = 2
    num_classes = 3
    H, W, C = 8, 8, 1
    N = B * 4  # divisible by B

    # Fixed-size tensors on CUDA
    x = torch.randn(N, H, W, C, device="cuda", dtype=torch.float32)
    y_idx = torch.randint(0, num_classes, (N, H, W), device="cuda", dtype=torch.int64)

    # Optional: touch meta tensor to "warm" meta paths without forcing #18440 behavior.
    if FORCE_META_INIT:
        try:
            xm = torch.empty((B, H, W, C), device="meta", dtype=torch.float32)
            print(f"DEBUG_META_TENSOR_CREATED: {xm.device}")
        except Exception as e:
            print(f"DEBUG_META_TENSOR_CREATE_FAILED: {type(e).__name__}: {e}")

    def _configure_dynamo():
        # Only configure if torch._dynamo exists
        try:
            import torch._dynamo  # type: ignore
            torch._dynamo.config.suppress_errors = bool(DYNAMO_SUPPRESS_ERRORS)
            if hasattr(torch._dynamo.config, "dynamic_shapes"):
                torch._dynamo.config.dynamic_shapes = bool(DYNAMO_DYNAMIC_SHAPES)
            if hasattr(torch._dynamo.config, "capture_scalar_outputs"):
                torch._dynamo.config.capture_scalar_outputs = bool(CAPTURE_SCALAR_OUTPUTS)
            print("DEBUG_DYNAMO_CONFIG_APPLIED: True")
            print(f"DEBUG_DYNAMO_SUPPRESS_ERRORS_APPLIED: {torch._dynamo.config.suppress_errors}")
            if hasattr(torch._dynamo.config, "dynamic_shapes"):
                print(f"DEBUG_DYNAMO_DYNAMIC_SHAPES_APPLIED: {torch._dynamo.config.dynamic_shapes}")
            if hasattr(torch._dynamo.config, "capture_scalar_outputs"):
                print(f"DEBUG_DYNAMO_CAPTURE_SCALARS_APPLIED: {torch._dynamo.config.capture_scalar_outputs}")
        except Exception as e:
            print(f"DEBUG_DYNAMO_CONFIG_FAILED: {type(e).__name__}: {e}")

    def build_model(jit_compile: bool):
        inputs = keras.Input(batch_shape=(B, H, W, C), dtype="float32")
        z = keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inputs)
        outputs = keras.layers.Conv2D(num_classes, 1, padding="same", activation="softmax")(z)
        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=sparse_nll_loss,
            metrics=[ArgmaxMeanIoU(num_classes=num_classes)],
            jit_compile=bool(jit_compile),
        )
        return model

    def run_fit(model):
        if USE_DATALOADER:
            from torch.utils.data import TensorDataset, DataLoader
            ds = TensorDataset(x, y_idx)
            dl = DataLoader(ds, batch_size=B, shuffle=False, drop_last=True)
            model.fit(dl, epochs=1, verbose=0)
        else:
            model.fit(x, y_idx, batch_size=B, epochs=1, shuffle=False, verbose=0)

    # Apply dynamo config once (if relevant).
    _configure_dynamo()

    # --------------------------
    # Phase A: jit_compile=False (expected OK)
    # --------------------------
    try:
        model_a = build_model(jit_compile=False)
        run_fit(model_a)
        print("DEBUG_PHASE_A_OK: jit_compile=False")
    except Exception as e:
        print("DEBUG_PHASE_A_EXCEPTION_TYPE:", type(e).__name__)
        print("DEBUG_PHASE_A_EXCEPTION_MSG:", str(e))
        print("DEBUG_PHASE_A_TRACEBACK:")
        print(traceback.format_exc())
        _fail()

    # --------------------------
    # Phase B: jit_compile=True (expected to FAIL for bug discovery)
    # --------------------------
    try:
        model_b = build_model(jit_compile=True)
        run_fit(model_b)
        print("DEBUG_PHASE_B_OK_UNEXPECTED: jit_compile=True did NOT fail")
        _fail()
    except Exception as e:
        print("DEBUG_PHASE_B_EXCEPTION_TYPE:", type(e).__name__)
        print("DEBUG_PHASE_B_EXCEPTION_MSG:", str(e))
        print("DEBUG_PHASE_B_TRACEBACK:")
        print(traceback.format_exc())
        # This is the “good news” case: eager works, jit_compile fails.
        _pass()


if __name__ == "__main__":
    main()


# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Pytorch


# Commands
# *****************
# 0) create + activate env
# conda create -y -n keras_22168 python=3.11 pip
# conda activate keras_22168
# python -m pip install -U pip

# # 1) install torch (CUDA 11.8 wheels) + keras + numpy
# python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# python -m pip install keras numpy

# # 2) run repro (torch backend)
# export CUDA_VISIBLE_DEVICES=0
# export KERAS_BACKEND=torch
# python /tmp/keras_testcase.py

# /tmp/keras_testcase.py



# Output:
# *****************
# PYTHON: 3.11.14
# KERAS_VERSION: 3.13.2
# TORCH_VERSION: 2.7.1+cu118
# KERAS_BACKEND_RESOLVED: torch
# CUDA_DEVICE_0: NVIDIA GeForce RTX 3090

# DEBUG_PHASE_A_OK: jit_compile=False

# DEBUG_PHASE_B_EXCEPTION_TYPE: InductorError
# DEBUG_PHASE_B_EXCEPTION_MSG: CppCompileError: C++ compile error

# Output:
# /tmp/torchinductor_*/**/*.cpp: In function ‘void kernel(double*, double*)’:
# ... error: ‘zuf0’ was not declared in this scope
# ... error: ‘zuf1’ was not declared in this scope

# torch._inductor.exc.InductorError: CppCompileError: C++ compile error

# Test Passed ✅




# Reported ✅
# Link: 
# https://github.com/pytorch/pytorch/issues/175057


# *****************
# chatgpt explanation:
# *****************

# Not automatically. What you have **is a strong signal**, but calling it a “real PyTorch bug” is only correct **if you frame it precisely**:

# ### What’s definitely true from your output

# * `jit_compile=False` works.
# * `jit_compile=True` fails inside **torch.compile / Inductor** with a **C++ codegen compile error**:

#   * undefined identifiers `zuf0`, `zuf1`
#   * raised as `torch._inductor.exc.InductorError: CppCompileError`
#     So yes: **PyTorch/Inductor generated invalid C++** and then failed to compile it. That’s almost always a PyTorch bug.

# ### What’s *not* proven yet

# * That it’s **purely** a PyTorch bug and not triggered by something unusual Keras is doing (graph structure, op patterns, dtype/layout).
#   This still counts as a PyTorch bug **if the backend is expected to support this workload**, but maintainers may ask for a “PyTorch-only repro” or tag it as “integration edge case”.

# ### How to state it correctly in the PyTorch issue

# Say something like:

# * “Keras (torch backend) `jit_compile=True` uses `torch.compile` and triggers an Inductor C++ codegen error (undefined identifiers), while eager mode works.”
#   That’s accurate and defensible.

# ### Bottom line

# Yes — **it’s a legitimate PyTorch/Inductor issue worth reporting**, because PyTorch produced broken generated C++.
# But don’t oversell it as “confirmed real bug” until:

# * a maintainer reproduces it, or
# * you reduce it to a minimal repro (ideally PyTorch-only).