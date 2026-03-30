# GCFL-OTHER-0037

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

# Keras


# Commands
# *****************
# conda activate keras_testcase
# export CUDA_VISIBLE_DEVICES=0
# export KERAS_BACKEND=torch
# export JIT_COMPILE=0
# export USE_DATALOADER=0
# export FORCE_META_INIT=0
# python /tmp/keras_testcase.py

# conda activate keras_testcase
# export CUDA_VISIBLE_DEVICES=0
# export KERAS_BACKEND=torch
# export JIT_COMPILE=1
# export USE_DATALOADER=0
# export FORCE_META_INIT=0
# python /tmp/keras_testcase.py


# conda activate keras_testcase

# export CUDA_VISIBLE_DEVICES=0
# export KERAS_BACKEND=torch

# python -c "import sys, torch, keras; \
# print('PYTHON:', sys.version.split()[0]); \
# print('KERAS:', keras.__version__); \
# print('TORCH:', torch.__version__); \
# print('CUDA_AVAILABLE:', torch.cuda.is_available())"

# PYTHON: 3.10.19
# KERAS: 3.3.0
# TORCH: 2.1.2+cu118
# CUDA_AVAILABLE: True


# Output:
# *****************
# (keras_testcase) talha@bitse-SYS-7048GR-TR:~/dl_testing$ conda activate keras_testcase
# export CUDA_VISIBLE_DEVICES=0
# export KERAS_BACKEND=torch
# export JIT_COMPILE=0
# export USE_DATALOADER=0
# export FORCE_META_INIT=0
# python /tmp/keras_testcase.py
# SCRIPT_PATH: /tmp/keras_testcase.py
# SCRIPT_SHA256: ca632e851a1720c100c1f4325b7e49cdc66c36c94436ed4e40ddf6dfaf714ac1
# DEBUG_KERAS_TRACEBACK_FILTERING: disabled
# PYTHON: 3.10.19
# KERAS_VERSION: 3.3.0
# TORCH_VERSION: 2.1.2+cu118
# KERAS_BACKEND_RESOLVED: torch
# CUDA_VISIBLE_DEVICES: 0
# USE_DATALOADER: False
# FORCE_META_INIT: False
# DYNAMO_SUPPRESS_ERRORS: False
# DYNAMO_DYNAMIC_SHAPES: False
# DYNAMO_CAPTURE_SCALARS: True
# CUDA_DEVICE_0: NVIDIA GeForce RTX 3090
# DEBUG_DYNAMO_CONFIG_APPLIED: True
# DEBUG_DYNAMO_SUPPRESS_ERRORS_APPLIED: False
# DEBUG_DYNAMO_DYNAMIC_SHAPES_APPLIED: False
# DEBUG_DYNAMO_CAPTURE_SCALARS_APPLIED: True
# DEBUG_PHASE_A_OK: jit_compile=False
# DEBUG_PHASE_B_EXCEPTION_TYPE: TypeError
# DEBUG_PHASE_B_EXCEPTION_MSG: an integer is required
# DEBUG_PHASE_B_TRACEBACK:
# Traceback (most recent call last):
#   File "/tmp/keras_testcase.py", line 251, in main
#     run_fit(model_b)
#   File "/tmp/keras_testcase.py", line 227, in run_fit
#     model.fit(x, y_idx, batch_size=B, epochs=1, shuffle=False, verbose=0)
#   File "/home/talha/miniconda3/envs/keras_testcase/lib/python3.10/site-packages/keras/src/utils/traceback_utils.py", line 113, in error_handler
#     return fn(*args, **kwargs)
#   File "/home/talha/miniconda3/envs/keras_testcase/lib/python3.10/site-packages/keras/src/backend/torch/trainer.py", line 254, in fit
#     logs = self.train_function(data)
#   File "/home/talha/miniconda3/envs/keras_testcase/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 328, in _fn
#     return fn(*args, **kwargs)
#   File "/home/talha/miniconda3/envs/keras_testcase/lib/python3.10/site-packages/keras/src/backend/torch/trainer.py", line 117, in one_step_on_data
#     return self.train_step(data)
#   File "/home/talha/miniconda3/envs/keras_testcase/lib/python3.10/site-packages/keras/src/backend/torch/trainer.py", line 44, in train_step
#     y_pred = self(x, training=True)
#   File "/home/talha/miniconda3/envs/keras_testcase/lib/python3.10/site-packages/keras/src/backend/torch/trainer.py", line 50, in <resume in train_step>
#     self.zero_grad()
#   File "/home/talha/miniconda3/envs/keras_testcase/lib/python3.10/site-packages/keras/src/backend/torch/trainer.py", line 52, in <resume in train_step>
#     loss = self.compute_loss(
#   File "/home/talha/miniconda3/envs/keras_testcase/lib/python3.10/site-packages/keras/src/trainers/trainer.py", line 316, in compute_loss
#     loss = self._compile_loss(y, y_pred, sample_weight)
#   File "/home/talha/miniconda3/envs/keras_testcase/lib/python3.10/site-packages/keras/src/trainers/trainer.py", line 319, in <resume in compute_loss>
#     for loss in self.losses:
#   File "/home/talha/miniconda3/envs/keras_testcase/lib/python3.10/site-packages/keras/src/layers/layer.py", line 1115, in losses
#     for layer in self._flatten_layers(include_self=False):
#   File "/home/talha/miniconda3/envs/keras_testcase/lib/python3.10/site-packages/keras/src/layers/layer.py", line 1386, in _flatten_layers
#     def _flatten_layers(self, include_self=True, recursive=True):
# TypeError: an integer is required

# Test Passed ✅
# (keras_testcase) talha@bitse-SYS-7048GR-TR:~/dl_testing$ conda activate keras_testcase
# export CUDA_VISIBLE_DEVICES=0
# export KERAS_BACKEND=torch
# export JIT_COMPILE=1
# export USE_DATALOADER=0
# export FORCE_META_INIT=0
# python /tmp/keras_testcase.py
# SCRIPT_PATH: /tmp/keras_testcase.py
# SCRIPT_SHA256: ca632e851a1720c100c1f4325b7e49cdc66c36c94436ed4e40ddf6dfaf714ac1
# DEBUG_KERAS_TRACEBACK_FILTERING: disabled
# PYTHON: 3.10.19
# KERAS_VERSION: 3.3.0
# TORCH_VERSION: 2.1.2+cu118
# KERAS_BACKEND_RESOLVED: torch
# CUDA_VISIBLE_DEVICES: 0
# USE_DATALOADER: False
# FORCE_META_INIT: False
# DYNAMO_SUPPRESS_ERRORS: False
# DYNAMO_DYNAMIC_SHAPES: False
# DYNAMO_CAPTURE_SCALARS: True
# CUDA_DEVICE_0: NVIDIA GeForce RTX 3090
# DEBUG_DYNAMO_CONFIG_APPLIED: True
# DEBUG_DYNAMO_SUPPRESS_ERRORS_APPLIED: False
# DEBUG_DYNAMO_DYNAMIC_SHAPES_APPLIED: False
# DEBUG_DYNAMO_CAPTURE_SCALARS_APPLIED: True
# DEBUG_PHASE_A_OK: jit_compile=False
# DEBUG_PHASE_B_EXCEPTION_TYPE: TypeError
# DEBUG_PHASE_B_EXCEPTION_MSG: an integer is required
# DEBUG_PHASE_B_TRACEBACK:
# Traceback (most recent call last):
#   File "/tmp/keras_testcase.py", line 251, in main
#     run_fit(model_b)
#   File "/tmp/keras_testcase.py", line 227, in run_fit
#     model.fit(x, y_idx, batch_size=B, epochs=1, shuffle=False, verbose=0)
#   File "/home/talha/miniconda3/envs/keras_testcase/lib/python3.10/site-packages/keras/src/utils/traceback_utils.py", line 113, in error_handler
#     return fn(*args, **kwargs)
#   File "/home/talha/miniconda3/envs/keras_testcase/lib/python3.10/site-packages/keras/src/backend/torch/trainer.py", line 254, in fit
#     logs = self.train_function(data)
#   File "/home/talha/miniconda3/envs/keras_testcase/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 328, in _fn
#     return fn(*args, **kwargs)
#   File "/home/talha/miniconda3/envs/keras_testcase/lib/python3.10/site-packages/keras/src/backend/torch/trainer.py", line 117, in one_step_on_data
#     return self.train_step(data)
#   File "/home/talha/miniconda3/envs/keras_testcase/lib/python3.10/site-packages/keras/src/backend/torch/trainer.py", line 44, in train_step
#     y_pred = self(x, training=True)
#   File "/home/talha/miniconda3/envs/keras_testcase/lib/python3.10/site-packages/keras/src/backend/torch/trainer.py", line 50, in <resume in train_step>
#     self.zero_grad()
#   File "/home/talha/miniconda3/envs/keras_testcase/lib/python3.10/site-packages/keras/src/backend/torch/trainer.py", line 52, in <resume in train_step>
#     loss = self.compute_loss(
#   File "/home/talha/miniconda3/envs/keras_testcase/lib/python3.10/site-packages/keras/src/trainers/trainer.py", line 316, in compute_loss
#     loss = self._compile_loss(y, y_pred, sample_weight)
#   File "/home/talha/miniconda3/envs/keras_testcase/lib/python3.10/site-packages/keras/src/trainers/trainer.py", line 319, in <resume in compute_loss>
#     for loss in self.losses:
#   File "/home/talha/miniconda3/envs/keras_testcase/lib/python3.10/site-packages/keras/src/layers/layer.py", line 1115, in losses
#     for layer in self._flatten_layers(include_self=False):
#   File "/home/talha/miniconda3/envs/keras_testcase/lib/python3.10/site-packages/keras/src/layers/layer.py", line 1386, in _flatten_layers
#     def _flatten_layers(self, include_self=True, recursive=True):
# TypeError: an integer is required

# Test Passed ✅


# Reported ✅
# Link: 
# https://github.com/keras-team/keras/issues/22168