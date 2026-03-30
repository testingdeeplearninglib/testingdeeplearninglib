# GCFL-TRACINGGRA-0013

import os
import sys
import json
import random
import tempfile
from typing import Callable, Tuple, Optional


def _skip(reason: str) -> None:
    print(f"SKIP_ENV: {reason}")
    sys.exit(0)


def _pass() -> None:
    print("Test Passed ✅")
    sys.exit(0)


def _fail() -> None:
    print("Test Failed ❌")
    sys.exit(0)


def _harness_error(e: BaseException) -> None:
    print(f"HARNESS_ERROR: {type(e).__name__}: {e}")
    sys.exit(1)


def _set_determinism() -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    random.seed(0)


def _read_text_file(path: str) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _clip(text: str, limit: int = 800) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def _describe_exception(exc: Optional[BaseException]) -> str:
    if exc is None:
        return "None"
    return f"{type(exc).__name__}: {exc}"


def _print_env(tf) -> None:
    try:
        env = {
            "python": sys.version.split()[0],
            "tensorflow": getattr(tf, "__version__", "unknown"),
            "visible_gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
            "pid": os.getpid(),
        }
        print("ENV:", json.dumps(env, sort_keys=True), flush=True)
    except Exception as e:
        print(f'ENV: {{"error": "{type(e).__name__}: {e}"}}', flush=True)


def _run_with_fd2_capture(fn: Callable[[], None]) -> Tuple[bool, Optional[BaseException], str]:
    """
    Run fn() while capturing OS-level stderr (fd=2) into a temp file.
    Returns: (ok, exception, captured_text)
    """
    tmp_path = None
    old_fd2 = None
    cap_fd = None
    ok = True
    exc: Optional[BaseException] = None

    try:
        try:
            sys.stderr.flush()
        except Exception:
            pass

        old_fd2 = os.dup(2)

        with tempfile.NamedTemporaryFile(prefix="tf_stderr_", suffix=".log", delete=False) as tmp:
            tmp_path = tmp.name

        cap_fd = os.open(tmp_path, os.O_WRONLY | os.O_TRUNC)
        os.dup2(cap_fd, 2)

        try:
            fn()
        except BaseException as e:
            ok = False
            exc = e

        try:
            sys.stderr.flush()
        except Exception:
            pass

        try:
            os.fsync(2)
        except Exception:
            pass

    finally:
        try:
            if old_fd2 is not None:
                os.dup2(old_fd2, 2)
        except Exception:
            pass

        try:
            if cap_fd is not None:
                os.close(cap_fd)
        except Exception:
            pass

        try:
            if old_fd2 is not None:
                os.close(old_fd2)
        except Exception:
            pass

    captured = _read_text_file(tmp_path) if tmp_path else ""

    try:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
    except Exception:
        pass

    return ok, exc, captured


def _matches_bug_signals(text: str) -> bool:
    """
    Tightened oracle: only targeted historical failure signals.
    Do NOT use generic tokens like 'grappler', 'oom', or 'implementation_selector'
    because they create false positives.
    """
    t = (text or "").lower()
    signals = [
        "constant folding failed",
        "unsupported type: 21",
        "invalid argument: unsupported type",
        "skipping optimization due to error while loading function libraries",
        "signatures do not match",
    ]
    return any(s in t for s in signals)


def main() -> None:
    try:
        _set_determinism()

        try:
            import numpy as np  # type: ignore
        except Exception as e:
            _skip(f"missing numpy ({type(e).__name__}: {e})")

        try:
            import tensorflow as tf  # type: ignore
        except Exception as e:
            _skip(f"missing tensorflow ({type(e).__name__}: {e})")

        _print_env(tf)

        try:
            tf.keras.utils.set_random_seed(0)
        except Exception:
            pass

        try:
            tf.random.set_seed(0)
        except Exception:
            pass

        try:
            tf.config.run_functions_eagerly(False)
        except Exception:
            pass

        x_np = np.random.RandomState(0).rand(10, 5).astype(np.float32)
        x_tf = tf.constant(x_np)

        class Model(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.dense = tf.keras.layers.Dense(10)

            def call(self, inputs):
                return self.dense(inputs)

        model = Model()

        _ = model(x_tf[:1])

        def _materialize_grads(grads):
            total = tf.constant(0.0, dtype=tf.float32)
            for g in grads:
                if g is None:
                    return tf.constant(float("nan"), dtype=tf.float32)
                total = total + tf.reduce_sum(tf.cast(g, tf.float32))
            return total

        def forward(x_tensor):
            batch_size = tf.shape(x_tensor)[0]
            ys = tf.TensorArray(
                dtype=tf.float32,
                size=batch_size,
                clear_after_read=False,
            )
            for i in tf.range(batch_size):
                y = model(x_tensor[i:i + 1])
                ys = ys.write(i, y)
            return ys.stack()

        def train(x_tensor, forward_func):
            with tf.GradientTape() as tape:
                ys = forward_func(x_tensor)
                loss = tf.reduce_mean(ys)
            grads = tape.gradient(loss, model.trainable_variables)
            return _materialize_grads(grads)

        def big_train(x_tensor):
            with tf.GradientTape() as tape:
                batch_size = tf.shape(x_tensor)[0]
                ys = tf.TensorArray(
                    dtype=tf.float32,
                    size=batch_size,
                    clear_after_read=False,
                )
                for i in tf.range(batch_size):
                    y = model(x_tensor[i:i + 1])
                    ys = ys.write(i, y)
                ys = ys.stack()
                loss = tf.reduce_mean(ys)
            grads = tape.gradient(loss, model.trainable_variables)
            return _materialize_grads(grads)

        train_fn = tf.function(train, autograph=True, jit_compile=False)
        big_train_fn = tf.function(big_train, autograph=True, jit_compile=False)
        forward_fn = tf.function(forward, autograph=True, jit_compile=False)

        def candidate_1():
            _ = train_fn(x_tf, forward)

        def candidate_2():
            _ = big_train_fn(x_tf)

        def sanity_variant():
            _ = train(x_tf, forward_fn)

        ok_s, exc_s, logs_s = _run_with_fd2_capture(sanity_variant)
        combined_s = _describe_exception(exc_s) + "\n" + logs_s

        print(f"DIAG: sanity ok={ok_s} exc={_describe_exception(exc_s)}", flush=True)
        if logs_s.strip():
            print("DIAG_LOG_SANITY:", _clip(logs_s), flush=True)

        if _matches_bug_signals(combined_s):
            print("DIAG: targeted signal matched in sanity path", flush=True)
            _pass()

        if not ok_s:
            raise RuntimeError(f"sanity variant failed unexpectedly: {_describe_exception(exc_s)}")

        for name, cand in (
            ("candidate_1_python_function_arg", candidate_1),
            ("candidate_2_big_train_tf_function", candidate_2),
        ):
            ok, exc, logs = _run_with_fd2_capture(cand)
            combined = _describe_exception(exc) + "\n" + logs

            print(f"DIAG: {name} ok={ok} exc={_describe_exception(exc)}", flush=True)
            if logs.strip():
                print(f"DIAG_LOG_{name}:", _clip(logs), flush=True)

            if _matches_bug_signals(combined):
                print(f"DIAG: targeted signal matched in {name}", flush=True)
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

# Tensorflow


# Commands
# *****************
# conda activate tf_venv
# cd ~/dl_testing
# python -m py_compile testcases/tensorflow_testcase.py

# conda activate tf_venv
# cd ~/dl_testing

# python - <<'PY'
# import sys
# import tensorflow as tf
# print("python =", sys.version.split()[0])
# print("tf =", tf.__version__)
# print("visible_gpus =", tf.config.list_physical_devices("GPU"))
# PY

# conda activate tf_venv
# cd ~/dl_testing

# mkdir -p logs

# unset TF_XLA_FLAGS
# export CUDA_VISIBLE_DEVICES=0
# export PYTHONUNBUFFERED=1
# export TF_CPP_MIN_LOG_LEVEL=0

# set -o pipefail
# python testcases/tensorflow_testcase.py 2>&1 | tee logs/gcfl_tracinggra_0013.log
# echo "exit_code=$?"

# conda activate tf_venv
# cd ~/dl_testing

# python - <<'PY' > logs/gcfl_tracinggra_0013_env.txt
# import sys
# import tensorflow as tf
# print("python =", sys.version)
# print("tf =", tf.__version__)
# print("build_info =", tf.sysconfig.get_build_info())
# print("visible_gpus =", tf.config.list_physical_devices("GPU"))
# PY


# Output:
# *****************
# python = 3.11.15
# tf = 2.21.0
# visible_gpus = [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1774334700.883388 2532922 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# ENV: {"pid": 2532922, "python": "3.11.15", "tensorflow": "2.21.0", "visible_gpus": ["/physical_device:GPU:0"]}
# I0000 00:00:1774334703.204445 2532922 gpu_device.cc:2043] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 22446 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:02:00.0, compute capability: 8.6
# DIAG: sanity ok=True exc=None
# DIAG: candidate_1_python_function_arg ok=True exc=None
# DIAG: candidate_2_big_train_tf_function ok=True exc=None
# Test Failed ❌
# exit_code=0

# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1774334708.532798 2533190 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.