# FILE: GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_saved_model_module_dictvars.py
import os, sys, json, tempfile, random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2"))
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

def _skip(reason: str):
    print(f"SKIP_ENV: {reason}", flush=True)
    sys.exit(0)

def _pass():
    print("Test Passed ✅", flush=True)
    sys.exit(0)

def _fail():
    print("Test Failed ❌", flush=True)
    sys.exit(0)

def _norm_tfver(v: str) -> str:
    v = (v or "").strip()
    v = v.split("+", 1)[0]
    v = v.split("-", 1)[0]
    return v

def _env_line(tf, np, knobs: dict):
    payload = {
        "python": sys.version.split()[0],
        "tensorflow": getattr(tf, "__version__", "unknown"),
        "numpy": getattr(np, "__version__", "unknown"),
        "gpu_count": len(tf.config.list_physical_devices("GPU")),
        "cpu_count": len(tf.config.list_physical_devices("CPU")),
        "knobs": knobs,
    }
    print("ENV: " + json.dumps(payload, sort_keys=True), flush=True)

def main():
    if not (sys.version_info.major == 3 and sys.version_info.minor in (10, 11)):
        _skip(f"Python not in {{3.10,3.11}}: {sys.version.split()[0]}")

    try:
        import numpy as np
    except Exception as e:
        _skip(f"numpy import failed: {type(e).__name__}: {e}")

    try:
        import tensorflow as tf
    except Exception as e:
        _skip(f"tensorflow import failed: {type(e).__name__}: {e}")

    if _norm_tfver(getattr(tf, "__version__", "")) != "2.20.0":
        _skip(f"tensorflow version != 2.20.0: {getattr(tf,'__version__','unknown')}")

    seed = int(os.environ.get("SEED", "2026"))
    d_model = int(os.environ.get("D_MODEL", "8"))
    batch = int(os.environ.get("BATCH", "2"))

    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass

    _env_line(tf, np, {"SEED": seed, "BATCH": batch, "D_MODEL": d_model})

    class Mod(tf.Module):
        def __init__(self):
            super().__init__()
            # dict of variables (trackable structure edge)
            self.vars = {
                "w": tf.Variable(tf.random.normal([d_model, d_model], seed=seed), name="w"),
                "b": tf.Variable(tf.zeros([d_model]), name="b"),
            }

        @tf.function(input_signature=[tf.TensorSpec([None, d_model], tf.float32)])
        def __call__(self, x):
            return tf.matmul(x, self.vars["w"]) + self.vars["b"]

    x = np.random.RandomState(seed).randn(batch, d_model).astype("float32")

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "sm")
        try:
            m = Mod()
            y0 = m(tf.constant(x)).numpy()
            tf.saved_model.save(m, p)
            m2 = tf.saved_model.load(p)
            y1 = m2(tf.constant(x)).numpy()
            if not (y0.shape == y1.shape and abs(float((y0 - y1).max())) <= 0.0):
                _pass()
        except Exception:
            _pass()

    _fail()

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        _skip(f"harness_error: {type(e).__name__}: {e}")



# Output:
# *****************
# bug no: GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_saved_model_module_dictvars
# Result: Test Failed ❌

# Triggering command:
# conda activate tf_venv_220_py311
# cd ~/dl_testing
# export CUDA_VISIBLE_DEVICES=0
# export TF_CPP_MIN_LOG_LEVEL=2
# export TF_DETERMINISTIC_OPS=1
# export KERAS_BACKEND=tensorflow
# export SEED=2026
# /home/talha/miniconda3/envs/tf_venv_220_py311/bin/python3.11 \
#   testcases/tf_serialization_inputs/GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_saved_model_module_dictvars.py \
#   > logs/GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_saved_model_module_dictvars_stdout.log \
#   2> logs/GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_saved_model_module_dictvars_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_saved_model_module_dictvars_stdout.log

# Observed output:
# exit_code=0
# Test Failed ❌

# Note:
# The dict-of-variables SavedModel probe did not show a mismatch or exception in this run.