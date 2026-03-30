# FILE: GCFL-AUTOGRAD_BACKWARD-0003_tc07_tf_embedding_grad_nan_or_nondet.py
import os
import sys
import json
import random
import traceback

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2"))
os.environ.setdefault("TF_DETERMINISTIC_OPS", os.environ.get("TF_DETERMINISTIC_OPS", "1"))

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

def _env_int(k: str, d: int) -> int:
    v = os.environ.get(k, "").strip()
    try:
        return int(v) if v else d
    except Exception:
        return d

def main():
    try:
        import numpy as np
        import tensorflow as tf
    except Exception as e:
        _skip(f"import failed: {type(e).__name__}: {e}")

    if not (sys.version_info.major == 3 and sys.version_info.minor in (10, 11)):
        _skip(f"Python not in {{3.10,3.11}}: {sys.version_info.major}.{sys.version_info.minor}")
    if tf.__version__ != "2.20.0":
        _skip(f"tensorflow!=2.20.0: {tf.__version__}")

    seed = _env_int("SEED", 2026)
    reps = _env_int("REPS", 3)
    vocab = _env_int("VOCAB", 2048)
    d_model = _env_int("D_MODEL", 64)
    batch = _env_int("BATCH", 8)
    seq = _env_int("SEQ", 32)
    atol = float(os.environ.get("ATOL", "0.0"))  # exact match by default

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    gpus = tf.config.list_physical_devices("GPU")
    # optional GPU path (still runs on CPU if no GPU)
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    env_payload = {
        "test_id": "GCFL-AUTOGRAD_BACKWARD-0003_tc07",
        "gcfl_id": "GCFL-AUTOGRAD_BACKWARD-0003",
        "python": sys.version,
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "devices": {"gpu": len(gpus), "cpu": len(tf.config.list_physical_devices("CPU"))},
        "knobs": {"SEED": seed, "REPS": reps, "VOCAB": vocab, "D_MODEL": d_model, "BATCH": batch, "SEQ": seq, "ATOL": atol},
        "oracle": "embedding grad contains NaN/Inf OR nondeterministic across identical runs",
    }
    print("ENV: " + json.dumps(env_payload, sort_keys=True))

    # fixed inputs
    rs = np.random.RandomState(seed)
    ids_np = rs.randint(0, vocab, size=(batch, seq), dtype=np.int32)
    w_np = rs.randn(batch, seq, d_model).astype("float32")

    grads = []
    for r in range(reps):
        try:
            tf.random.set_seed(seed)  # reset seed each rep
        except Exception:
            pass
        emb = tf.Variable(tf.random.normal([vocab, d_model], dtype=tf.float32, seed=seed), trainable=True)

        try:
            with tf.GradientTape() as tape:
                x = tf.nn.embedding_lookup(emb, ids_np)  # [B,S,D]
                loss = tf.reduce_sum(x * w_np)
            g = tape.gradient(loss, emb)
        except Exception:
            _pass()

        if g is None:
            _pass()

        gv = g.numpy()
        if not np.isfinite(gv).all():
            _pass()

        grads.append(gv)

    # nondeterminism check: identical runs should match
    for i in range(1, len(grads)):
        diff = float(np.max(np.abs(grads[i] - grads[0])))
        if diff > atol:
            _pass()

    _fail()

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        _harness_error(e)



# Output:
# *****************
# bug no: GCFL-AUTOGRAD_BACKWARD-0003_tc07
# Result: HARNESS_ERROR
## Result: Test Failed ❌

# Triggering command:
# conda activate tf_venv_220_py311
# cd ~/dl_testing
# export CUDA_VISIBLE_DEVICES=0
# export TF_CPP_MIN_LOG_LEVEL=2
# export TF_DETERMINISTIC_OPS=1
# export KERAS_BACKEND=tensorflow
# export SEED=2026
# /home/talha/miniconda3/envs/tf_venv_220_py311/bin/python3.11 \
#   testcases/tf_batch_inputs/GCFL-AUTOGRAD_BACKWARD-0003_tc07_tf_embedding_grad_nan_or_nondet.py \
#   > logs/GCFL-AUTOGRAD_BACKWARD-0003_tc07_stdout.log \
#   2> logs/GCFL-AUTOGRAD_BACKWARD-0003_tc07_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-AUTOGRAD_BACKWARD-0003_tc07_stdout.log
#
# Observed output:
# exit_code=1
# HARNESS_ERROR: AttributeError: 'IndexedSlices' object has no attribute 'numpy'
#
# Note:
# This is not clean evidence of a TensorFlow bug yet. The testcase itself is flawed because
# embedding gradients may be returned as IndexedSlices, and the script incorrectly calls .numpy()
# directly on that object.
