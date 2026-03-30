# FILE: GCFL-OTHER-0001_tf_case07_gru_recurrent_dropout_effectiveness.py
import os
import sys
import json
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2"))
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")


def _skip(r): print(f"SKIP_ENV: {r}"); sys.exit(0)
def _pass_(): print("Test Passed ✅"); sys.exit(0)
def _fail_(): print("Test Failed ❌"); sys.exit(0)
def _herr(e): print(f"HARNESS_ERROR: {type(e).__name__}: {e}"); sys.exit(1)


def _env_int(k, d):
    v = os.environ.get(k, "").strip()
    if not v:
        return d
    try:
        return int(v)
    except Exception:
        return d


def main():
    try:
        import numpy as np
    except Exception as e:
        _skip(f"numpy_import_failed:{type(e).__name__}:{e}")
    try:
        import tensorflow as tf
    except Exception as e:
        _skip(f"tf_import_failed:{type(e).__name__}:{e}")

    if not (sys.version_info.major == 3 and sys.version_info.minor in (10, 11)):
        _skip(f"python_not_supported:{sys.version_info.major}.{sys.version_info.minor}")
    if tf.__version__ != "2.20.0":
        _skip(f"tf_version_mismatch:{tf.__version__}")

    seed = _env_int("SEED", 2026)
    iters = _env_int("ITERS", 5)
    batch = _env_int("BATCH", 2)
    seq = _env_int("SEQ", 6)
    d_model = _env_int("D_MODEL", 16)
    hidden = _env_int("HIDDEN", 8)

    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass

    env_payload = {
        "python": sys.version,
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "knobs": {"SEED": seed, "ITERS": iters, "BATCH": batch, "SEQ": seq, "D_MODEL": d_model, "HIDDEN": hidden},
        "testcase": "gru_recurrent_dropout_effect_check",
    }
    print("ENV: " + json.dumps(env_payload, sort_keys=True))

    keras = tf.keras

    # build two GRU layers with identical weights but different recurrent_dropout
    layer0 = keras.layers.GRU(hidden, return_sequences=True, recurrent_dropout=0.0, dropout=0.0)
    layer1 = keras.layers.GRU(hidden, return_sequences=True, recurrent_dropout=0.95, dropout=0.0)

    # initialize by calling once
    x0 = tf.zeros([batch, seq, d_model], dtype=tf.float32)
    _ = layer0(x0, training=True)
    _ = layer1(x0, training=True)

    # copy weights from layer0 to layer1
    w0 = layer0.get_weights()
    layer1.set_weights(w0)

    # oracle: if recurrent_dropout is ignored, outputs from layer0 and layer1 will be too similar (allclose)
    hits = 0
    for i in range(iters):
        rs = np.random.RandomState(seed + i)
        x = tf.constant(rs.randn(batch, seq, d_model).astype("float32"))

        # ensure same seed for deterministic comparison
        try:
            tf.random.set_seed(seed + 123)
        except Exception:
            pass
        y0 = layer0(x, training=True)

        try:
            tf.random.set_seed(seed + 123)
        except Exception:
            pass
        y1 = layer1(x, training=True)

        y0n = y0.numpy()
        y1n = y1.numpy()
        max_abs = float(np.max(np.abs(y0n - y1n)))
        if max_abs < 1e-6:
            hits += 1

    # if many runs show near-identical outputs despite huge recurrent_dropout, suspicious
    if hits >= max(2, iters // 2):
        _pass_()
    _fail_()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        _herr(e)


# Output:
# *****************
# bug no: GCFL-OTHER-0001-07
# Result: Test Failed ❌
#
# Triggering command:
# conda activate tf_venv_220_py311
# cd ~/dl_testing
# export CUDA_VISIBLE_DEVICES=0
# export TF_CPP_MIN_LOG_LEVEL=2
# export TF_DETERMINISTIC_OPS=1
# export KERAS_BACKEND=tensorflow
# export SEED=2026
# /home/talha/miniconda3/envs/tf_venv_220_py311/bin/python3.11 \
#   testcases/tf_other_0001_inputs/GCFL-OTHER-0001-07_gru_recurrent_dropout_effectiveness.py \
#   > logs/GCFL-OTHER-0001-07_stdout.log \
#   2> logs/GCFL-OTHER-0001-07_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-OTHER-0001-07_stdout.log
#
# Observed output:
# exit_code=0
# Test Failed ❌
#
# Note:
# The test did not detect suspicious ineffectiveness of recurrent_dropout under this configuration.