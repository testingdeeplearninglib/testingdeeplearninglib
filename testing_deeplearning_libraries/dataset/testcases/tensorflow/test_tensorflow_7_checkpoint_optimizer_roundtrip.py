# FILE: GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_checkpoint_optimizer_roundtrip.py
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
    batch = int(os.environ.get("BATCH", "4"))
    d_model = int(os.environ.get("D_MODEL", "16"))

    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass

    _env_line(tf, np, {"SEED": seed, "BATCH": batch, "D_MODEL": d_model})

    def build_model():
        x = tf.keras.Input(shape=(d_model,), name="x")
        y = tf.keras.layers.Dense(8, activation="relu")(x)
        y = tf.keras.layers.Dense(1)(y)
        return tf.keras.Model(x, y)

    def one_step(model, opt, x, y):
        with tf.GradientTape() as tape:
            pred = model(x, training=True)
            loss = tf.reduce_mean(tf.square(pred - y))
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return float(loss.numpy())

    x = np.random.RandomState(seed).randn(batch, d_model).astype("float32")
    y = np.random.RandomState(seed + 1).randn(batch, 1).astype("float32")

    with tempfile.TemporaryDirectory() as td:
        ckpt_dir = os.path.join(td, "ckpts")
        try:
            m = build_model()
            opt = tf.keras.optimizers.Adam(learning_rate=1e-2)
            # create optimizer slots
            l0 = one_step(m, opt, x, y)
            step0 = int(getattr(opt, "iterations").numpy())

            ckpt = tf.train.Checkpoint(model=m, opt=opt)
            mgr = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
            path = mgr.save()

            # restore into fresh objects
            m2 = build_model()
            opt2 = tf.keras.optimizers.Adam(learning_rate=1e-2)
            _ = one_step(m2, opt2, x, y)  # slot creation
            ckpt2 = tf.train.Checkpoint(model=m2, opt=opt2)
            ckpt2.restore(path).expect_partial()

            step1 = int(getattr(opt2, "iterations").numpy())
            if step1 < step0:
                _pass()  # optimizer iteration state not restored (suspicious)

            l1 = one_step(m2, opt2, x, y)
            if not (l1 == l1 and abs(l1) < 1e30):
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
# bug no: GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_checkpoint_optimizer_roundtrip
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
#   testcases/tf_serialization_inputs/GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_checkpoint_optimizer_roundtrip.py \
#   > logs/GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_checkpoint_optimizer_roundtrip_stdout.log \
#   2> logs/GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_checkpoint_optimizer_roundtrip_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_checkpoint_optimizer_roundtrip_stdout.log

# Observed output:
# exit_code=0
# Test Failed ❌

# Note:
# No suspicious checkpoint/optimizer-state restoration issue was triggered in this run.