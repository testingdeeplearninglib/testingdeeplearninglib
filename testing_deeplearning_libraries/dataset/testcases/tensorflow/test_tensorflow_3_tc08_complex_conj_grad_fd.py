# FILE: GCFL-AUTOGRAD_BACKWARD-0003_tc08_tf_complex_conj_grad_fd.py
import os
import sys
import json
import random

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
    n = _env_int("N", 8)
    eps = float(os.environ.get("EPS", "1e-3"))
    atol = float(os.environ.get("ATOL", "5e-2"))

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    env_payload = {
        "test_id": "GCFL-AUTOGRAD_BACKWARD-0003_tc08",
        "gcfl_id": "GCFL-AUTOGRAD_BACKWARD-0003",
        "python": sys.version,
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "devices": {"gpu": len(tf.config.list_physical_devices("GPU")), "cpu": len(tf.config.list_physical_devices("CPU"))},
        "knobs": {"SEED": seed, "N": n, "EPS": eps, "ATOL": atol},
        "oracle": "complex conj/abs2 gradient mismatch vs finite differences",
    }
    print("ENV: " + json.dumps(env_payload, sort_keys=True))

    rs = np.random.RandomState(seed)
    z = (rs.randn(n) + 1j * rs.randn(n)).astype(np.complex64)

    # TF grad
    try:
        zvar = tf.Variable(z)
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(tf.math.real(tf.math.conj(zvar) * zvar))  # sum(|z|^2)
        gz = tape.gradient(loss, zvar)
    except Exception:
        _pass()

    if gz is None:
        _pass()

    gz_np = gz.numpy().astype(np.complex64)
    if not np.isfinite(np.real(gz_np)).all() or not np.isfinite(np.imag(gz_np)).all():
        _pass()

    # FD grad for real and imag parts
    zr = np.real(z).astype(np.float64)
    zi = np.imag(z).astype(np.float64)

    def loss_np(zr_, zi_):
        zz = (zr_ + 1j * zi_).astype(np.complex64)
        return float(np.real(np.conj(zz) * zz).sum())

    gr_fd = np.zeros_like(zr)
    gi_fd = np.zeros_like(zi)
    for i in range(n):
        zrp = zr.copy(); zrm = zr.copy()
        zrp[i] += eps; zrm[i] -= eps
        gr_fd[i] = (loss_np(zrp, zi) - loss_np(zrm, zi)) / (2.0 * eps)

        zip_ = zi.copy(); zim_ = zi.copy()
        zip_[i] += eps; zim_[i] -= eps
        gi_fd[i] = (loss_np(zr, zip_) - loss_np(zr, zim_)) / (2.0 * eps)

    err_r = float(np.max(np.abs(np.real(gz_np).astype(np.float64) - gr_fd)))
    err_i = float(np.max(np.abs(np.imag(gz_np).astype(np.float64) - gi_fd)))
    if max(err_r, err_i) > atol:
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
# bug no: GCFL-AUTOGRAD_BACKWARD-0003_tc08
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
#   testcases/tf_batch_inputs/GCFL-AUTOGRAD_BACKWARD-0003_tc08_tf_complex_conj_grad_fd.py \
#   > logs/GCFL-AUTOGRAD_BACKWARD-0003_tc08_stdout.log \
#   2> logs/GCFL-AUTOGRAD_BACKWARD-0003_tc08_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-AUTOGRAD_BACKWARD-0003_tc08_stdout.log
#
# Observed output:
# exit_code=0
# Test Failed ❌
#
# Note:
# The suspicious complex conjugate gradient mismatch was not triggered in this run.