# FILE: GCFL-AUTOGRAD_BACKWARD-0003_tc01_tf_complex64_matmul_grad_fd.py
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

def _finite(x):
    import numpy as np
    return bool(np.isfinite(x).all())

def _numeric_grad_matmul_real_sum(A, B, eps=1e-3):
    import numpy as np
    # loss = sum(real(A@B))
    def loss_np(Ac, Bc):
        C = Ac @ Bc
        return float(np.real(C).sum())

    Ar = np.real(A).astype(np.float64)
    Ai = np.imag(A).astype(np.float64)
    Br = np.real(B).astype(np.float64)
    Bi = np.imag(B).astype(np.float64)

    grad_Ar = np.zeros_like(Ar)
    grad_Ai = np.zeros_like(Ai)
    grad_Br = np.zeros_like(Br)
    grad_Bi = np.zeros_like(Bi)

    def pack(Ar_, Ai_, Br_, Bi_):
        return (Ar_ + 1j * Ai_).astype(np.complex64), (Br_ + 1j * Bi_).astype(np.complex64)

    baseA, baseB = pack(Ar, Ai, Br, Bi)
    _ = loss_np(baseA, baseB)

    # A real
    for i in range(Ar.shape[0]):
        for j in range(Ar.shape[1]):
            Arp = Ar.copy(); Arm = Ar.copy()
            Arp[i, j] += eps; Arm[i, j] -= eps
            Ap, Bp = pack(Arp, Ai, Br, Bi)
            Am, Bm = pack(Arm, Ai, Br, Bi)
            grad_Ar[i, j] = (loss_np(Ap, Bp) - loss_np(Am, Bm)) / (2.0 * eps)

    # A imag
    for i in range(Ai.shape[0]):
        for j in range(Ai.shape[1]):
            Aip = Ai.copy(); Aim = Ai.copy()
            Aip[i, j] += eps; Aim[i, j] -= eps
            Ap, Bp = pack(Ar, Aip, Br, Bi)
            Am, Bm = pack(Ar, Aim, Br, Bi)
            grad_Ai[i, j] = (loss_np(Ap, Bp) - loss_np(Am, Bm)) / (2.0 * eps)

    # B real
    for i in range(Br.shape[0]):
        for j in range(Br.shape[1]):
            Brp = Br.copy(); Brm = Br.copy()
            Brp[i, j] += eps; Brm[i, j] -= eps
            Ap, Bp = pack(Ar, Ai, Brp, Bi)
            Am, Bm = pack(Ar, Ai, Brm, Bi)
            grad_Br[i, j] = (loss_np(Ap, Bp) - loss_np(Am, Bm)) / (2.0 * eps)

    # B imag
    for i in range(Bi.shape[0]):
        for j in range(Bi.shape[1]):
            Bip = Bi.copy(); Bim = Bi.copy()
            Bip[i, j] += eps; Bim[i, j] -= eps
            Ap, Bp = pack(Ar, Ai, Br, Bip)
            Am, Bm = pack(Ar, Ai, Br, Bim)
            grad_Bi[i, j] = (loss_np(Ap, Bp) - loss_np(Am, Bm)) / (2.0 * eps)

    return grad_Ar, grad_Ai, grad_Br, grad_Bi

def main():
    try:
        import numpy as np
        import tensorflow as tf
    except Exception as e:
        _skip(f"import failed: {type(e).__name__}: {e}")

    py_ok = (sys.version_info.major == 3 and sys.version_info.minor in (10, 11))
    if not py_ok:
        _skip(f"Python not in {{3.10,3.11}}: {sys.version_info.major}.{sys.version_info.minor}")

    if getattr(tf, "__version__", "") != "2.20.0":
        _skip(f"tensorflow!=2.20.0: {getattr(tf,'__version__','unknown')}")

    seed = _env_int("SEED", 2026)
    iters = _env_int("ITERS", 3)
    m = _env_int("M", 2)
    k = _env_int("K", 2)
    n = _env_int("N", 2)
    eps = float(os.environ.get("EPS", "1e-3"))
    atol = float(os.environ.get("ATOL", "5e-2"))

    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass

    env_payload = {
        "test_id": "GCFL-AUTOGRAD_BACKWARD-0003_tc01",
        "gcfl_id": "GCFL-AUTOGRAD_BACKWARD-0003",
        "python": sys.version,
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "eager": bool(tf.executing_eagerly()),
        "devices": {"gpu": len(tf.config.list_physical_devices("GPU")), "cpu": len(tf.config.list_physical_devices("CPU"))},
        "knobs": {"SEED": seed, "ITERS": iters, "M": m, "K": k, "N": n, "EPS": eps, "ATOL": atol},
        "oracle": "grad_mismatch_vs_finite_diff",
    }
    print("ENV: " + json.dumps(env_payload, sort_keys=True))

    # Small shapes, few iterations
    for i in range(iters):
        rs = np.random.RandomState(seed + i)
        A = (rs.randn(m, k) + 1j * rs.randn(m, k)).astype(np.complex64)
        B = (rs.randn(k, n) + 1j * rs.randn(k, n)).astype(np.complex64)

        # TF grad
        try:
            Avar = tf.Variable(A)
            Bvar = tf.Variable(B)
            with tf.GradientTape(persistent=False) as tape:
                C = tf.matmul(Avar, Bvar)
                loss = tf.reduce_sum(tf.math.real(C))  # real scalar
            gA, gB = tape.gradient(loss, [Avar, Bvar])
        except Exception as e:
            _pass()

        if gA is None or gB is None:
            _pass()

        gA_np = gA.numpy()
        gB_np = gB.numpy()

        if not _finite(np.real(gA_np)) or not _finite(np.imag(gA_np)) or not _finite(np.real(gB_np)) or not _finite(np.imag(gB_np)):
            _pass()

        # Numeric FD grad
        gAr_fd, gAi_fd, gBr_fd, gBi_fd = _numeric_grad_matmul_real_sum(A, B, eps=eps)

        # Compare: TF complex grad real/imag vs FD for real/imag parts
        errA_r = np.max(np.abs(np.real(gA_np).astype(np.float64) - gAr_fd))
        errA_i = np.max(np.abs(np.imag(gA_np).astype(np.float64) - gAi_fd))
        errB_r = np.max(np.abs(np.real(gB_np).astype(np.float64) - gBr_fd))
        errB_i = np.max(np.abs(np.imag(gB_np).astype(np.float64) - gBi_fd))

        max_err = float(max(errA_r, errA_i, errB_r, errB_i))
        if max_err > atol:
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
# bug no: GCFL-AUTOGRAD_BACKWARD-0003_tc01
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
#   testcases/tf_batch_inputs/GCFL-AUTOGRAD_BACKWARD-0003_tc01_tf_complex64_matmul_grad_fd.py \
#   > logs/GCFL-AUTOGRAD_BACKWARD-0003_tc01_stdout.log \
#   2> logs/GCFL-AUTOGRAD_BACKWARD-0003_tc01_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-AUTOGRAD_BACKWARD-0003_tc01_stdout.log
#
# Observed output:
# exit_code=0
# Test Failed ❌
#
# Note:
# The suspicious gradient mismatch was not triggered in this run.
