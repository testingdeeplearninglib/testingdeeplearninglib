# GCFL-TRACINGGRA-0075

import json
import os
import random
import sys
from contextlib import contextmanager

def _print_env() -> None:
    env = {
        "python": sys.version.split()[0],
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", "<unset>"),
    }
    print(f"ENV: {json.dumps(env, sort_keys=True)}", flush=True)

def _skip(reason: str) -> None:
    print(f"SKIP_ENV: {reason}", flush=True)
    sys.exit(0)

def _pass() -> None:
    print("Test Passed ✅", flush=True)
    sys.exit(0)

def _fail() -> None:
    print("Test Failed ❌", flush=True)
    sys.exit(0)

def _harness_error(e: BaseException) -> None:
    print(f"HARNESS_ERROR: {type(e).__name__}: {e}", flush=True)
    sys.exit(1)

@contextmanager
def _nullcontext():
    yield

_print_env()

try:
    import tensorflow as tf
except Exception as e:
    _skip(f"tensorflow import failed: {type(e).__name__}: {e}")

try:
    import numpy as np
except Exception:
    np = None

def _set_determinism() -> None:
    random.seed(0)
    if np is not None:
        np.random.seed(0)
    try:
        tf.random.set_seed(0)
    except Exception:
        try:
            tf.set_random_seed(0)
        except Exception:
            pass

def _get_v1():
    if hasattr(tf, "compat") and hasattr(tf.compat, "v1"):
        return tf.compat.v1
    return tf

def _configure_gpu_runtime() -> None:
    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception:
        gpus = []
    print(f"INFO: tf_version={getattr(tf, '__version__', '<unknown>')} gpus={len(gpus)}", flush=True)
    if not gpus:
        print("INFO: running without visible GPU", flush=True)
        return
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

def _disable_eager_if_possible(v1mod) -> None:
    try:
        if hasattr(v1mod, "disable_eager_execution"):
            v1mod.disable_eager_execution()
    except Exception:
        pass

def _concat(axis, tensors):
    try:
        return tf.concat(tensors, axis=axis)
    except TypeError:
        return tf.concat(axis, tensors)

def _global_init_op(v1mod):
    for name in ("global_variables_initializer", "initialize_all_variables"):
        fn = getattr(v1mod, name, None)
        if fn is not None:
            try:
                return fn()
            except Exception:
                continue
    return None

def _make_gd_optimizer(v1mod, lr=0.01):
    train = getattr(v1mod, "train", None)
    if train is not None:
        opt_cls = getattr(train, "GradientDescentOptimizer", None)
        if opt_cls is not None:
            return opt_cls(lr)
    train2 = getattr(tf, "train", None)
    if train2 is not None:
        opt_cls2 = getattr(train2, "GradientDescentOptimizer", None)
        if opt_cls2 is not None:
            return opt_cls2(lr)
    raise RuntimeError("GradientDescentOptimizer not available")

def _looks_like_shape_guard_preventing_forward(msg: str) -> bool:
    m = msg.lower()
    markers = [
        "shape invariant",
        "shape_invariants",
        "input tensor",
        "enters the loop with shape",
        "after one iteration, shape is",
    ]
    has_shape = "shape" in m
    return any(marker in m for marker in markers) or (has_shape and "invariant" in m)

def _build_graph(strategy_name: str, use_shape_invariants: bool, dynamic_h0: bool):
    del strategy_name
    v1 = _get_v1()

    Graph = getattr(v1, "Graph", None)
    Session = getattr(v1, "Session", None)
    if Graph is None or Session is None:
        _skip("TensorFlow graph/session APIs unavailable in this install")

    g = Graph()
    ctx = g.as_default() if hasattr(g, "as_default") else _nullcontext()

    with ctx:
        x = tf.constant([[1.0, 2.0]], dtype=tf.float32)
        X = v1.get_variable("X", initializer=x)
        i0 = tf.constant(0)

        if dynamic_h0:
            shape_vec = tf.concat(
                [tf.shape(X)[:1] * 0, tf.constant([2], dtype=tf.int32)],
                axis=0,
            )
            H0 = tf.zeros(shape_vec, dtype=tf.float32)
        else:
            H0 = tf.zeros([0, 2], dtype=tf.float32)

        def cond(i, H):
            del H
            return i < 2

        def body(i, H):
            return i + 1, _concat(0, [H, X])

        kwargs = {}
        if use_shape_invariants:
            shape_i = getattr(i0, "shape", None)
            kwargs["shape_invariants"] = [shape_i, tf.TensorShape([None, 2])]

        try:
            _, H = v1.while_loop(cond, body, [i0, H0], **kwargs)
        except TypeError:
            _, H = v1.while_loop(cond, body, [i0, H0])

        s = tf.reduce_sum(H)
        init_op = _global_init_op(v1)
        if init_op is None:
            raise RuntimeError("Could not create variables initializer op")

    return g, Session, X, H, s, init_op

def _run_strategy(strategy_name: str, use_shape_invariants: bool, dynamic_h0: bool) -> bool:
    print(
        f"INFO: strategy={strategy_name} use_shape_invariants={use_shape_invariants} dynamic_h0={dynamic_h0}",
        flush=True,
    )

    try:
        g, Session, X, H, s, init_op = _build_graph(
            strategy_name=strategy_name,
            use_shape_invariants=use_shape_invariants,
            dynamic_h0=dynamic_h0,
        )
    except SystemExit:
        raise
    except Exception as e:
        if _looks_like_shape_guard_preventing_forward(str(e)):
            print(f"INFO: strategy={strategy_name} blocked during graph build by shape guard", flush=True)
            return False
        raise

    try:
        with g.as_default():
            try:
                opt = _make_gd_optimizer(_get_v1(), lr=0.01)
            except Exception:
                _skip("GradientDescentOptimizer unavailable")

            try:
                train_op = opt.minimize(s, var_list=[X])
                print(f"INFO: strategy={strategy_name} gradient_graph_built_ok", flush=True)
            except Exception as e_grad_build:
                print(
                    f"INFO: strategy={strategy_name} gradient_build_exception={type(e_grad_build).__name__}: {e_grad_build}",
                    flush=True,
                )
                return True

        with Session(graph=g) as sess:
            sess.run(init_op)

            try:
                h_val, s_val = sess.run([H, s])
                print(
                    f"INFO: strategy={strategy_name} forward_ok shape={getattr(h_val, 'shape', None)} sum={s_val}",
                    flush=True,
                )
                del h_val, s_val
            except Exception as e_fwd:
                if _looks_like_shape_guard_preventing_forward(str(e_fwd)):
                    print(f"INFO: strategy={strategy_name} blocked during forward by shape guard", flush=True)
                    return False
                print(f"INFO: strategy={strategy_name} forward_exception={type(e_fwd).__name__}: {e_fwd}", flush=True)
                return False

            try:
                sess.run(train_op)
            except Exception as e_grad_run:
                print(
                    f"INFO: strategy={strategy_name} gradient_run_exception={type(e_grad_run).__name__}: {e_grad_run}",
                    flush=True,
                )
                return True

            print(f"INFO: strategy={strategy_name} backward_ok", flush=True)
            return False

    except SystemExit:
        raise
    except Exception as e:
        _harness_error(e)

def main() -> None:
    try:
        _configure_gpu_runtime()
        v1 = _get_v1()
        _disable_eager_if_possible(v1)
        _set_determinism()

        strategies = [
            ("static_h0_no_invariants", False, False),
            ("static_h0_with_invariants", True, False),
            ("dynamic_h0_no_invariants", False, True),
            ("dynamic_h0_with_invariants", True, True),
        ]

        for strategy_name, use_inv, dyn_h0 in strategies:
            try:
                if _run_strategy(strategy_name, use_inv, dyn_h0):
                    _pass()
            except SystemExit:
                raise
            except Exception as e:
                _harness_error(e)

        _fail()

    except SystemExit:
        raise
    except Exception as e:
        _harness_error(e)

if __name__ == "__main__":
    main()



# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Tensorflow


# Commands
# *****************
# GPU run
# conda activate tf_venv
# cd ~/dl_testing
# mkdir -p logs

# export CUDA_VISIBLE_DEVICES=0
# export PYTHONUNBUFFERED=1
# export TF_CPP_MIN_LOG_LEVEL=1

# python testcases/tensorflow_testcase.py 2>&1 | tee logs/tensorflow_testcase_graphfix_gpu_$(date +%Y%m%d_%H%M%S).log
# echo "exit_code=$?"

# CPU run
# conda activate tf_venv
# cd ~/dl_testing
# mkdir -p logs

# export CUDA_VISIBLE_DEVICES=""
# export PYTHONUNBUFFERED=1
# export TF_CPP_MIN_LOG_LEVEL=1

# python testcases/tensorflow_testcase.py 2>&1 | tee logs/tensorflow_testcase_graphfix_cpu_$(date +%Y%m%d_%H%M%S).log
# echo "exit_code=$?"



# Output:
# *****************
# GPU output
# ENV: {"cuda_visible_devices": "0", "ld_library_path": "/usr/lib/x86_64-linux-gnu:/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/nvidia/cublas/lib:/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/nvidia/cuda_cupti/lib:/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/nvidia/cudnn/lib:/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/nvidia/cufft/lib:/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/nvidia/curand/lib:/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/nvidia/cusolver/lib:/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/nvidia/cusparse/lib:/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/nvidia/nccl/lib:/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/nvidia/nvjitlink/lib", "pid": 2722846, "python": "3.11.15"}
# INFO: tf_version=2.21.0 gpus=1
# INFO: strategy=static_h0_no_invariants use_shape_invariants=False dynamic_h0=False
# INFO: strategy=static_h0_no_invariants blocked during graph build by shape guard
# INFO: strategy=static_h0_with_invariants use_shape_invariants=True dynamic_h0=False
# INFO: strategy=static_h0_with_invariants gradient_graph_built_ok
# INFO: strategy=static_h0_with_invariants forward_ok shape=(2, 2) sum=6.0
# INFO: strategy=static_h0_with_invariants backward_ok
# INFO: strategy=dynamic_h0_no_invariants use_shape_invariants=False dynamic_h0=True
# INFO: strategy=dynamic_h0_no_invariants blocked during graph build by shape guard
# INFO: strategy=dynamic_h0_with_invariants use_shape_invariants=True dynamic_h0=True
# INFO: strategy=dynamic_h0_with_invariants gradient_graph_built_ok
# INFO: strategy=dynamic_h0_with_invariants forward_ok shape=(2, 2) sum=6.0
# INFO: strategy=dynamic_h0_with_invariants backward_ok
# Test Failed ❌
# exit_code=0
# CPU output
# ENV: {"cuda_visible_devices": "", "ld_library_path": "/usr/lib/x86_64-linux-gnu:/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/nvidia/cublas/lib:/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/nvidia/cuda_cupti/lib:/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/nvidia/cudnn/lib:/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/nvidia/cufft/lib:/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/nvidia/curand/lib:/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/nvidia/cusolver/lib:/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/nvidia/cusparse/lib:/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/nvidia/nccl/lib:/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/nvidia/nvjitlink/lib", "pid": 2723107, "python": "3.11.15"}
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# E0000 00:00:1774344446.579101 2723107 cuda_platform.cc:52] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
# INFO: tf_version=2.21.0 gpus=0
# INFO: running without visible GPU
# INFO: strategy=static_h0_no_invariants use_shape_invariants=False dynamic_h0=False
# INFO: strategy=static_h0_no_invariants blocked during graph build by shape guard
# INFO: strategy=static_h0_with_invariants use_shape_invariants=True dynamic_h0=False
# INFO: strategy=static_h0_with_invariants gradient_graph_built_ok
# INFO: strategy=static_h0_with_invariants forward_ok shape=(2, 2) sum=6.0
# INFO: strategy=static_h0_with_invariants backward_ok
# INFO: strategy=dynamic_h0_no_invariants use_shape_invariants=False dynamic_h0=True
# INFO: strategy=dynamic_h0_no_invariants blocked during graph build by shape guard
# INFO: strategy=dynamic_h0_with_invariants use_shape_invariants=True dynamic_h0=True
# INFO: strategy=dynamic_h0_with_invariants gradient_graph_built_ok
# INFO: strategy=dynamic_h0_with_invariants forward_ok shape=(2, 2) sum=6.0
# INFO: strategy=dynamic_h0_with_invariants backward_ok
# Test Failed ❌
# exit_code=0