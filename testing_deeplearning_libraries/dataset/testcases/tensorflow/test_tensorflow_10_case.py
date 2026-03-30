# GCFL-OTHER-0077

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile


def _set_determinism():
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _seed_everything():
    try:
        import random
        random.seed(2021)
    except Exception:
        pass
    try:
        import numpy as np
        np.random.seed(2021)
    except Exception:
        pass


def _print_env(n=None, num_weights=None):
    payload = {
        "python": sys.version.split()[0],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", None),
        "gcfl_target_mb": os.environ.get("GCFL_TARGET_MB", None),
        "gcfl_num_weights": os.environ.get("GCFL_NUM_WEIGHTS", None),
        "n": n,
        "num_weights": num_weights,
        "transform_graph_in_path": shutil.which("transform_graph") is not None,
    }
    try:
        import tensorflow as tf
        payload["tensorflow"] = getattr(tf, "__version__", "unknown")
        try:
            from tensorflow.tools.graph_transforms import TransformGraph as _  # type: ignore
            payload["python_transformgraph"] = True
        except Exception:
            payload["python_transformgraph"] = False
        try:
            payload["gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
        except Exception:
            payload["gpus"] = []
    except Exception as e:
        payload["tensorflow"] = f"missing ({e})"
        payload["python_transformgraph"] = False
        payload["gpus"] = []
    print("ENV: " + json.dumps(payload, sort_keys=True))


# ---------------- Memory sizing helpers ----------------
def _get_total_ram_bytes() -> int:
    try:
        if hasattr(os, "sysconf"):
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            if page_size > 0 and pages > 0:
                return page_size * pages
    except Exception:
        pass
    return 4 * 1024 * 1024 * 1024


def _choose_n_and_k():
    try:
        target_mb = int(str(os.environ.get("GCFL_TARGET_MB", "1024")).strip())
    except Exception:
        target_mb = 1024
    target_mb = max(128, min(target_mb, 2048))

    try:
        num_weights = int(str(os.environ.get("GCFL_NUM_WEIGHTS", "3")).strip())
    except Exception:
        num_weights = 3
    num_weights = max(1, min(num_weights, 6))

    ram = _get_total_ram_bytes()
    safe_cap_mb = max(256, int((ram * 0.25) / (1024 * 1024)))
    target_mb = min(target_mb, safe_cap_mb)

    target_bytes = target_mb * 1024 * 1024
    n = int(math.sqrt(max(1, target_bytes // 4)))
    n = max(1024, (n // 256) * 256)

    return n, num_weights


# ---------------- Crash detection ----------------
_BAD_ALLOC_MARKERS = (
    "std::bad_alloc",
    "terminate called after throwing",
    "what(): std::bad_alloc",
    "aborted (core dumped)",
    "core dumped",
    "sigabrt",
    "killed",
)


def _looks_like_bad_alloc(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(m in low for m in _BAD_ALLOC_MARKERS)


# ---------------- TensorFlow graph generation ----------------
def _require_tensorflow():
    try:
        import tensorflow as tf  # noqa: F401
    except Exception as e:
        _skip(f"missing tensorflow ({e})")


def _build_and_write_graph_pb(out_path: str, n: int, num_weights: int):
    import tensorflow as tf

    try:
        tf.compat.v1.disable_eager_execution()
    except Exception:
        pass

    v1 = tf.compat.v1
    try:
        v1.reset_default_graph()
    except Exception:
        pass

    with v1.Graph().as_default():
        _seed_everything()
        try:
            v1.set_random_seed(2021)
        except Exception:
            pass

        x = v1.placeholder(tf.float32, shape=[1, n], name="input")
        scalar = tf.constant(1.0, dtype=tf.float32, name="scalar")

        ys = []
        for i in range(num_weights):
            w = tf.fill([n, n], scalar, name=f"fill_w_{i}")
            y = tf.matmul(x, w, name=f"matmul_{i}")
            ys.append(y)

        acc = ys[0]
        for i in range(1, len(ys)):
            acc = tf.add(acc, ys[i], name=f"add_{i}")

        _ = tf.identity(acc, name="output")

        graph_def = v1.get_default_graph().as_graph_def(add_shapes=True)
        with tf.io.gfile.GFile(out_path, "wb") as f:
            f.write(graph_def.SerializeToString())


def _child_run_python_transform(in_graph: str, out_graph: str, n: int):
    import tensorflow as tf

    try:
        tf.compat.v1.disable_eager_execution()
    except Exception:
        pass

    TransformGraph = None
    try:
        from tensorflow.tools.graph_transforms import TransformGraph as _TG  # type: ignore
        TransformGraph = _TG
    except Exception:
        TransformGraph = None

    if TransformGraph is None:
        _skip("tensorflow graph_transforms (TransformGraph) not available in this installation")

    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(in_graph, "rb") as f:
        graph_def.ParseFromString(f.read())

    transforms = [
        f'strip_unused_nodes(type=float, shape="1,{n}")',
        "remove_nodes(op=Identity, op=CheckNumerics)",
        "fold_constants(ignore_errors=false)",
        "quantize_weights",
        "round_weights(num_bits=8)",
    ]

    new_graph_def = TransformGraph(graph_def, ["input"], ["output"], transforms)

    with tf.io.gfile.GFile(out_graph, "wb") as f:
        f.write(new_graph_def.SerializeToString())

    sys.exit(0)


def _parent_run():
    _set_determinism()
    _require_tensorflow()

    n, num_weights = _choose_n_and_k()
    _print_env(n=n, num_weights=num_weights)

    with tempfile.TemporaryDirectory() as td:
        in_graph = os.path.join(td, "in_graph.pb")
        out_graph = os.path.join(td, "out_graph.pb")

        _build_and_write_graph_pb(in_graph, n=n, num_weights=num_weights)

        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--child_py_transform",
            in_graph,
            out_graph,
            str(n),
        ]

        try:
            child = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=180,
                env=dict(os.environ),
            )
        except subprocess.TimeoutExpired:
            _fail()

        child_stdout = (child.stdout or "").strip()
        child_stderr = (child.stderr or "").strip()

        if child.returncode == 0 and child_stdout.startswith("SKIP_ENV:"):
            print(child_stdout)
            sys.exit(0)

        if child.returncode < 0:
            _pass()

        if child.returncode != 0 and (
            _looks_like_bad_alloc(child_stderr) or _looks_like_bad_alloc(child_stdout)
        ):
            _pass()

        if child_stdout:
            print("INFO: child stdout tail=" + child_stdout[-2000:])
        if child_stderr:
            print("INFO: child stderr tail=" + child_stderr[-2000:])
        _fail()


def main():
    try:
        _set_determinism()
        if len(sys.argv) >= 2 and sys.argv[1] == "--child_py_transform":
            if len(sys.argv) != 5:
                _harness_error(ValueError("invalid child arguments"))
            in_graph = sys.argv[2]
            out_graph = sys.argv[3]
            try:
                n = int(sys.argv[4])
            except Exception:
                _harness_error(ValueError("invalid n"))
            _seed_everything()
            _require_tensorflow()
            _child_run_python_transform(in_graph, out_graph, n)
            _harness_error(RuntimeError("child did not exit as expected"))
        else:
            _parent_run()
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



# Output:
# *****************

# ENV: {"cuda_visible_devices": "", "gcfl_num_weights": "3", "gcfl_target_mb": "512", "gpus": [], "n": 11520, "num_weights": 3, "python": "3.7.16", "python_transformgraph": false, "tensorflow": "2.11.0", "transform_graph_in_path": false}
# SKIP_ENV: tensorflow graph_transforms (TransformGraph) not available in this installation


# Output:
# *****************
# ALLOCATOR_STATUS: {"allocated_by": "cupy", "leave_free_mib": 2048, "ok": true, "physical_gpu_index": 0, "target_alloc_mib": 20606}
# WORKER_RESULTS: [{"corrupt": false, "cpu_sum": 8386560.0, "expected_sum": 8386560.0, "gpu_sum": 8386560.0, "physical_gpu_index": 0, "reasons": [], "rmax": 0.9999771118164062, "rmean": -0.014401411637663841, "rmin": -0.9996249675750732, "rvar": 0.3416896462440491}, {"corrupt": false, "cpu_sum": 8386560.0, "expected_sum": 8386560.0, "gpu_sum": 8386560.0, "physical_gpu_index": 0, "reasons": [], "rmax": 0.9999771118164062, "rmean": -0.014401411637663841, "rmin": -0.9996249675750732, "rvar": 0.3416896462440491}]
# Test Failed ❌
