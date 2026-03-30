# GCFL-OTHER-0079

import os
import sys
import time
import json
import math
import random
import shutil
import traceback
import subprocess
import multiprocessing as mp
import importlib.util


def _print_and_exit(msg: str, code: int):
    print(msg)
    sys.exit(code)


def _skip(reason: str):
    _print_and_exit(f"SKIP_ENV: {reason}", 0)


def _pass():
    _print_and_exit("Test Passed ✅", 0)


def _fail():
    _print_and_exit("Test Failed ❌", 0)


def _harness_error(exc: BaseException):
    _print_and_exit(f"HARNESS_ERROR: {type(exc).__name__}: {exc}", 1)


def _run_cmd(cmd, timeout=10):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as e:
        return 999, "", str(e)


def _resolve_physical_gpu_index() -> int:
    override = os.environ.get("GCFL_GPU_PHYSICAL_INDEX", "").strip()
    if override and override.lstrip("-").isdigit():
        return int(override)

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        first = cvd.split(",")[0].strip()
        if first and first.lstrip("-").isdigit():
            return int(first)

    return 0


def _env_snapshot(physical_gpu_index: int):
    return {
        "python": sys.version.split()[0],
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "resolved_physical_gpu_index": physical_gpu_index,
        "tf_cpp_min_log_level": os.environ.get("TF_CPP_MIN_LOG_LEVEL", ""),
    }


def _nvidia_smi_mem(gpu_index: int):
    if shutil.which("nvidia-smi") is None:
        return None, None
    cmd = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    rc, out, err = _run_cmd(cmd, timeout=10)
    if rc != 0 or not out:
        return None, None
    try:
        parts = [p.strip() for p in out.split(",")]
        if len(parts) < 2:
            return None, None
        return int(parts[0]), int(parts[1])
    except Exception:
        return None, None


def _chunk_plan(total_bytes: int, chunk_mib: int = 256):
    chunk_bytes = int(chunk_mib) * 1024 * 1024
    if total_bytes <= 0:
        return []
    chunks = []
    remaining = int(total_bytes)
    while remaining > 0:
        n = min(chunk_bytes, remaining)
        chunks.append(n)
        remaining -= n
    return chunks


def _allocator_process(physical_gpu_index: int, leave_free_mib: int, ready_ev, stop_ev, status_q):
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_index)

        total, free = _nvidia_smi_mem(physical_gpu_index)
        if total is None or free is None:
            status_q.put({"ok": False, "skip_reason": "nvidia-smi not available to size GPU allocation"})
            ready_ev.set()
            return

        raw_target_alloc_mib = max(int(free) - int(leave_free_mib), 0)
        capped_target_alloc_mib = min(raw_target_alloc_mib, int(int(free) * 0.85))
        target_alloc_mib = max(capped_target_alloc_mib, 0)
        target_bytes = target_alloc_mib * 1024 * 1024
        if target_bytes <= 0:
            status_q.put({"ok": False, "skip_reason": f"not enough free GPU memory to preallocate (free_mib={free})"})
            ready_ev.set()
            return

        allocated_by = None
        hold_obj = None

        try:
            import cupy as cp

            cp.cuda.Device(0).use()
            bufs = []
            for n in _chunk_plan(target_bytes, chunk_mib=256):
                bufs.append(cp.empty((n,), dtype=cp.uint8))
            if bufs:
                bufs[0][0] = 1
                cp.cuda.Stream.null.synchronize()
            hold_obj = bufs
            allocated_by = "cupy"
        except Exception:
            hold_obj = None
            allocated_by = None

        if hold_obj is None:
            try:
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError("torch.cuda not available")
                torch.cuda.set_device(0)
                bufs = []
                for n in _chunk_plan(target_bytes, chunk_mib=256):
                    bufs.append(torch.empty((n,), dtype=torch.uint8, device="cuda"))
                if bufs:
                    bufs[0][0] = 1
                    torch.cuda.synchronize()
                hold_obj = bufs
                allocated_by = "torch"
            except Exception:
                hold_obj = None
                allocated_by = None

        if hold_obj is None:
            try:
                import tensorflow as tf

                gpus = tf.config.list_physical_devices("GPU")
                if not gpus:
                    raise RuntimeError("no TF GPU devices")
                tf.config.set_visible_devices(gpus[0], "GPU")
                try:
                    tf.config.experimental.set_memory_growth(gpus[0], True)
                except Exception:
                    pass

                bufs = []
                with tf.device("/GPU:0"):
                    for n in _chunk_plan(target_bytes, chunk_mib=128):
                        n_floats = max(1, n // 4)
                        bufs.append(tf.Variable(tf.ones([n_floats], dtype=tf.float32)))
                    _ = float(bufs[0].read_value()[0].numpy())
                hold_obj = bufs
                allocated_by = "tensorflow"
            except Exception:
                hold_obj = None
                allocated_by = None

        if hold_obj is None:
            status_q.put(
                {
                    "ok": False,
                    "skip_reason": "could not preallocate GPU memory (cupy/torch unavailable or failed; TF fallback failed)",
                }
            )
            ready_ev.set()
            return

        status_q.put(
            {
                "ok": True,
                "allocated_by": allocated_by,
                "target_alloc_mib": int(target_alloc_mib),
                "leave_free_mib": int(leave_free_mib),
                "physical_gpu_index": int(physical_gpu_index),
            }
        )
        ready_ev.set()

        t0 = time.time()
        while not stop_ev.is_set():
            time.sleep(0.1)
            if time.time() - t0 > 120:
                break

    except Exception as e:
        status_q.put({"ok": False, "harness_error": f"{type(e).__name__}: {e}"})
        try:
            ready_ev.set()
        except Exception:
            pass


def _worker_process(physical_gpu_index: int, result_q):
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_index)

        seed = 2021
        random.seed(seed)

        try:
            import numpy as np
            np.random.seed(seed)
        except Exception:
            pass

        try:
            import tensorflow as tf
        except Exception as e:
            result_q.put({"skip": True, "reason": f"missing tensorflow ({e})"})
            return

        try:
            gpus = tf.config.list_physical_devices("GPU")
        except Exception as e:
            result_q.put({"skip": True, "reason": f"cannot list TF GPUs ({e})"})
            return

        if not gpus:
            result_q.put({"skip": True, "reason": "no GPU visible to TensorFlow"})
            return

        try:
            tf.config.set_visible_devices(gpus[0], "GPU")
        except Exception:
            pass

        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except Exception:
            pass

        try:
            tf.compat.v1.disable_eager_execution()
        except Exception as e:
            result_q.put({"skip": True, "reason": f"cannot disable eager execution ({e})"})
            return

        try:
            tf.compat.v1.set_random_seed(seed)
        except Exception:
            pass

        import numpy as np

        N = 4096
        host = np.arange(N, dtype=np.float32)
        expected_sum = float(host.sum())
        expected_head = host[:16].copy()

        with tf.device("/cpu:0"):
            x_cpu = tf.constant(host, dtype=tf.float32)
            cpu_sum = tf.reduce_sum(x_cpu)
            cpu_head = x_cpu[:16]

        with tf.device("/gpu:0"):
            x_gpu = tf.identity(x_cpu)
            gpu_sum = tf.reduce_sum(x_gpu)
            gpu_head = x_gpu[:16]
            r = tf.compat.v1.random_uniform([2048], minval=-1.0, maxval=1.0, dtype=tf.float32, seed=seed)
            r_min = tf.reduce_min(r)
            r_max = tf.reduce_max(r)
            r_mean = tf.reduce_mean(r)
            r_var = tf.math.reduce_variance(r)

        config = tf.compat.v1.ConfigProto()
        config.allow_soft_placement = False
        try:
            config.gpu_options.allow_growth = True
            config.gpu_options.visible_device_list = "0"
        except Exception:
            pass

        try:
            sess = tf.compat.v1.Session(config=config)
        except Exception as e:
            result_q.put(
                {
                    "corrupt": False,
                    "phase": "session_create",
                    "exception": f"{type(e).__name__}: {e}",
                }
            )
            return

        try:
            cpu_sum_v, gpu_sum_v, cpu_head_v, gpu_head_v, rmin_v, rmax_v, rmean_v, rvar_v = sess.run(
                [cpu_sum, gpu_sum, cpu_head, gpu_head, r_min, r_max, r_mean, r_var]
            )
        except Exception as e:
            result_q.put(
                {
                    "corrupt": False,
                    "phase": "session_run",
                    "exception": f"{type(e).__name__}: {e}",
                }
            )
            try:
                sess.close()
            except Exception:
                pass
            return
        finally:
            try:
                sess.close()
            except Exception:
                pass

        def _is_finite(x):
            try:
                return bool(np.isfinite(x).all())
            except Exception:
                try:
                    return math.isfinite(float(x))
                except Exception:
                    return False

        cpu_sum_v = float(cpu_sum_v)
        gpu_sum_v = float(gpu_sum_v)
        cpu_head_v = np.asarray(cpu_head_v, dtype=np.float32).reshape(-1)
        gpu_head_v = np.asarray(gpu_head_v, dtype=np.float32).reshape(-1)

        corrupt = False
        reasons = []

        if not _is_finite(gpu_sum_v):
            corrupt = True
            reasons.append("gpu_sum_non_finite")
        if not _is_finite(gpu_head_v):
            corrupt = True
            reasons.append("gpu_head_non_finite")

        if not _is_finite(cpu_sum_v) or abs(cpu_sum_v - expected_sum) > 1e-2:
            result_q.put(
                {
                    "corrupt": False,
                    "phase": "cpu_baseline",
                    "exception": f"CPU baseline unexpected (cpu_sum={cpu_sum_v}, expected={expected_sum})",
                }
            )
            return

        if abs(gpu_sum_v - expected_sum) > 1e-2 or abs(gpu_sum_v - cpu_sum_v) > 1e-2:
            corrupt = True
            reasons.append("gpu_sum_mismatch")

        if cpu_head_v.shape != expected_head.shape or gpu_head_v.shape != expected_head.shape:
            corrupt = True
            reasons.append("head_shape_mismatch")
        else:
            if not np.allclose(gpu_head_v, expected_head, atol=1e-3, rtol=0.0):
                corrupt = True
                reasons.append("gpu_head_mismatch")

        rmin_v = float(rmin_v)
        rmax_v = float(rmax_v)
        rmean_v = float(rmean_v)
        rvar_v = float(rvar_v)

        if not all(map(math.isfinite, [rmin_v, rmax_v, rmean_v, rvar_v])):
            corrupt = True
            reasons.append("gpu_random_non_finite")
        else:
            if rmin_v < -1.5 or rmax_v > 1.5:
                corrupt = True
                reasons.append("gpu_random_out_of_bounds")
            if rvar_v < 1e-6:
                corrupt = True
                reasons.append("gpu_random_degenerate")

        result_q.put(
            {
                "corrupt": bool(corrupt),
                "reasons": reasons,
                "cpu_sum": cpu_sum_v,
                "gpu_sum": gpu_sum_v,
                "expected_sum": expected_sum,
                "rmin": rmin_v,
                "rmax": rmax_v,
                "rmean": rmean_v,
                "rvar": rvar_v,
                "physical_gpu_index": int(physical_gpu_index),
            }
        )

    except Exception as e:
        result_q.put({"harness_error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})


def main():
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

        if importlib.util.find_spec("tensorflow") is None:
            _skip("missing tensorflow")

        if shutil.which("nvidia-smi") is None:
            _skip("nvidia-smi not available (cannot size GPU memory pressure test)")

        rc, out, err = _run_cmd(["nvidia-smi", "-L"], timeout=10)
        if rc != 0 or "GPU" not in out:
            _skip("no NVIDIA GPU detected via nvidia-smi")

        physical_gpu_index = _resolve_physical_gpu_index()
        total_mib, free_mib = _nvidia_smi_mem(physical_gpu_index)

        print(
            "ENV: "
            + json.dumps(
                {
                    **_env_snapshot(physical_gpu_index),
                    "gpu_total_mib": total_mib,
                    "gpu_free_mib_before_alloc": free_mib,
                },
                sort_keys=True,
            )
        )

        if total_mib is None or free_mib is None:
            _skip(f"cannot query memory for physical GPU index {physical_gpu_index}")

        leave_free_mib = 2048

        ctx = mp.get_context("spawn")
        ready_ev = ctx.Event()
        stop_ev = ctx.Event()
        alloc_status_q = ctx.Queue()
        worker_q = ctx.Queue()

        alloc_p = ctx.Process(
            target=_allocator_process,
            args=(physical_gpu_index, leave_free_mib, ready_ev, stop_ev, alloc_status_q),
        )
        alloc_p.start()

        if not ready_ev.wait(timeout=40):
            try:
                stop_ev.set()
            except Exception:
                pass
            try:
                if alloc_p.is_alive():
                    alloc_p.terminate()
            except Exception:
                pass
            _skip("GPU memory preallocator did not become ready in time")

        try:
            alloc_status = alloc_status_q.get(timeout=5)
        except Exception:
            alloc_status = None

        print("ALLOCATOR_STATUS: " + json.dumps(alloc_status, sort_keys=True))

        if alloc_status and alloc_status.get("harness_error"):
            try:
                stop_ev.set()
            except Exception:
                pass
            try:
                if alloc_p.is_alive():
                    alloc_p.terminate()
            except Exception:
                pass
            _harness_error(RuntimeError(alloc_status["harness_error"]))

        if not alloc_status or not alloc_status.get("ok", False):
            try:
                stop_ev.set()
            except Exception:
                pass
            try:
                if alloc_p.is_alive():
                    alloc_p.terminate()
            except Exception:
                pass
            reason = "unknown"
            if alloc_status and alloc_status.get("skip_reason"):
                reason = alloc_status["skip_reason"]
            _skip(reason)

        workers = []
        for _ in range(2):
            p = ctx.Process(target=_worker_process, args=(physical_gpu_index, worker_q))
            workers.append(p)
            p.start()
            time.sleep(0.3)

        results = []
        deadline = time.time() + 90
        while len(results) < len(workers) and time.time() < deadline:
            try:
                r = worker_q.get(timeout=2)
                results.append(r)
            except Exception:
                pass

        try:
            stop_ev.set()
        except Exception:
            pass

        for p in workers:
            try:
                p.join(timeout=5)
            except Exception:
                pass
            try:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=2)
            except Exception:
                pass

        try:
            alloc_p.join(timeout=5)
        except Exception:
            pass
        try:
            if alloc_p.is_alive():
                alloc_p.terminate()
                alloc_p.join(timeout=2)
        except Exception:
            pass

        print("WORKER_RESULTS: " + json.dumps(results, sort_keys=True))

        if len(results) < len(workers):
            _harness_error(RuntimeError(f"worker timeout or missing results ({len(results)}/{len(workers)})"))

        for r in results:
            if isinstance(r, dict) and r.get("harness_error"):
                _harness_error(RuntimeError(r.get("harness_error", "worker harness error")))

        for r in results:
            if isinstance(r, dict) and r.get("skip"):
                _skip(r.get("reason", "worker requested skip"))

        any_corrupt = any(isinstance(r, dict) and r.get("corrupt") is True for r in results)

        if any_corrupt:
            _pass()
        else:
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
# conda activate tf_venv
# cd ~/dl_testing
# export CUDA_VISIBLE_DEVICES=0
# export TF_CPP_MIN_LOG_LEVEL=3

# python -u testcases/tensorflow_testcase.py 2>&1 | tee gcfl_other_0079_tf211.log
# echo "exit_code=$?"


# conda activate tf_venv_modern
# cd ~/dl_testing
# export CUDA_VISIBLE_DEVICES=0
# export TF_CPP_MIN_LOG_LEVEL=3

# python -u testcases/tensorflow_testcase.py 2>&1 | tee gcfl_other_0079.log
# echo "exit_code=$?"


# Output:
# *****************
# ENV: {"cuda_visible_devices": "0", "gpu_free_mib_before_alloc": 24243, "gpu_total_mib": 24576, "pid": 923275, "python": "3.7.12", "resolved_physical_gpu_index": 0, "tf_cpp_min_log_level": "3"}
# ALLOCATOR_STATUS: {"allocated_by": "cupy", "leave_free_mib": 2048, "ok": true, "physical_gpu_index": 0, "target_alloc_mib": 20606}
# WORKER_RESULTS: [{"corrupt": false, "cpu_sum": 8386560.0, "expected_sum": 8386560.0, "gpu_sum": 8386560.0, "physical_gpu_index": 0, "reasons": [], "rmax": 0.9999771118164062, "rmean": -0.014401411637663841, "rmin": -0.9996249675750732, "rvar": 0.3416896462440491}, {"corrupt": false, "cpu_sum": 8386560.0, "expected_sum": 8386560.0, "gpu_sum": 8386560.0, "physical_gpu_index": 0, "reasons": [], "rmax": 0.9999771118164062, "rmean": -0.014401411637663841, "rmin": -0.9996249675750732, "rvar": 0.3416896462440491}]
# Test Failed ❌
# exit_code=0


# ENV: {"cuda_visible_devices": "0", "gpu_free_mib_before_alloc": 24243, "gpu_total_mib": 24576, "pid": 923730, "python": "3.10.20", "resolved_physical_gpu_index": 0, "tf_cpp_min_log_level": "3"}
# ALLOCATOR_STATUS: {"allocated_by": "cupy", "leave_free_mib": 2048, "ok": true, "physical_gpu_index": 0, "target_alloc_mib": 20606}
# WORKER_RESULTS: [{"corrupt": false, "cpu_sum": 8386560.0, "expected_sum": 8386560.0, "gpu_sum": 8386560.0, "physical_gpu_index": 0, "reasons": [], "rmax": 0.9999771118164062, "rmean": -0.014401411637663841, "rmin": -0.9996249675750732, "rvar": 0.3416896462440491}, {"corrupt": false, "cpu_sum": 8386560.0, "expected_sum": 8386560.0, "gpu_sum": 8386560.0, "physical_gpu_index": 0, "reasons": [], "rmax": 0.9999771118164062, "rmean": -0.014401411637663841, "rmin": -0.9996249675750732, "rvar": 0.3416896462440491}]
# Test Failed ❌
# exit_code=0


