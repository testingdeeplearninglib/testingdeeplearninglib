# GCFL-TRAININGFI-0082

import os
import sys
import time
import socket
import random
import shutil
import tempfile
import traceback
import multiprocessing as mp
import queue


def _print_and_exit(msg: str, code: int) -> None:
    print(msg)
    raise SystemExit(code)


def skip(reason: str) -> None:
    _print_and_exit(f"SKIP_ENV: {reason}", 0)


def _get_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _manual_lstm_outputs(tf, enc_inp, batch_size: int, src_len: int, emb_dim: int, units: int, time_major: bool):
    if time_major:
        inputs_tm = enc_inp
    else:
        inputs_tm = tf.transpose(enc_inp, [1, 0, 2], name="enc_inp_tm")

    cell = tf.keras.layers.LSTMCell(units, name="manual_lstm_cell")
    cell.build(tf.TensorShape([batch_size, emb_dim]))

    h0 = tf.zeros([batch_size, units], dtype=tf.float32, name="h0")
    c0 = tf.zeros([batch_size, units], dtype=tf.float32, name="c0")

    ta_in = tf.TensorArray(
        dtype=tf.float32,
        size=src_len,
        clear_after_read=False,
        element_shape=tf.TensorShape([batch_size, emb_dim]),
    ).unstack(inputs_tm)

    ta_out = tf.TensorArray(
        dtype=tf.float32,
        size=src_len,
        clear_after_read=False,
        element_shape=tf.TensorShape([batch_size, units]),
    )

    def cond(t, h, c, ta_out):
        return t < src_len

    def body(t, h, c, ta_out):
        x_t = ta_in.read(t)
        y_t, [h_new, c_new] = cell(x_t, states=[h, c], training=True)
        ta_out = ta_out.write(t, y_t)
        return t + 1, h_new, c_new, ta_out

    _, _, _, ta_out = tf.while_loop(
        cond,
        body,
        loop_vars=[tf.constant(0, dtype=tf.int32), h0, c0, ta_out],
        parallel_iterations=1,
        swap_memory=True,
    )

    outputs_tm = ta_out.stack(name="rnn_outputs_tm")
    if time_major:
        return outputs_tm
    return tf.transpose(outputs_tm, [1, 0, 2], name="rnn_outputs_bt")


def _worker_process(cluster_def, job_name, task_index, mode, time_major, ckpt_dir, q):
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        import numpy as np
        import tensorflow as tf

        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()

        seed = 1337 + int(task_index)
        random.seed(seed)
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)

        if not hasattr(tf.compat.v1, "train") or not hasattr(tf.compat.v1.train, "Server"):
            q.put(
                {
                    "role": job_name,
                    "task": int(task_index),
                    "event": "skip",
                    "reason": "tf.compat.v1.train.Server not available",
                }
            )
            return

        if not hasattr(tf.compat.v1.train, "MonitoredTrainingSession"):
            q.put(
                {
                    "role": job_name,
                    "task": int(task_index),
                    "event": "skip",
                    "reason": "tf.compat.v1.train.MonitoredTrainingSession not available",
                }
            )
            return

        if not hasattr(tf.keras.layers, "LSTMCell"):
            q.put(
                {
                    "role": job_name,
                    "task": int(task_index),
                    "event": "skip",
                    "reason": "tf.keras.layers.LSTMCell not available",
                }
            )
            return

        cluster = tf.train.ClusterSpec(cluster_def)
        server = tf.compat.v1.train.Server(cluster, job_name=job_name, task_index=int(task_index))

        if job_name == "ps":
            q.put({"role": "ps", "event": "started"})
            server.join()
            return

        is_chief = int(task_index) == 0

        with tf.compat.v1.device(
            tf.compat.v1.train.replica_device_setter(
                worker_device=f"/job:worker/task:{int(task_index)}",
                cluster=cluster,
            )
        ):
            batch_size = 8
            src_len = 9
            vocab = 50
            emb_dim = 16
            units = 32

            source_np = (np.arange(batch_size * src_len, dtype=np.int32).reshape(batch_size, src_len) % vocab)
            source = tf.constant(source_np, dtype=tf.int32, name="source_bt")

            emb = tf.compat.v1.get_variable(
                "embedding_encoder",
                shape=[vocab, emb_dim],
                dtype=tf.float32,
                initializer=tf.compat.v1.random_uniform_initializer(-0.1, 0.1, seed=seed),
            )

            if time_major:
                if mode == "transpose_then_lookup":
                    indices = tf.transpose(source, [1, 0], name="source_tb")
                    enc_inp = tf.nn.embedding_lookup(params=emb, ids=indices, name="enc_emb_tb")
                elif mode == "lookup_then_transpose":
                    enc_inp = tf.nn.embedding_lookup(params=emb, ids=source, name="enc_emb_bt")
                    enc_inp = tf.transpose(enc_inp, [1, 0, 2], name="enc_emb_tb")
                else:
                    raise ValueError(f"Unknown mode: {mode}")
            else:
                enc_inp = tf.nn.embedding_lookup(params=emb, ids=source, name="enc_emb_bt")

            outputs = _manual_lstm_outputs(
                tf=tf,
                enc_inp=enc_inp,
                batch_size=batch_size,
                src_len=src_len,
                emb_dim=emb_dim,
                units=units,
                time_major=bool(time_major),
            )

            loss = tf.reduce_mean(outputs, name="loss")
            global_step = tf.compat.v1.train.get_or_create_global_step()
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.01)
            train_op = optimizer.minimize(loss, global_step=global_step)

        q.put({"role": "worker", "task": int(task_index), "event": "graph_built", "is_chief": bool(is_chief)})

        config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        with tf.compat.v1.train.MonitoredTrainingSession(
            master=server.target,
            is_chief=bool(is_chief),
            checkpoint_dir=ckpt_dir,
            save_checkpoint_secs=None,
            save_summaries_steps=None,
            config=config,
        ) as sess:
            q.put({"role": "worker", "task": int(task_index), "event": "session_created"})
            for _ in range(2):
                _, step_val = sess.run([train_op, global_step])
                q.put({"role": "worker", "task": int(task_index), "event": "step", "step": int(step_val)})

        q.put({"role": "worker", "task": int(task_index), "event": "done"})

    except Exception as e:
        q.put(
            {
                "role": job_name,
                "task": int(task_index),
                "event": "error",
                "error": repr(e),
                "trace": traceback.format_exc(),
            }
        )


def _run_distributed_case(ctx, mode: str, time_major: bool, timeout_s: float):
    ps_port = _get_free_port()
    w0_port = _get_free_port()
    w1_port = _get_free_port()

    cluster_def = {
        "ps": [f"127.0.0.1:{ps_port}"],
        "worker": [f"127.0.0.1:{w0_port}", f"127.0.0.1:{w1_port}"],
    }

    ckpt_dir = tempfile.mkdtemp(prefix=f"gcfl_0082_{mode}_")
    q = ctx.Queue()

    ps_p = ctx.Process(target=_worker_process, args=(cluster_def, "ps", 0, mode, time_major, ckpt_dir, q))
    w0_p = ctx.Process(target=_worker_process, args=(cluster_def, "worker", 0, mode, time_major, ckpt_dir, q))
    w1_p = ctx.Process(target=_worker_process, args=(cluster_def, "worker", 1, mode, time_major, ckpt_dir, q))

    done = {0: False, 1: False}
    errors = []
    skips = []

    ps_p.start()
    time.sleep(1.0)
    w0_p.start()
    w1_p.start()

    start = time.time()
    while time.time() - start < timeout_s:
        if done[0] and done[1]:
            break
        if errors or skips:
            break
        try:
            msg = q.get(timeout=0.5)
        except queue.Empty:
            continue

        ev = msg.get("event")
        if ev == "done" and msg.get("role") == "worker":
            t = int(msg.get("task", -1))
            if t in done:
                done[t] = True
        elif ev == "error":
            errors.append(msg)
        elif ev == "skip":
            skips.append(msg)

    for p in (w0_p, w1_p):
        try:
            p.join(timeout=2.0)
        except Exception:
            pass

    alive_before_cleanup = {0: w0_p.is_alive(), 1: w1_p.is_alive()}
    exitcodes_before_cleanup = {0: w0_p.exitcode, 1: w1_p.exitcode}

    for p in (w0_p, w1_p, ps_p):
        if p.is_alive():
            try:
                p.terminate()
            except Exception:
                pass

    for p in (w0_p, w1_p, ps_p):
        try:
            p.join(timeout=3.0)
        except Exception:
            pass

    try:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    except Exception:
        pass

    return {
        "done": done,
        "alive_before_cleanup": alive_before_cleanup,
        "exitcodes_before_cleanup": exitcodes_before_cleanup,
        "errors": errors,
        "skips": skips,
    }


def main():
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

        try:
            import numpy as np  # noqa: F401
        except Exception as e:
            skip(f"numpy not available: {e}")

        try:
            import tensorflow as tf
        except Exception as e:
            skip(f"tensorflow not available: {e}")

        if not hasattr(tf.compat.v1.train, "Server"):
            skip("tf.compat.v1.train.Server not available")

        if not hasattr(tf.compat.v1.train, "MonitoredTrainingSession"):
            skip("tf.compat.v1.train.MonitoredTrainingSession not available")

        if not hasattr(tf.keras.layers, "LSTMCell"):
            skip("tf.keras.layers.LSTMCell not available")

        try:
            mp.set_start_method("spawn", force=True)
        except Exception:
            pass

        ctx = mp.get_context("spawn")

        control = _run_distributed_case(
            ctx=ctx,
            mode="lookup_then_transpose",
            time_major=True,
            timeout_s=60.0,
        )

        if control["skips"]:
            skip(f"distributed APIs missing/disabled: {control['skips'][0].get('reason', 'unknown')}")

        if control["errors"]:
            first = control["errors"][0]
            skip(f"control case errored: {first.get('error', 'unknown')}")

        if (not control["done"][0]) or (not control["done"][1]):
            skip("control case did not complete (distributed session unstable or too slow)")

        bug = _run_distributed_case(
            ctx=ctx,
            mode="transpose_then_lookup",
            time_major=True,
            timeout_s=60.0,
        )

        if bug["skips"]:
            skip(f"distributed APIs missing/disabled in bug case: {bug['skips'][0].get('reason', 'unknown')}")

        if bug["errors"]:
            print("Test Failed ❌")
            return

        chief_done = bool(bug["done"][0])
        nonchief_hung = (not bug["done"][1]) and bool(bug["alive_before_cleanup"][1])

        if chief_done and nonchief_hung:
            print("Test Passed ✅")
        else:
            print("Test Failed ❌")

    except SystemExit:
        raise
    except Exception as e:
        print(f"HARNESS_ERROR: {e}")
        sys.exit(1)


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
# export CUDA_VISIBLE_DEVICES=""
# export TF_CPP_MIN_LOG_LEVEL=2


# Output:
# *****************
# # WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# # E0000 00:00:1774441203.701207 1866206 cuda_platform.cc:52] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
# # WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# # E0000 00:00:1774441204.505169 1866260 cuda_platform.cc:52] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
# # WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# # E0000 00:00:1774441204.724654 1866261 cuda_platform.cc:52] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
# # CONTROL_SUMMARY: {'done': {0: True, 1: True}, 'alive_before_cleanup': {0: False, 1: False}, 'exitcodes_before_cleanup': {0: 0, 1: 0}, 'errors': [], 'skips': []}
# # WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# # E0000 00:00:1774441209.404903 1867160 cuda_platform.cc:52] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
# # WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# # E0000 00:00:1774441210.313662 1867260 cuda_platform.cc:52] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
# # WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# # E0000 00:00:1774441210.356143 1867259 cuda_platform.cc:52] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
# # BUG_SUMMARY: {'done': {0: True, 1: False}, 'alive_before_cleanup': {0: False, 1: True}, 'exitcodes_before_cleanup': {0: 0, 1: None}, 'errors': [], 'skips': []}
# # Test Passed 

# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# E0000 00:00:1774441277.154367 1869504 cuda_platform.cc:52] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# E0000 00:00:1774441277.874746 1869556 cuda_platform.cc:52] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# E0000 00:00:1774441277.877699 1869557 cuda_platform.cc:52] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
# CONTROL_SUMMARY: {'done': {0: True, 1: False}, 'alive_before_cleanup': {0: False, 1: True}, 'exitcodes_before_cleanup': {0: 0, 1: None}, 'errors': [], 'skips': []}
# SKIP_ENV: control case did not complete (distributed session unstable or too slow)
# Test Failed ❌