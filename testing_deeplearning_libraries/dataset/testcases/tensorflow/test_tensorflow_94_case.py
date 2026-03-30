# GCFL-SPARSE-0094

# GCFL-SPARSE-0094

import os
import sys
import random


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


# Determinism
os.environ.setdefault("PYTHONHASHSEED", "0")
random.seed(0)

try:
    import numpy as np
except Exception as e:
    _skip(f"numpy not available: {e}")

np.random.seed(0)

try:
    import tensorflow as tf
except Exception as e:
    _skip(f"tensorflow not available: {e}")

try:
    tf.random.set_seed(0)
except Exception:
    pass


def _print_env():
    try:
        gpus = [d.name for d in tf.config.list_physical_devices("GPU")]
    except Exception:
        gpus = []

    try:
        tpus = [d.name for d in tf.config.list_physical_devices("TPU")]
    except Exception:
        tpus = []

    print(
        "ENV: "
        f"python={sys.version.split()[0]} "
        f"tf={getattr(tf, '__version__', 'unknown')} "
        f"eager={tf.executing_eagerly()} "
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
        f"gpus={gpus} "
        f"tpus={tpus}"
    )


def _msg_contains_dense_to_dense_set_op(err: BaseException) -> bool:
    msg = f"{type(err).__name__}: {err}"
    needles = [
        "DenseToDenseSetOperation",
        "No registered 'DenseToDenseSetOperation'",
        "OpKernel for XLA_TPU_JIT",
        "XLA_TPU_JIT",
        "broadcast_weights/assert_broadcastable",
    ]
    return any(n in msg for n in needles) and (
        "DenseToDenseSetOperation" in msg
        or "No registered 'DenseToDenseSetOperation'" in msg
    )


def _cpu_gpu_baseline_should_work():
    vocab = 17
    seq_len = 13
    batch = 2

    y_true = tf.random.uniform(
        [batch, seq_len], minval=0, maxval=vocab, dtype=tf.int32, seed=1
    )
    y_pred = tf.random.uniform(
        [batch, seq_len, vocab], minval=-1.0, maxval=1.0, dtype=tf.float32, seed=2
    )

    # Keep [batch, 1] intentionally to preserve the original broadcast path.
    sample_weight = tf.cast(
        tf.random.uniform([batch, 1], minval=1, maxval=3, dtype=tf.int32, seed=3),
        tf.float32,
    )

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE,
    )

    loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
    m = tf.reduce_mean(loss)

    if tf.executing_eagerly():
        _ = m.numpy()


def _tpu_repro_attempt():
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    except Exception as e:
        _skip(f"TPU not available or cannot be initialized: {type(e).__name__}: {e}")

    vocab = 17
    seq_len = 13
    per_replica_batch = 2

    with strategy.scope():
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE,
        )

        def step_fn():
            y_true = tf.random.uniform(
                [per_replica_batch, seq_len],
                minval=0,
                maxval=vocab,
                dtype=tf.int32,
                seed=11,
            )
            y_pred = tf.random.uniform(
                [per_replica_batch, seq_len, vocab],
                minval=-1.0,
                maxval=1.0,
                dtype=tf.float32,
                seed=12,
            )
            sample_weight = tf.cast(
                tf.random.uniform(
                    [per_replica_batch, 1],
                    minval=1,
                    maxval=3,
                    dtype=tf.int32,
                    seed=13,
                ),
                tf.float32,
            )
            loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
            return tf.reduce_mean(loss)

        @tf.function
        def distributed_step():
            per_replica_out = strategy.run(step_fn)
            return strategy.reduce(
                tf.distribute.ReduceOp.MEAN, per_replica_out, axis=None
            )

        try:
            out = distributed_step()
            try:
                _ = out.numpy()
            except Exception:
                tf.print(out)
            return None
        except Exception as e:
            return e


def main():
    _print_env()

    try:
        _cpu_gpu_baseline_should_work()
    except Exception as e:
        _harness_error(
            Exception(f"CPU/GPU baseline failed unexpectedly: {type(e).__name__}: {e}")
        )

    tpu_err = _tpu_repro_attempt()

    if tpu_err is None:
        _fail()

    if _msg_contains_dense_to_dense_set_op(tpu_err):
        _pass()
    else:
        _fail()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        _harness_error(e)




# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Tensorflow


# Commands
# *****************
# GPU run
# cd ~/dl_testing
# conda activate tf_venv
# mkdir -p logs

# set -o pipefail
# export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES=0

# python testcases/tensorflow_testcase.py 2>&1 | tee logs/GCFL-SPARSE-0094_gpu_run.log
# echo "exit_code=$?"


# CPU-only run
# cd ~/dl_testing
# conda activate tf_venv
# mkdir -p logs

# set -o pipefail
# export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES=""

# python testcases/tensorflow_testcase.py 2>&1 | tee logs/GCFL-SPARSE-0094_cpu_run.log
# echo "exit_code=$?"


# Output:
# *****************

# GPU run output
# ENV: python=3.11.15 tf=2.21.0 eager=True cuda_visible_devices=0 gpus=['/physical_device:GPU:0'] tpus=[]
# SKIP_ENV: TPU not available or cannot be initialized: ValueError: Please provide a TPU Name to connect to.
# exit_code=0

# CPU-only run output
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# E0000 00:00:1774276356.402831 1594808 cuda_platform.cc:52] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
# ENV: python=3.11.15 tf=2.21.0 eager=True cuda_visible_devices= gpus=[] tpus=[]
# SKIP_ENV: TPU not available or cannot be initialized: ValueError: Please provide a TPU Name to connect to.
# exit_code=0
# Test Failed ❌