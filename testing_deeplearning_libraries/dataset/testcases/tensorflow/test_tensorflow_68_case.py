# GCFL-DATAPIPELI-0068_tc_01


import os
import sys
import time
import random

SEED = 2021


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


def safe_import_tensorflow():
    try:
        import tensorflow as tf
        return tf
    except Exception as e:
        _skip(f"tensorflow_missing_or_unimportable: {type(e).__name__}: {e}")


def log_tf_env(tf):
    print(f"TF_VERSION: {getattr(tf, '__version__', 'unknown')}")
    try:
        gpus = tf.config.list_physical_devices("GPU")
        print(f"TF_GPUS: {gpus}")
    except Exception as e:
        print(f"TF_GPUS: <error: {type(e).__name__}: {e}>")

    try:
        with tf.device("/GPU:0"):
            x = tf.random.uniform([1024, 1024], dtype=tf.float32, seed=SEED)
            y = tf.linalg.matmul(x, x)
            _ = float(tf.reduce_sum(y).numpy())
        print("TF_GPU_SMOKE: OK")
    except Exception as e:
        print(f"TF_GPU_SMOKE: FAILED ({type(e).__name__}: {e})")


class ModelStub:
    def get_batch(self, dev_set, bucket_id):
        bucket = dev_set[bucket_id]
        if bucket is None or len(bucket) == 0:
            raise ValueError(f"Empty bucket encountered for bucket_id={bucket_id}")
        return bucket[:1]


def build_bucketed_dev_set():
    return [
        ["hello world"],
        ["short"],
        [],                  # EMPTY bucket
        ["another sample"],
    ]


def simulate_unguarded_eval_loop(dev_set, model):
    empty_bucket_ids = {i for i, b in enumerate(dev_set) if b is None or len(b) == 0}
    if not empty_bucket_ids:
        raise RuntimeError("dev_set has no empty bucket; expected at least one empty bucket")

    for bucket_id in range(len(dev_set)):
        try:
            _ = model.get_batch(dev_set, bucket_id)
        except Exception as e:
            print(f"OBSERVED_EXCEPTION: bucket_id={bucket_id} type={type(e).__name__} msg={e}")
            return bucket_id in empty_bucket_ids

    return False


def main():
    start = time.time()
    try:
        random.seed(SEED)

        tf = safe_import_tensorflow()
        log_tf_env(tf)

        dev_set = build_bucketed_dev_set()
        model = ModelStub()

        reproduced = simulate_unguarded_eval_loop(dev_set, model)
        print(f"RUNTIME_SEC: {time.time() - start:.3f}")

        if reproduced:
            _pass()
        else:
            _fail()

    except SystemExit:
        raise
    except BaseException as e:
        _harness_error(e)


if __name__ == "__main__":
    main()



# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Tensorflow


# Commands
# *****************
# source ~/.venvs/dl_testing/bin/activate
# export CUDA_VISIBLE_DEVICES=0
# export TF_CPP_MIN_LOG_LEVEL=0
# ts=$(date +%Y%m%d_%H%M%S)
# python testcases/tf_cases/tensorflow_testcase.py 2>&1 | tee "GCFL-DATAPIPELI-0068_tc_01_${ts}.log"



# Output:
# # *****************
# TF_VERSION: 2.20.0
# TF_GPUS: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
# TF_GPU_SMOKE: OK
# OBSERVED_EXCEPTION: bucket_id=2 type=ValueError msg=Empty bucket encountered for bucket_id=2
# RUNTIME_SEC: 3.403

# Test Failed ❌