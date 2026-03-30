# GCFL-TRACINGGRA-0096

import json
import os
import sys
import random
import shutil
import tempfile


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


def _set_determinism(seed: int = 1337):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def _print_env(tf):
    try:
        gpus = [d.name for d in tf.config.list_physical_devices("GPU")]
    except Exception:
        gpus = []
    env = {
        "python": sys.version.split()[0],
        "tensorflow": getattr(tf, "__version__", "unknown"),
        "keras": getattr(tf.keras, "__version__", "unknown"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "visible_gpus": gpus,
    }
    print(f"ENV: {json.dumps(env, sort_keys=True)}")


def _is_target_exception(e: BaseException) -> bool:
    msg = f"{type(e).__name__}: {e}".lower()
    ragged_related = "ragged" in msg
    tensor_contract_related = (
        "non-tensor" in msg
        or "got a non-tensor value" in msg
        or "must be a single tensor" in msg
        or "dictionary from string to tensor" in msg
        or "sequence of tensors" in msg
        or ("signature" in msg and "tensor" in msg)
        or ("signatures" in msg and "tensor" in msg)
    )
    return ragged_related and tensor_contract_related


def _build_model(tf):
    layers = tf.keras.layers
    inp = layers.Input(shape=(None, 4), ragged=True, name="inp")
    out = layers.Identity(name="identity")(inp)
    model = tf.keras.Model(inp, out, name="ragged_identity_model")
    return model


def _warmup_model(tf, model):
    sample = tf.ragged.constant(
        [
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            [[9.0, 10.0, 11.0, 12.0]],
        ],
        dtype=tf.float32,
    )
    _ = model(sample)


def _attempt_savedmodel_export(tf, model, export_dir: str):
    """
    Return:
      ("success", method_name, None)
      ("exception", method_name, exc)
      ("no_method", None, None)
    """
    attempts = []

    if hasattr(model, "export"):
        attempts.append(("model.export(export_dir)", lambda: model.export(export_dir)))

    if hasattr(tf.saved_model, "save"):
        attempts.append(("tf.saved_model.save(model, export_dir)", lambda: tf.saved_model.save(model, export_dir)))

    if not attempts:
        return ("no_method", None, None)

    last_exc = None
    for name, fn in attempts:
        try:
            if os.path.exists(export_dir):
                shutil.rmtree(export_dir, ignore_errors=True)
        except Exception:
            pass

        try:
            fn()
            return ("success", name, None)
        except Exception as e:
            last_exc = e
            return ("exception", name, e)

    return ("exception", attempts[-1][0], last_exc)


def main():
    root = None
    try:
        _set_determinism(1337)

        try:
            import tensorflow as tf
        except Exception as e:
            _skip(f"tensorflow not installed ({type(e).__name__}: {e})")

        try:
            tf.random.set_seed(1337)
        except Exception:
            pass

        _print_env(tf)

        model = _build_model(tf)
        _warmup_model(tf, model)

        root = tempfile.mkdtemp(prefix="gcfl_tracinggra_0096_")
        export_dir = os.path.join(root, "saved_model_export")

        status, method, exc = _attempt_savedmodel_export(tf, model, export_dir)

        if status == "no_method":
            _skip("No compatible SavedModel export method available in this TensorFlow/Keras build")

        if status == "success":
            _fail()

        if _is_target_exception(exc):
            _pass()

        _fail()

    except SystemExit:
        raise
    except Exception as e:
        _harness_error(e)
    finally:
        if root is not None:
            try:
                shutil.rmtree(root, ignore_errors=True)
            except Exception:
                pass


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
# export PYTHONUNBUFFERED=1
# export TF_CPP_MIN_LOG_LEVEL=1
# unset TF_XLA_FLAGS

# set -o pipefail
# python testcases/tensorflow_testcase.py 2>&1 | tee logs/GCFL-TRACINGGRA-0096/run.log
# echo "exit_code=$?"


# Output:
# *****************
# ENV: {"cuda_visible_devices": "0", "keras": "3.13.2", "python": "3.11.15", "tensorflow": "2.21.0", "visible_gpus": ["/physical_device:GPU:0"]}
# Saved artifact at '/tmp/gcfl_tracinggra_0096_yxag9tx5/saved_model_export'. The following endpoints are available:

# * Endpoint 'serve'
#   args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, None, 4), dtype=tf.float32, name='inp')
# Output Type:
#   TensorSpec(shape=(None, None, 4), dtype=tf.float32, name=None)
# Captures:
#   None
# Test Failed ❌
# exit_code=0