# GCFL-SERIALIZATI-0001

import json
import os
import random
import sys
from typing import Any

def _skip(reason: str) -> None:
    print(f"SKIP_ENV: {reason}")
    sys.exit(0)

def _pass() -> None:
    print("Test Passed ✅")
    sys.exit(0)

def _fail() -> None:
    print("Test Failed ❌")
    sys.exit(0)

def _harness_error(e: BaseException) -> None:
    print(f"HARNESS_ERROR: {type(e).__name__}: {e}")
    sys.exit(1)

def _set_determinism() -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    random.seed(0)
    try:
        import numpy as np  # type: ignore
        np.random.seed(0)
    except Exception:
        pass

def _print_env(tf: Any) -> None:
    try:
        gpus = [d.name for d in tf.config.list_physical_devices("GPU")]
    except Exception:
        gpus = []
    env = {
        "python": sys.version.split()[0],
        "tf": getattr(tf, "__version__", "unknown"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_count": len(gpus),
        "gpus": gpus,
    }
    print("ENV:", json.dumps(env, sort_keys=True))

def main() -> None:
    _set_determinism()

    try:
        import numpy as np  # type: ignore
    except Exception as e:
        _skip(f"missing numpy: {type(e).__name__}: {e}")

    try:
        import tensorflow as tf  # type: ignore
    except Exception as e:
        _skip(f"missing tensorflow: {type(e).__name__}: {e}")

    try:
        try:
            tf.random.set_seed(0)
        except Exception:
            pass

        _print_env(tf)

        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(32,)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation="softmax"),
            ],
            name="gcfl_seq_model",
        )

        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Keep runtime tiny and deterministic.
        x = np.random.RandomState(0).random((8, 32)).astype("float32")
        x_tf = tf.constant(x)

        # Build original model in inference mode to avoid Dropout randomness.
        _ = model(x_tf, training=False)
        original_out = model(x_tf, training=False).numpy()
        original_weights = model.get_weights()
        config: Any = model.get_config()

        if not isinstance(config, dict):
            _harness_error(TypeError(f"unexpected config type: {type(config).__name__}"))

        # 1) Canonical path must work first, otherwise the oracle is weak.
        seq_model = tf.keras.Sequential.from_config(config)
        _ = seq_model(x_tf, training=False)
        seq_model.set_weights(original_weights)
        seq_out = seq_model(x_tf, training=False).numpy()

        np.testing.assert_allclose(original_out, seq_out, rtol=1e-6, atol=1e-6)

        # 2) Target path from the historical issue.
        try:
            generic_model = tf.keras.Model.from_config(config)
        except Exception as e:
            print(f"GENERIC_FROM_CONFIG_EXCEPTION: {type(e).__name__}: {e}")
            _pass()

        # If generic reconstruction succeeds, verify it is actually equivalent.
        _ = generic_model(x_tf, training=False)

        try:
            generic_model.set_weights(original_weights)
        except Exception as e:
            print(f"GENERIC_SET_WEIGHTS_EXCEPTION: {type(e).__name__}: {e}")
            _pass()

        generic_out = generic_model(x_tf, training=False).numpy()

        try:
            np.testing.assert_allclose(original_out, generic_out, rtol=1e-6, atol=1e-6)
        except AssertionError as e:
            print(f"GENERIC_OUTPUT_MISMATCH: {e}")
            _pass()

        # No failure or mismatch => bug did not reproduce.
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
# cd ~/dl_testing
# conda activate tf_venv
# export TF_CPP_MIN_LOG_LEVEL=2
# set -o pipefail
# python testcase/tensorflow_testcase.py 2>&1 | tee logs/GCFL-SERIALIZATI-0001/run.log
# echo "exit_code=$?"


# Output:
# *****************
# ENV: {"cuda_visible_devices": "0", "gpu_count": 1, "gpus": ["/physical_device:GPU:0"], "python": "3.11.15", "tf": "2.21.0"}
# INFO: Model.from_config rejected Sequential config: ValueError: Unrecognized keyword arguments passed to Model: {'layers': [{'module': 'keras.layers', 'class_name': 'InputLayer', 'config': {'batch_shape': (None, 32), 'dtype': 'float32', 'sparse': False, 'ragged': False, 'name': 'input_layer', 'optional': False}, 'registered_name': None}, {'module': 'keras.layers', 'class_name': 'Dense', 'config': {'name': 'dense', 'trainable': True, 'dtype': {'module': 'keras', 'class_name': 'DTypePolicy', 'config': {'name': 'float32'}, 'registered_name': None}, 'units': 128, 'activation': 'relu', 'use_bias': True, 'kernel_initializer': {'module': 'keras.initializers', 'class_name': 'GlorotUniform', 'config': {'seed': None}, 'registered_name': None}, 'bias_initializer': {'module': 'keras.initializers', 'class_name': 'Zeros', 'config': {}, 'registered_name': None}, 'kernel_regularizer': None, 'bias_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'quantization_config': None}, 'registered_name': None, 'build_config': {'input_shape': (None, 32)}}, {'module': 'keras.layers', 'class_name': 'Dropout', 'config': {'name': 'dropout', 'trainable': True, 'dtype': {'module': 'keras', 'class_name': 'DTypePolicy', 'config': {'name': 'float32'}, 'registered_name': None}, 'rate': 0.2, 'seed': None, 'noise_shape': None}, 'registered_name': None}, {'module': 'keras.layers', 'class_name': 'Dense', 'config': {'name': 'dense_1', 'trainable': True, 'dtype': {'module': 'keras', 'class_name': 'DTypePolicy', 'config': {'name': 'float32'}, 'registered_name': None}, 'units': 10, 'activation': 'softmax', 'use_bias': True, 'kernel_initializer': {'module': 'keras.initializers', 'class_name': 'GlorotUniform', 'config': {'seed': None}, 'registered_name': None}, 'bias_initializer': {'module': 'keras.initializers', 'class_name': 'Zeros', 'config': {}, 'registered_name': None}, 'kernel_regularizer': None, 'bias_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'quantization_config': None}, 'registered_name': None, 'build_config': {'input_shape': (None, 128)}}], 'build_input_shape': (None, 32)}
# Test Failed ❌
# exit_code=0