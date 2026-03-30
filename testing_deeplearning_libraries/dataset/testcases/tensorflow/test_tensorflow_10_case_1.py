# GCFL-SERIALIZATI-0010

import json
import os
import platform
import random
import sys
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


# Best-effort noise reduction.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

try:
    import numpy as np
except Exception as e:
    _skip(f"missing numpy: {type(e).__name__}: {e}")

try:
    import tensorflow as tf
except Exception as e:
    _skip(f"missing tensorflow: {type(e).__name__}: {e}")

try:
    import keras
    from keras import layers, ops
except Exception as e:
    _skip(f"missing keras: {type(e).__name__}: {e}")

try:
    import h5py  # noqa: F401
except Exception as e:
    _skip(f"missing h5py for .weights.h5: {type(e).__name__}: {e}")

try:
    try:
        from keras.saving import register_keras_serializable
    except Exception:
        from keras.utils import register_keras_serializable
except Exception as e:
    _skip(f"cannot import register_keras_serializable: {type(e).__name__}: {e}")

# Require Keras v3.
try:
    ver = getattr(keras, "__version__", "")
    major = int(str(ver).split(".")[0]) if ver else None
    if major is None or major < 3:
        _skip(f"requires keras>=3, found keras=={ver!r}")
except Exception as e:
    _skip(f"cannot determine keras version: {type(e).__name__}: {e}")

# Determinism.
random.seed(0)
np.random.seed(0)
try:
    tf.random.set_seed(0)
except Exception:
    pass

try:
    tf.get_logger().setLevel("ERROR")
except Exception:
    pass

try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass


def _print_env():
    try:
        gpu_devices = tf.config.list_physical_devices("GPU")
        gpu_names = [d.name for d in gpu_devices]
    except Exception:
        gpu_devices = []
        gpu_names = []

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "tf": getattr(tf, "__version__", "unknown"),
        "keras": getattr(keras, "__version__", "unknown"),
        "numpy": getattr(np, "__version__", "unknown"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_count": len(gpu_devices),
        "gpus": gpu_names,
    }
    print("ENV:", json.dumps(info, sort_keys=True))


@register_keras_serializable(package="gcfl")
class SubModel(keras.Model):
    def __init__(self, num_layer_norm, name_prefix="sub", **kwargs):
        super().__init__(**kwargs)
        self.num_layer_norm = int(num_layer_norm)
        self.name_prefix = str(name_prefix)

        self.dense_layers = [
            layers.Dense(64, activation="relu", name=f"{self.name_prefix}_dense_{i}")
            for i in range(self.num_layer_norm)
        ]
        self.layer_norms = [
            layers.LayerNormalization(name=f"{self.name_prefix}_ln_{i}")
            for i in range(self.num_layer_norm)
        ]
        self.output_layer = layers.Dense(32, activation="relu", name=f"{self.name_prefix}_out")

    def call(self, inputs):
        x = inputs
        for i, (dense, layer_norm) in enumerate(zip(self.dense_layers, self.layer_norms)):
            # Intentional trigger: first two children never execute on the normal path.
            if i < 2:
                continue
            x = dense(x)
            x = layer_norm(x)
        return self.output_layer(x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_layer_norm": self.num_layer_norm,
            "name_prefix": self.name_prefix,
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package="gcfl")
class CombinedModel(keras.Model):
    def __init__(self, num_submodels, num_layer_norm, **kwargs):
        super().__init__(**kwargs)
        self.num_submodels = int(num_submodels)
        self.num_layer_norm = int(num_layer_norm)

        self.submodels = [
            SubModel(self.num_layer_norm, name_prefix=f"sm{j}", name=f"submodel_{j}")
            for j in range(self.num_submodels)
        ]
        self.combine_layer = layers.Dense(10, activation="softmax", name="combined_dense")

    def call(self, inputs):
        outs = [sm(inputs) for sm in self.submodels]
        combined = ops.concatenate(outs, axis=-1)
        return self.combine_layer(combined)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_submodels": self.num_submodels,
            "num_layer_norm": self.num_layer_norm,
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _problematic_build_only(model, input_shape):
    # Deliberately use build() only, to preserve the suspected bad state where
    # parent build marks things built without creating all child variables.
    model.build(input_shape)


def _force_build_skipped_layers_only(model, x):
    """
    Build only the intentionally skipped prefix (first two Dense + LN pairs)
    so that save_weights() includes variables that the problematic build path
    in model_b may not create.
    """
    # First, build the normal executed path and the final combine layer.
    _ = model(x)

    for sm in model.submodels:
        y = x
        # Only force-build the skipped prefix; do not run the whole chain again.
        for i in range(min(2, len(sm.dense_layers))):
            y = sm.dense_layers[i](y)
            y = sm.layer_norms[i](y)


def _matches_oracle_error(e: BaseException) -> bool:
    msg = " ".join(str(e).lower().split())

    strong_patterns = [
        ("was never built", "weights file lists"),
        ("doesn't have any variables", "weights file lists"),
        ("was never built", "doesn't have any variables"),
    ]
    for a, b in strong_patterns:
        if a in msg and b in msg:
            return True

    extra_hints = [
        "did not create the state of the child layer",
        "calling the parent's `build()` method did not create the state",
        "calling the parent's build() method did not create the state",
    ]
    return any(h in msg for h in extra_hints)


def main():
    _print_env()

    num_submodels = 5
    num_layer_norm = 10
    input_shape = (None, 100)
    x = tf.zeros((2, 100), dtype=tf.float32)

    with tempfile.TemporaryDirectory() as td:
        weights_path = os.path.join(td, "model.weights.h5")

        # Model A: create variables for all children, including skipped ones.
        model_a = CombinedModel(num_submodels, num_layer_norm, name="combined_model")
        _force_build_skipped_layers_only(model_a, x)
        print(f"INFO: model_a_weight_count={len(model_a.weights)}")

        try:
            model_a.save_weights(weights_path)
            print(f"INFO: saved_weights={weights_path}")
        except Exception as e:
            _skip(f"cannot save .weights.h5: {type(e).__name__}: {e}")

        # Model B: intentionally rely on build() only.
        model_b = CombinedModel(num_submodels, num_layer_norm, name="combined_model")
        _problematic_build_only(model_b, input_shape)
        print(f"INFO: model_b_weight_count_before_load={len(model_b.weights)}")

        try:
            model_b.load_weights(weights_path)
            print("INFO: load_weights completed without exception")
            _fail()
        except Exception as e:
            print(f"INFO: load_weights raised {type(e).__name__}: {e}")
            if _matches_oracle_error(e):
                _pass()
            _fail()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as e:
        _harness_error(e)
        
        

# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Tensorflow


# Commands
# *****************
# cd ~/dl_testing
# conda activate tf_venv
# export CUDA_VISIBLE_DEVICES=0
# export TF_CPP_MIN_LOG_LEVEL=2
# unset TF_XLA_FLAGS
# set -o pipefail

# python testcase/tensorflow_testcase.py 2>&1 | tee logs/GCFL-SERIALIZATI-0010/run.log
# echo "exit_code=$?"


# Output:
# *****************
# ENV: {"cuda_visible_devices": "0", "gpu_count": 1, "gpus": ["/physical_device:GPU:0"], "keras": "3.13.2", "numpy": "1.26.4", "platform": "Linux-6.8.0-31-generic-x86_64-with-glibc2.39", "python": "3.11.15", "tf": "2.21.0"}

# /home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/keras/src/layers/layer.py:424: UserWarning: `build()` was called on layer 'submodel_0', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
# /home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/keras/src/layers/layer.py:424: UserWarning: `build()` was called on layer 'submodel_1', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
# /home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/keras/src/layers/layer.py:424: UserWarning: `build()` was called on layer 'submodel_2', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
# /home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/keras/src/layers/layer.py:424: UserWarning: `build()` was called on layer 'submodel_3', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
# /home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/keras/src/layers/layer.py:424: UserWarning: `build()` was called on layer 'submodel_4', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.

# INFO: model_a_weight_count=212
# INFO: saved_weights=/tmp/tmpvbdjj9xa/model.weights.h5

# /home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/keras/src/layers/layer.py:424: UserWarning: `build()` was called on layer 'combined_model', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.

# INFO: model_b_weight_count_before_load=0
# INFO: load_weights completed without exception
# Test Failed ❌
# exit_code=0