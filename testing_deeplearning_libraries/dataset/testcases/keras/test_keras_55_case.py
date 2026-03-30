# GCFL-OTHER-0055

import os
import sys
import time
import random
import warnings


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


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["KERAS_BACKEND"] = "tensorflow"
warnings.filterwarnings("ignore")


def _parse_version_prefix(v: str):
    parts = []
    for token in v.split("."):
        num = ""
        for ch in token:
            if ch.isdigit():
                num += ch
            else:
                break
        parts.append(int(num) if num else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def _to_float_scalar(x):
    try:
        return float(x)
    except Exception:
        pass
    try:
        import numpy as np
        arr = np.asarray(x)
        if arr.size == 1:
            return float(arr.reshape(()))
    except Exception:
        pass
    try:
        if hasattr(x, "numpy"):
            return float(x.numpy())
    except Exception:
        pass
    raise TypeError(f"cannot convert {type(x).__name__} to float")


def _is_bad_number(x) -> bool:
    try:
        import numpy as np
        xf = _to_float_scalar(x)
        return (not np.isfinite(xf)) or np.isnan(xf)
    except Exception:
        return False


def main():
    try:
        try:
            import numpy as np
        except Exception as e:
            _skip(f"numpy not available: {e}")

        try:
            import tensorflow as tf
        except Exception as e:
            _skip(f"tensorflow not available: {e}")

        try:
            import keras
        except Exception as e:
            _skip(f"keras not available: {e}")

        try:
            import keras_hub
        except Exception as e:
            _skip(f"keras_hub not available: {e}")

        kv = getattr(keras, "__version__", "0.0.0")
        if _parse_version_prefix(str(kv)) < (3, 7, 0):
            _skip(f"keras version {kv} < 3.7")

        seed = 2021
        random.seed(seed)
        np.random.seed(seed)
        try:
            tf.random.set_seed(seed)
        except Exception:
            pass
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

        try:
            gpus = tf.config.list_physical_devices("GPU")
        except Exception:
            gpus = []
        if not gpus or len(gpus) < 2:
            _skip("need >=2 GPUs for MirroredStrategy multi-replica")

        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

        try:
            strategy = tf.distribute.MirroredStrategy()
        except Exception as e:
            _skip(f"MirroredStrategy unavailable: {e}")

        replicas = getattr(strategy, "num_replicas_in_sync", 0)
        if replicas < 2:
            _skip("MirroredStrategy has <2 replicas (check CUDA_VISIBLE_DEVICES)")

        IMAGE_SIZE = int(os.environ.get("GCFL_IMAGE_SIZE", "512"))
        NUM_CLASSES = int(os.environ.get("GCFL_NUM_CLASSES", "2"))
        MAX_STEPS = int(os.environ.get("GCFL_MAX_STEPS", "140"))
        TIME_LIMIT_SEC = int(os.environ.get("GCFL_TIME_LIMIT_SEC", "240"))

        if NUM_CLASSES < 2:
            _skip("NUM_CLASSES must be >= 2")
        if MAX_STEPS < 1:
            _skip("MAX_STEPS must be >= 1")

        bs_env = os.environ.get("GCFL_BATCH_SIZE", "").strip()
        if bs_env == "":
            BATCH_SIZE = int(replicas)
        else:
            BATCH_SIZE = int(bs_env)
        if BATCH_SIZE < replicas:
            BATCH_SIZE = int(replicas)
        if BATCH_SIZE % replicas != 0:
            BATCH_SIZE = max(int(replicas), BATCH_SIZE - (BATCH_SIZE % replicas))

        num_examples = MAX_STEPS * BATCH_SIZE

        def _make_sample(i):
            i32 = tf.cast(i, tf.int32)
            s_img = tf.stack([i32, tf.constant(seed, tf.int32)])
            s_msk = tf.stack([i32, tf.constant(seed + 999, tf.int32)])

            img = tf.random.stateless_uniform(
                shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                seed=s_img,
                minval=0,
                maxval=256,
                dtype=tf.int32,
            )
            img = tf.cast(img, tf.uint8)

            mask = tf.random.stateless_uniform(
                shape=(IMAGE_SIZE, IMAGE_SIZE),
                seed=s_msk,
                minval=0,
                maxval=NUM_CLASSES,
                dtype=tf.int32,
            )
            mask_oh = tf.one_hot(mask, depth=NUM_CLASSES, dtype=tf.float32)
            return img, mask_oh

        ds = tf.data.Dataset.range(num_examples)
        ds = ds.map(_make_sample, num_parallel_calls=1, deterministic=True)
        ds = ds.batch(BATCH_SIZE, drop_remainder=True)
        ds = ds.prefetch(1)

        class _NaNDetector(keras.callbacks.Callback):
            def __init__(self, t0, time_limit_sec):
                super().__init__()
                self.reproduced = False
                self.t0 = t0
                self.time_limit_sec = time_limit_sec

            def on_train_batch_end(self, batch, logs=None):
                if (time.time() - self.t0) > self.time_limit_sec:
                    self.model.stop_training = True
                    return
                if not logs:
                    return
                for _, v in logs.items():
                    if v is None:
                        continue
                    if _is_bad_number(v):
                        self.reproduced = True
                        self.model.stop_training = True
                        return

        t0 = time.time()

        def _offline_resnet50_like_backbone():
            # Direct construction avoids preset downloads. (ResNetBackbone supports this.) :contentReference[oaicite:0]{index=0}
            return keras_hub.models.ResNetBackbone(
                input_conv_filters=[64],
                input_conv_kernel_sizes=[7],
                stackwise_num_filters=[64, 128, 256, 512],
                stackwise_num_blocks=[3, 4, 6, 3],
                stackwise_num_strides=[1, 2, 2, 2],
                block_type="bottleneck_block",
                use_pre_activation=False,
                image_shape=(None, None, 3),
                data_format="channels_last",
            )

        preset = os.environ.get("GCFL_IMAGE_ENCODER_PRESET", "").strip()
        require_preset = os.environ.get("GCFL_REQUIRE_PRESET", "0").strip() == "1"
        debug = os.environ.get("GCFL_DEBUG", "0").strip() == "1"

        try:
            with strategy.scope():
                image_converter = keras_hub.layers.DeepLabV3ImageConverter(
                    image_size=(IMAGE_SIZE, IMAGE_SIZE),
                    interpolation="bilinear",
                    data_format="channels_last",
                )
                preprocessor = keras_hub.models.DeepLabV3ImageSegmenterPreprocessor(image_converter)

                preset_used = False
                if preset:
                    try:
                        image_encoder = keras_hub.models.ResNetBackbone.from_preset(
                            preset,
                            load_weights=False,
                        )
                        preset_used = True
                    except Exception as e:
                        if require_preset:
                            _skip(f"preset load failed: {type(e).__name__}: {e}")
                        image_encoder = _offline_resnet50_like_backbone()
                else:
                    image_encoder = _offline_resnet50_like_backbone()

                deeplab_backbone = keras_hub.models.DeepLabV3Backbone(
                    image_encoder=image_encoder,
                    low_level_feature_key="P2",
                    spatial_pyramid_pooling_key="P5",
                    dilation_rates=[6, 12, 18],
                    upsampling_size=8,
                )
                model = keras_hub.models.DeepLabV3ImageSegmenter(
                    backbone=deeplab_backbone,
                    num_classes=NUM_CLASSES,
                    activation="softmax",
                    preprocessor=preprocessor,
                )

                initial_lr = 0.007 * float(BATCH_SIZE) / 16.0
                learning_rate = keras.optimizers.schedules.CosineDecay(
                    initial_lr,
                    decay_steps=max(1, MAX_STEPS),
                )

                model.compile(
                    optimizer=keras.optimizers.SGD(
                        learning_rate=learning_rate,
                        weight_decay=0.0001,
                        momentum=0.9,
                        clipnorm=10.0,
                    ),
                    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                    metrics=[
                        keras.metrics.MeanIoU(
                            num_classes=NUM_CLASSES,
                            sparse_y_true=False,
                            sparse_y_pred=False,
                        ),
                        keras.metrics.CategoricalAccuracy(),
                    ],
                )
        except tf.errors.ResourceExhaustedError as e:
            _skip(f"OOM / insufficient device memory: {e}")
        except Exception as e:
            _skip(f"model build/compile failed: {type(e).__name__}: {e}")

        if debug:
            print(f"DEBUG: replicas={replicas} batch={BATCH_SIZE} image={IMAGE_SIZE} preset='{preset or 'OFFLINE'}' preset_used={preset_used}")

        detector = _NaNDetector(t0=t0, time_limit_sec=TIME_LIMIT_SEC)

        try:
            model.fit(
                ds,
                epochs=1,
                steps_per_epoch=MAX_STEPS,
                verbose=0,
                callbacks=[detector],
            )
        except tf.errors.ResourceExhaustedError as e:
            _skip(f"OOM during fit: {e}")
        except Exception as e:
            _skip(f"fit failed (not a NaN verdict): {type(e).__name__}: {e}")

        if detector.reproduced:
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

# Keras


# Commands
# *****************
# conda activate keras_nightly

# python - <<'PY'
# import tensorflow as tf
# import keras
# import keras_hub
# print("tf:", tf.__version__)
# print("keras:", keras.__version__)
# print("keras_hub:", getattr(keras_hub, "__version__", "unknown"))
# print("gpus:", tf.config.list_physical_devices("GPU"))
# PY


# Output:
# *****************
# Traceback (most recent call last):
#   ...
#   File ".../keras/src/tree/torchtree_impl.py", line 3, in <module>
#     from torch.utils import _pytree as torch_tree
# ModuleNotFoundError: No module named 'torch'

# Test Failed ❌