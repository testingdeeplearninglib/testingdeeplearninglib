# GCFL-OTHER-0100

import sys
import random


def _print_and_exit(msg: str, code: int = 0):
    print(msg)
    sys.exit(code)


def _skip(reason: str):
    _print_and_exit(f"SKIP_ENV: {reason}", 0)


def _pass():
    _print_and_exit("Test Passed ✅", 0)


def _fail():
    _print_and_exit("Test Failed ❌", 0)


def main():
    try:
        try:
            import numpy as np
        except Exception as e:
            _skip(f"missing numpy ({e})")

        try:
            import tensorflow as tf
        except Exception as e:
            _skip(f"missing tensorflow ({e})")

        # Configure GPU memory growth before runtime gets fully initialized.
        try:
            gpus = tf.config.list_physical_devices("GPU")
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
        except Exception:
            pass

        # Determinism
        seed = 1337
        random.seed(seed)
        np.random.seed(seed)
        try:
            tf.random.set_seed(seed)
        except Exception:
            pass

        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass

        DIM = 32
        N_BLOCKS = 4
        DEPTH_PER_BLOCK = 2

        def _make_dense(units: int, act, layer_seed: int, name: str):
            return tf.keras.layers.Dense(
                units,
                activation=act,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=layer_seed),
                bias_initializer=tf.keras.initializers.Zeros(),
                name=name,
            )

        def _make_block(depth: int, dim: int, start_seed: int, name_prefix: str):
            layers = []
            s = start_seed
            for d in range(depth):
                act = "relu" if d < depth - 1 else None
                layers.append(_make_dense(dim, act, s, name=f"{name_prefix}_dense{d}"))
                s += 1
            return tf.keras.Sequential(layers, name=f"{name_prefix}_seq"), s

        class ResidualPlus(tf.keras.layers.Layer):
            def __init__(self, n_blocks: int, depth: int, dim: int, base_seed: int = 1000, **kwargs):
                super().__init__(**kwargs)
                self.blocks = []
                s = base_seed
                for b in range(n_blocks):
                    block, s = _make_block(depth, dim, s, name_prefix=f"blk{b}")
                    self.blocks.append(block)

            def call(self, x, training=False):
                for block in self.blocks:
                    h = block(x, training=training)
                    # Keep the exact suspected trigger pattern.
                    x = x + h
                return x

        class ResidualAdd(tf.keras.layers.Layer):
            def __init__(self, n_blocks: int, depth: int, dim: int, base_seed: int = 1000, **kwargs):
                super().__init__(**kwargs)
                self.blocks = []
                self.adds = []
                s = base_seed
                for b in range(n_blocks):
                    block, s = _make_block(depth, dim, s, name_prefix=f"blk{b}")
                    self.blocks.append(block)
                    self.adds.append(tf.keras.layers.Add(name=f"res_add_{b}"))

            def call(self, x, training=False):
                for block, add in zip(self.blocks, self.adds):
                    h = block(x, training=training)
                    x = add([x, h])
                return x

        def build_model(res_layer, final_seed: int, name: str):
            inp = tf.keras.Input(shape=(DIM,), dtype=tf.float32, name=f"{name}_in")
            x = res_layer(inp)
            out = tf.keras.layers.Dense(
                1,
                activation=None,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=final_seed),
                bias_initializer=tf.keras.initializers.Zeros(),
                name=f"{name}_out",
            )(x)
            return tf.keras.Model(inp, out, name=name)

        plus_layer = ResidualPlus(
            N_BLOCKS,
            DEPTH_PER_BLOCK,
            DIM,
            base_seed=2000,
            name="res_plus",
        )
        add_layer = ResidualAdd(
            N_BLOCKS,
            DEPTH_PER_BLOCK,
            DIM,
            base_seed=2000,
            name="res_add",
        )

        plus_model = build_model(plus_layer, final_seed=3000, name="plus_model")
        add_model = build_model(add_layer, final_seed=3000, name="add_model")

        def _flatten_layers(obj):
            try:
                return list(obj._flatten_layers(include_self=False, recursive=True))
            except Exception:
                try:
                    return list(obj.layers)
                except Exception:
                    return []

        def _is_add_like(layer) -> bool:
            try:
                if isinstance(layer, tf.keras.layers.Add):
                    return True
            except Exception:
                pass

            cls = layer.__class__.__name__.lower()
            lname = str(getattr(layer, "name", "")).lower()

            if "add" in cls:
                return True
            if "add" in lname:
                return True
            if "__operators__.add" in lname or "tf.__operators__.add" in lname:
                return True
            return False

        # Oracle A: only flag graph anomaly if an actual add-like layer instance
        # appears to be reused across multiple inbound nodes.
        graph_bug = False
        plus_flat_layers = _flatten_layers(plus_model)
        plus_add_like = [l for l in plus_flat_layers if _is_add_like(l)]
        for l in plus_add_like:
            inbound = getattr(l, "_inbound_nodes", None)
            try:
                if inbound is not None and len(inbound) > 1:
                    graph_bug = True
                    break
            except Exception:
                pass

        x_np = np.random.RandomState(2021).normal(size=(4, DIM)).astype(np.float32)
        y_np = np.random.RandomState(2022).normal(size=(4, 1)).astype(np.float32)

        x = tf.convert_to_tensor(x_np)
        y = tf.convert_to_tensor(y_np)

        def compute_preds_and_grads(model):
            with tf.GradientTape() as tape:
                preds = model(x, training=True)
                loss = tf.reduce_mean(tf.square(preds - y))
            vars_ = list(model.trainable_variables)
            grads = tape.gradient(loss, vars_)
            return float(loss.numpy()), preds.numpy(), vars_, grads

        loss_plus, pred_plus, vars_plus, grads_plus = compute_preds_and_grads(plus_model)
        loss_add, pred_add, vars_add, grads_add = compute_preds_and_grads(add_model)

        out_diff = float(np.max(np.abs(pred_plus - pred_add)))
        loss_diff = abs(loss_plus - loss_add)

        plus_shapes = [tuple(v.shape) for v in vars_plus]
        add_shapes = [tuple(v.shape) for v in vars_add]
        structural_mismatch = plus_shapes != add_shapes

        grad_mismatch = False
        if not structural_mismatch:
            none_plus = sum(g is None for g in grads_plus)
            none_add = sum(g is None for g in grads_add)

            if none_plus != none_add:
                grad_mismatch = True
            else:
                max_rel = 0.0
                for gp, ga in zip(grads_plus, grads_add):
                    if gp is None or ga is None:
                        continue

                    gp_np = gp.numpy()
                    ga_np = ga.numpy()

                    if gp_np.shape != ga_np.shape:
                        grad_mismatch = True
                        break

                    diff = gp_np - ga_np
                    n_ga = float(np.linalg.norm(ga_np.reshape(-1)))
                    n_diff = float(np.linalg.norm(diff.reshape(-1)))
                    rel = n_diff / (n_ga + 1e-12)
                    if rel > max_rel:
                        max_rel = rel

                # Reproduce only when forward behavior is essentially identical
                # but gradients diverge meaningfully.
                if not grad_mismatch:
                    if loss_diff < 1e-6 and out_diff < 1e-5 and max_rel > 1e-3:
                        grad_mismatch = True

        bug_reproduced = bool(graph_bug or structural_mismatch or grad_mismatch)

        if bug_reproduced:
            _pass()
        else:
            _fail()

    except SystemExit:
        raise
    except Exception as e:
        _print_and_exit(f"HARNESS_ERROR: {type(e).__name__}: {e}", 1)


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
# python -V
# which python

# python - <<'PY'
# import tensorflow as tf
# print("TF:", tf.__version__)
# print("Visible GPUs:", tf.config.list_physical_devices("GPU"))
# PY

# mkdir -p logs/GCFL-OTHER-0099

# export CUDA_VISIBLE_DEVICES=0
# export PYTHONUNBUFFERED=1
# export TF_CPP_MIN_LOG_LEVEL=1
# export TF_DETERMINISTIC_OPS=1

# set -o pipefail
# python testcase/tensorflow_testcase.py 2>&1 | tee logs/GCFL-OTHER-0099/run.log
# echo "exit_code=$?"


# Output:
# *****************
# Python 3.11.15
# /home/talha/miniconda3/envs/tf_venv/bin/python

# TF: 2.21.0
# Visible GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

# ZERO_RC=0
# ZERO_STDOUT=EXC:ValueError
# NEG_RC=0
# NEG_STDOUT=EXC:ValueError
# Test Failed ❌
# exit_code=0