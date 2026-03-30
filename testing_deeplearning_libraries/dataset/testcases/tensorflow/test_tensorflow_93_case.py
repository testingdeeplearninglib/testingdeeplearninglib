# GCFL-TRAININGFI-0093

import os
import sys
import random


def _print_and_exit(msg: str, code: int) -> None:
    try:
        sys.stdout.write(str(msg).strip() + "\n")
        sys.stdout.flush()
    finally:
        raise SystemExit(code)


def skip(reason: str) -> None:
    _print_and_exit(f"SKIP_ENV: {reason}", 0)


def passed() -> None:
    _print_and_exit("Test Passed ✅", 0)


def failed() -> None:
    _print_and_exit("Test Failed ❌", 0)


def harness_error(exc: BaseException) -> None:
    _print_and_exit(f"HARNESS_ERROR: {type(exc).__name__}: {exc}", 1)


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
# Do NOT force CPU here. Control GPU/CPU from the launcher command.


def main() -> None:
    try:
        try:
            import numpy as np
        except Exception as e:
            skip(f"numpy not available: {type(e).__name__}: {e}")

        try:
            import tensorflow as tf
        except Exception as e:
            skip(f"tensorflow not available: {type(e).__name__}: {e}")

        seed = 1337
        random.seed(seed)
        np.random.seed(seed)
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception:
            try:
                tf.random.set_seed(seed)
            except Exception:
                pass

        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

        BOARD_SIZE = 9
        INPUT_DIM = BOARD_SIZE * 3
        N = 256
        EPOCHS = 30
        BATCH = 32

        rng = np.random.RandomState(seed)
        x = rng.uniform(-1.0, 1.0, size=(N, INPUT_DIM)).astype(np.float32)

        w_true = rng.normal(loc=0.0, scale=0.4, size=(INPUT_DIM, BOARD_SIZE)).astype(np.float32)
        b_true = rng.normal(loc=0.0, scale=0.1, size=(BOARD_SIZE,)).astype(np.float32)
        y_q = (x @ w_true + b_true).astype(np.float32)

        y_prob = np.zeros((N, BOARD_SIZE), dtype=np.float32)

        def build_model(build_seed: int):
            try:
                tf.keras.backend.clear_session()
            except Exception:
                pass

            random.seed(build_seed)
            np.random.seed(build_seed)
            try:
                tf.keras.utils.set_random_seed(build_seed)
            except Exception:
                try:
                    tf.random.set_seed(build_seed)
                except Exception:
                    pass

            inp = tf.keras.Input(shape=(INPUT_DIM,), name="input")
            x1 = tf.keras.layers.Dense(128, activation="relu", name="d1")(inp)
            x2 = tf.keras.layers.Dense(256, activation="relu", name="d2")(x1)
            x3 = tf.keras.layers.Dense(128, activation="relu", name="d3")(x2)
            q_values = tf.keras.layers.Dense(BOARD_SIZE, activation=None, name="q_values")(x3)
            probabilities = tf.keras.layers.Softmax(name="probabilities")(q_values)
            return tf.keras.Model(inputs=inp, outputs=[probabilities, q_values], name="multi_out_model")

        def compile_model(model, eager_like: bool):
            import inspect

            mse = tf.keras.losses.MeanSquaredError()
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

            try:
                sig = inspect.signature(model.compile)
                params = set(sig.parameters)
            except Exception:
                params = set()

            mode_kwargs_candidates = []
            if "experimental_run_tf_function" in params:
                mode_kwargs_candidates.append({"experimental_run_tf_function": (not eager_like)})
            if "run_eagerly" in params:
                mode_kwargs_candidates.append({"run_eagerly": bool(eager_like)})
            if not mode_kwargs_candidates:
                mode_kwargs_candidates.append({"run_eagerly": bool(eager_like)})

            loss_candidates = [
                [None, mse],
                {"probabilities": None, "q_values": mse},
            ]

            last_error = None
            for mode_kwargs in mode_kwargs_candidates:
                for loss_spec in loss_candidates:
                    kwargs = {
                        "optimizer": optimizer,
                        "loss": loss_spec,
                    }
                    if "jit_compile" in params:
                        kwargs["jit_compile"] = False
                    kwargs.update(mode_kwargs)

                    try:
                        model.compile(**kwargs)
                        return
                    except TypeError as e:
                        last_error = e
                        continue
                    except Exception as e:
                        last_error = e
                        lowered = str(e).lower()
                        if "none" in lowered and "loss" in lowered:
                            continue
                        skip(f"Failed to compile model with required loss structure: {type(e).__name__}: {e}")

            msg = f"{type(last_error).__name__}: {last_error}" if last_error is not None else "unknown compile error"
            skip(f"Keras/TF version does not support the required None-loss multi-output setup: {msg}")

        def extract_target_loss(eval_result):
            if isinstance(eval_result, dict):
                for key in ("q_values_loss", "loss"):
                    value = eval_result.get(key, None)
                    if value is not None:
                        value = float(value)
                        if value == value:
                            return value
                return None

            if isinstance(eval_result, (list, tuple)):
                if len(eval_result) >= 2:
                    candidate = float(eval_result[1])
                    if candidate == candidate:
                        return candidate
                if len(eval_result) >= 1:
                    candidate = float(eval_result[0])
                    if candidate == candidate:
                        return candidate
                return None

            try:
                value = float(eval_result)
                return value if value == value else None
            except Exception:
                return None

        def train_and_get_ratio(eager_like: bool):
            m = build_model(seed)
            compile_model(m, eager_like=eager_like)

            try:
                before_eval = m.evaluate(
                    x,
                    [y_prob, y_q],
                    batch_size=BATCH,
                    verbose=0,
                    return_dict=True,
                )
            except TypeError:
                before_eval = m.evaluate(
                    x,
                    [y_prob, y_q],
                    batch_size=BATCH,
                    verbose=0,
                )

            before = extract_target_loss(before_eval)
            if before is None or before <= 0.0:
                return None

            m.fit(
                x,
                [y_prob, y_q],
                epochs=EPOCHS,
                batch_size=BATCH,
                verbose=0,
                shuffle=False,
            )

            try:
                after_eval = m.evaluate(
                    x,
                    [y_prob, y_q],
                    batch_size=BATCH,
                    verbose=0,
                    return_dict=True,
                )
            except TypeError:
                after_eval = m.evaluate(
                    x,
                    [y_prob, y_q],
                    batch_size=BATCH,
                    verbose=0,
                )

            after = extract_target_loss(after_eval)
            if after is None or after != after:
                return None

            return after / before

        ratio_graph = train_and_get_ratio(eager_like=False)
        ratio_eager = train_and_get_ratio(eager_like=True)

        if ratio_graph is None or ratio_eager is None:
            skip("Could not obtain stable loss ratios for oracle comparison")

        eager_learns = ratio_eager <= 0.40
        graph_stalls = ratio_graph >= 0.85

        if eager_learns and graph_stalls:
            passed()
        else:
            failed()

    except SystemExit:
        raise
    except Exception as e:
        harness_error(e)


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

# python testcases/tensorflow_testcase.py > logs/bug_36044_cpu_stdout.log 2> logs/bug_36044_cpu_stderr.log
# echo "exit_code=$?"
# cat logs/bug_36044_cpu_stdout.log


# Output:
# *****************
# bug no: 36044
# Result: Test Failed ❌

# Triggering command:
# conda activate tf_venv
# cd ~/dl_testing
# export CUDA_VISIBLE_DEVICES=""
# export TF_CPP_MIN_LOG_LEVEL=2
# python testcases/tensorflow_testcase.py > logs/bug_36044_cpu_stdout.log 2> logs/bug_36044_cpu_stderr.log
# echo "exit_code=$?"
# cat logs/bug_36044_cpu_stdout.log

# Observed output:
# exit_code=0
# Test Failed ❌

# Note:
# GPU execution was unavailable in this session because TensorFlow failed CUDA initialization
# (CUDA_ERROR_NOT_INITIALIZED), and nvidia-smi reported a server-side GPU handle error for
# PCI device 0000:82:00.0.