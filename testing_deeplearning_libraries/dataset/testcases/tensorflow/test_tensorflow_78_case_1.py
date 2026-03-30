# GCFL-SPARSE-0078

import os
import sys
import tempfile
import subprocess
import textwrap


def _skip(reason: str) -> None:
    print(f"SKIP_ENV: {reason}")
    sys.exit(0)


def _pass() -> None:
    print("Test Passed ✅")
    sys.exit(0)


def _fail() -> None:
    print("Test Failed ❌")
    sys.exit(0)


def _harness_error(msg: str) -> None:
    print(f"HARNESS_ERROR: {msg}")
    sys.exit(1)


CHILD_CODE = r'''
import sys

def main():
    try:
        try:
            import tensorflow as tf
        except Exception as e:
            print(f"CHILD_SKIP:tensorflow import failed: {type(e).__name__}: {e}")
            return 0

        tf_debug = None
        try:
            from tensorflow.python import debug as tf_debug_mod  # type: ignore
            if hasattr(tf_debug_mod, "LocalCLIDebugWrapperSession"):
                tf_debug = tf_debug_mod
        except Exception:
            tf_debug = None

        if tf_debug is None:
            try:
                from tensorflow.python.debug.wrappers import local_cli_wrapper  # type: ignore

                class _TFDebugCompat:
                    LocalCLIDebugWrapperSession = local_cli_wrapper.LocalCLIDebugWrapperSession

                tf_debug = _TFDebugCompat()
            except Exception:
                tf_debug = None

        if tf_debug is None:
            print("CHILD_SKIP:tfdbg LocalCLIDebugWrapperSession not available")
            return 0

        try:
            tf.compat.v1.disable_eager_execution()
        except Exception:
            pass

        g = tf.Graph()
        with g.as_default():
            if not hasattr(tf.compat.v1, "sparse_placeholder"):
                print("CHILD_SKIP:tf.compat.v1.sparse_placeholder not available")
                return 0

            a = tf.compat.v1.sparse_placeholder(tf.float32, shape=(None, 5, 5), name="tensor1")
            b = tf.compat.v1.sparse_placeholder(tf.float32, shape=(None, 5, 5), name="tensor2")
            add = tf.sparse.add(a, b)

        base_sess = tf.compat.v1.Session(graph=g)
        wrapped_sess = None
        try:
            wrapped_sess = tf_debug.LocalCLIDebugWrapperSession(base_sess)

            indices = [[0, 0, 1], [0, 0, 2]]
            values = [1.0, 2.0]
            dense_shape = [1, 5, 5]

            a_val = tf.compat.v1.SparseTensorValue(indices=indices, values=values, dense_shape=dense_shape)
            b_val = tf.compat.v1.SparseTensorValue(indices=indices, values=values, dense_shape=dense_shape)

            wrapped_sess.run(add, feed_dict={a: a_val, b: b_val})
            print("CHILD_OK")
            return 0

        finally:
            for sess_obj in (wrapped_sess, base_sess):
                if sess_obj is None:
                    continue
                try:
                    sess_obj.close()
                except Exception:
                    pass

    except Exception as e:
        print(f"CHILD_EXCEPTION:{type(e).__name__}:{e}")
        return 0

if __name__ == "__main__":
    raise SystemExit(main())
'''


def main() -> None:
    try:
        with tempfile.NamedTemporaryFile("w", suffix="_gcfl_sparse_0078_child.py", delete=False) as f:
            child_path = f.name
            f.write(textwrap.dedent(CHILD_CODE))

        try:
            env = os.environ.copy()
            proc = subprocess.run(
                [sys.executable, child_path],
                input="run -n\n",
                text=True,
                capture_output=True,
                timeout=30,
                env=env,
            )
        except subprocess.TimeoutExpired:
            _skip("child timed out (possible interactive tfdbg / tty requirement)")
        finally:
            try:
                os.remove(child_path)
            except Exception:
                pass

        combined = (proc.stdout or "") + (proc.stderr or "")
        sys.stdout.write(combined)
        if combined and not combined.endswith("\n"):
            sys.stdout.write("\n")

        if "CHILD_SKIP:" in combined:
            reason = combined.split("CHILD_SKIP:", 1)[1].splitlines()[0].strip()
            _skip(reason)

        if "CHILD_OK" in combined:
            _fail()

        if "CHILD_EXCEPTION:" in combined:
            payload = combined.split("CHILD_EXCEPTION:", 1)[1].splitlines()[0]
            parts = payload.split(":", 1)
            exc_type = parts[0].strip()
            exc_msg = parts[1].strip() if len(parts) > 1 else ""

            if exc_type == "AttributeError" and ("SparseTensor" in exc_msg) and ("name" in exc_msg):
                _pass()

            terminal_markers = ("tty", "stdin", "fileno", "terminal", "termios", "EOF", "reading a line")
            if exc_type in {"EOFError", "UnsupportedOperation"} or any(t.lower() in exc_msg.lower() for t in terminal_markers):
                _skip(f"interactive tfdbg/tty issue: {exc_type}: {exc_msg}")

            _fail()

        _harness_error(f"unexpected child outcome (returncode={proc.returncode})")

    except SystemExit:
        raise
    except Exception as e:
        _harness_error(f"{type(e).__name__}: {e}")


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
# mkdir -p logs

# export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES=""

# python testcases/tensorflow_testcase.py 2>&1 | tee logs/GCFL-SPARSE-0078_subprocess.log
# echo "exit_code=$?"


# Output:
# *****************
# run-start: run 
# TensorFlow version: 2.21.0

# Session.run() call #1:
# Fetch(es):
#   SparseTensor(indices=Tensor("SparseAdd:0", ...), values=Tensor("SparseAdd:1", ...), dense_shape=Tensor("SparseAdd:2", ...))

# Feed dict:
#   SparseTensor(indices=Tensor("tensor1/indices:0", ...), values=Tensor("tensor1/values:0", ...), dense_shape=Tensor("tensor1/shape:0", ...))
#   SparseTensor(indices=Tensor("tensor2/indices:0", ...), values=Tensor("tensor2/values:0", ...), dense_shape=Tensor("tensor2/shape:0", ...))

# tfdbg> CHILD_OK
# Test Failed ❌
# exit_code=0