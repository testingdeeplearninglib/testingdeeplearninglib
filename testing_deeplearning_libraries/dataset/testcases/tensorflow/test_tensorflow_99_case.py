# GCFL-OTHER-0099

import os
import sys
import subprocess
import textwrap
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


def _run_child(code: str, timeout_s: int = 90):
    env = os.environ.copy()
    # This testcase intentionally exercises the CPU path in a fresh process.
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"

    try:
        p = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except subprocess.TimeoutExpired:
        return 124, "", "TIMEOUT"
    except BaseException as e:
        raise RuntimeError(f"subprocess failed: {e}") from e


CHILD_TEMPLATE = r"""
import sys
import random

def main():
    try:
        import numpy as np
    except BaseException as e:
        print("SETUP_EXC:numpy:" + e.__class__.__name__)
        return 0

    try:
        import tensorflow as tf
    except BaseException as e:
        print("SETUP_EXC:tensorflow:" + e.__class__.__name__)
        return 0

    random.seed(2021)
    np.random.seed(2021)
    try:
        tf.random.set_seed(2021)
    except BaseException:
        pass

    try:
        pool_size = {POOL_SIZE}
        layer = tf.keras.layers.MaxPooling3D(strides=1, pool_size=pool_size)
        x = tf.random.uniform([3, 4, 10, 11, 12], dtype=tf.float32)
        y = layer(x)
        {POST}
    except BaseException as e:
        print("EXC:" + e.__class__.__name__)
        return 0

    return 0

if __name__ == "__main__":
    sys.exit(main())
"""


def _child_code_zero():
    post = 'print("NO_EXC_ZERO")'
    return textwrap.dedent(
        CHILD_TEMPLATE.format(POOL_SIZE="[2, 2, 0]", POST=post)
    ).strip()


def _child_code_negative():
    post = textwrap.dedent(
        """
        try:
            nonfinite = bool(tf.reduce_any(~tf.math.is_finite(y)).numpy())
        except BaseException:
            nonfinite = False
        print("NO_EXC_NEG_NONFINITE" if nonfinite else "NO_EXC_NEG_FINITE")
        """
    ).strip()
    return textwrap.dedent(
        CHILD_TEMPLATE.format(POOL_SIZE="[2, 2, -2]", POST=post)
    ).strip()


def main():
    try:
        # Parent-side import checks only.
        try:
            import tensorflow as tf
        except BaseException as e:
            _skip(f"missing tensorflow ({e})")

        try:
            import numpy as np
            np.random.seed(2021)
        except BaseException:
            _skip("missing numpy")

        random.seed(2021)
        try:
            tf.random.set_seed(2021)
        except BaseException:
            pass

        # Case 1: pool_size contains zero.
        rc0, out0, err0 = _run_child(_child_code_zero(), timeout_s=90)
        print(f"ZERO_RC={rc0}")
        if out0:
            print("ZERO_STDOUT=" + out0.replace("\n", "\\n"))
        if err0:
            print("ZERO_STDERR=" + err0.replace("\n", "\\n"))

        if "SETUP_EXC:" in out0:
            _skip(out0)

        if rc0 != 0:
            zero_bug = True
        elif "NO_EXC_ZERO" in out0:
            zero_bug = True
        elif "EXC:" in out0:
            zero_bug = False
        else:
            raise RuntimeError(
                f"unexpected zero-case child outcome: rc={rc0}, out={out0!r}, err={err0!r}"
            )

        # Case 2: pool_size contains negative value.
        rcN, outN, errN = _run_child(_child_code_negative(), timeout_s=90)
        print(f"NEG_RC={rcN}")
        if outN:
            print("NEG_STDOUT=" + outN.replace("\n", "\\n"))
        if errN:
            print("NEG_STDERR=" + errN.replace("\n", "\\n"))

        if "SETUP_EXC:" in outN:
            _skip(outN)

        if rcN != 0:
            neg_bug = True
        elif "NO_EXC_NEG_NONFINITE" in outN:
            neg_bug = True
        elif "NO_EXC_NEG_FINITE" in outN:
            # Invalid negative pool_size was accepted at all -> still suspicious/buggy for this testcase.
            neg_bug = True
        elif "EXC:" in outN:
            neg_bug = False
        else:
            raise RuntimeError(
                f"unexpected negative-case child outcome: rc={rcN}, out={outN!r}, err={errN!r}"
            )

        if zero_bug or neg_bug:
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
# cd ~/dl_testing
# conda activate tf_venv

# export CUDA_VISIBLE_DEVICES=0
# export PYTHONUNBUFFERED=1

# python testcase/tensorflow_testcase.py 2>&1 | tee logs/GCFL-OTHER-0099.log
# echo "exit_code=$?"



# Output:
# *****************
# ZERO_RC=0
# ZERO_STDOUT=EXC:ValueError
# NEG_RC=0
# NEG_STDOUT=EXC:ValueError
# Test Failed ❌
# exit_code=0