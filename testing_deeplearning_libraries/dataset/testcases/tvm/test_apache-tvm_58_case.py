# GCFL-OTHER-0058

import sys
import random
import traceback


def _skip(reason: str) -> None:
    print(f"SKIP_ENV: {reason}")
    sys.exit(0)


def _pass() -> None:
    print("Test Passed ✅")
    sys.exit(0)


def _fail(reason: str = "") -> None:
    if reason:
        print(f"DEBUG_FAIL_REASON: {reason}")
    print("Test Failed ❌")
    sys.exit(0)


def _harness_error(exc: BaseException) -> None:
    print(f"HARNESS_ERROR: {type(exc).__name__}: {exc}")
    sys.exit(1)


def main() -> None:
    seed = 2021
    random.seed(seed)

    try:
        import numpy as np
    except Exception as e:
        _skip(f"numpy not available: {e}")

    try:
        np.random.seed(seed)
    except Exception:
        pass

    try:
        import mxnet as mx
    except Exception as e:
        _skip(f"mxnet not available: {e}")

    try:
        mx.random.seed(seed)
    except Exception:
        pass

    print(f"ENV: mxnet={mx.__version__}, numpy={np.__version__}, python={sys.version.split()[0]}")

    quantize_net_v2 = None
    try:
        from mxnet.contrib.quantization import quantize_net_v2 as _qnv2  # type: ignore
        quantize_net_v2 = _qnv2
    except Exception:
        try:
            from mxnet.contrib import quantization as _q  # type: ignore
            quantize_net_v2 = getattr(_q, "quantize_net_v2", None)
        except Exception as e:
            _skip(f"mxnet.contrib.quantization import failed: {e}")

    if quantize_net_v2 is None:
        _skip("mxnet.contrib.quantization.quantize_net_v2 not available in this MXNet build/version")

    try:
        from mxnet.gluon import nn  # type: ignore

        net = nn.HybridSequential()
        with net.name_scope():
            net.add(nn.Dense(4, flatten=True))

        net.initialize(ctx=mx.cpu())
        net.hybridize()
    except Exception as e:
        _harness_error(f"failed to construct/init HybridBlock: {e}")

    try:
        quantize_net_v2(
            network=net,
            calib_data=None,
            data_shapes=None,
            calib_mode="none",
        )
    except UnboundLocalError as e:
        print(f"DEBUG_EXCEPTION: {type(e).__name__}: {e}")
        if "dshapes" in str(e):
            _pass()
        _fail("UnboundLocalError occurred, but did not mention dshapes")
    except Exception as e:
        print(f"DEBUG_EXCEPTION: {type(e).__name__}: {e}")
        print("DEBUG_TRACEBACK_START")
        traceback.print_exc()
        print("DEBUG_TRACEBACK_END")
        _fail(f"Different exception type encountered: {type(e).__name__}")
    else:
        _fail("No exception raised by quantize_net_v2")


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

# MXNet


# Commands
# *****************
# set -o pipefail
# export CUDA_VISIBLE_DEVICES=""
# python ~/dl_testing/testcases/GCFL-OTHER-0058.py 2>&1 | tee ~/dl_testing/GCFL-OTHER-0058-debug.log
# echo "exit_code=$?"


# Output:
# *****************
# Traceback (most recent call last):
#   File "/home/talha/dl_testing/testcases/GCFL-OTHER-0058.py", line 83, in main
#     quantize_net_v2(
#   File "/home/talha/miniconda3/envs/tf_venv/lib/python3.10/site-packages/mxnet/contrib/quantization.py", line 899, in quantize_net_v2
#     raise ValueError('data_shapes required')
# ValueError: data_shapes required
# ENV: mxnet=1.9.1, numpy=1.23.5, python=3.10.19
# DEBUG_EXCEPTION: ValueError: data_shapes required
# DEBUG_TRACEBACK_START
# DEBUG_TRACEBACK_END
# DEBUG_FAIL_REASON: Different exception type encountered: ValueError
# Test Failed ❌
# exit_code=0

# Triggering command:
# set -o pipefail
# export CUDA_VISIBLE_DEVICES=""
# python ~/dl_testing/testcases/GCFL-OTHER-0058.py 2>&1 | tee ~/dl_testing/GCFL-OTHER-0058-debug.log
# echo "exit_code=$?"

# Output:
# ENV: mxnet=1.9.1, numpy=1.23.5, python=3.10.19
# DEBUG_EXCEPTION: ValueError: data_shapes required
# DEBUG_FAIL_REASON: Different exception type encountered: ValueError
# Test Failed ❌
# exit_code=0