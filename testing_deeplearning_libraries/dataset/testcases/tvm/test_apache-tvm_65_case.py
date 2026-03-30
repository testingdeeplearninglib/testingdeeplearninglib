# GCFL-AUTOGRADBA-0065

import sys
import time
import traceback

def _skip(reason: str):
    print(f"SKIP_ENV: {reason}")
    sys.exit(0)

def _fail():
    print("Test Failed ❌")
    sys.exit(0)

def _pass():
    print("Test Passed ✅")
    sys.exit(0)

def main():
    try:
        try:
            import numpy as np
        except Exception as e:
            _skip(f"numpy not available: {e}")

        try:
            import mxnet as mx
            from mxnet import autograd, gluon
        except Exception as e:
            _skip(f"mxnet not available: {e}")

        # --- GPU checks ---
        try:
            num_gpus = mx.context.num_gpus()
        except Exception as e:
            _skip(f"unable to query GPU count: {e}")

        if not num_gpus or num_gpus < 1:
            _skip("no GPU detected by MXNet (mx.context.num_gpus() == 0)")

        try:
            ctx = mx.gpu()
            _ = (mx.nd.ones((1,), ctx=ctx) + 1).asnumpy()
        except Exception as e:
            _skip(f"mx.gpu() not usable: {e}")

        # --- Seeds ---
        np.random.seed(1)
        try:
            mx.random.seed(1)
        except Exception:
            pass

        # --- Spec parameters ---
        N = 50000
        embed_dim = 300
        idx_shape = (1024, 160)
        max_iters = 2000
        max_runtime_sec = 120

        # --- Build embedding ---
        try:
            embedding = gluon.nn.Embedding(N, embed_dim)
            embedding.initialize(ctx=ctx)
        except Exception as e:
            _skip(f"failed to initialize Embedding on GPU: {e}")

        # --- Fixed index tensor ---
        try:
            idx_np = np.random.randint(0, N, size=idx_shape, dtype=np.int64)
            idx = mx.nd.array(idx_np, ctx=ctx, dtype="int64")
        except Exception as e:
            _skip(f"failed to create idx on GPU: {e}")

        start = time.time()
        saw_nonfinite = False
        later_all_finite_after_nonfinite = False

        for _i in range(1, max_iters + 1):
            if (time.time() - start) > max_runtime_sec:
                break

            try:
                embedding.collect_params().zero_grad()

                with autograd.record():
                    emb_in = embedding(idx)
                    loss = emb_in.sum()
                loss.backward()
            except Exception as e:
                _skip(f"exception during forward/backward: {e}")

            try:
                g = embedding.weight.grad(ctx)
                nan_mask = mx.nd.contrib.isnan(g)
                inf_mask = mx.nd.contrib.isinf(g)
                nonfinite_count = mx.nd.sum(nan_mask + inf_mask).asscalar()
            except Exception as e:
                _skip(f"unable to check grad finiteness on GPU: {e}")

            is_finite = (nonfinite_count == 0)

            if not is_finite:
                saw_nonfinite = True
            else:
                if saw_nonfinite:
                    later_all_finite_after_nonfinite = True
                    break

        if saw_nonfinite and later_all_finite_after_nonfinite:
            _pass()
        else:
            _fail()

    except SystemExit:
        raise
    except Exception as e:
        print(f"HARNESS_ERROR: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()



# ******************************************************************************
#                                    Result 
# ******************************************************************************

# mxnet

# Output:

# (mxnet_gpu) talha@bitse-SYS-7048GR-TR:~/dl_testing/testcases$ conda activate mxnet_gpu
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
# export MXNET_CUDNN_LIB_CHECKING=0
# python mxnet_embedding_nan_test.py
# echo $?
# Test Failed ❌
# 0
# (mxnet_gpu) talha@bitse-SYS-7048GR-TR:~/dl_testing/testcases$ cat > run_mxnet.sh <<'SH'
# #!/usr/bin/env bash
# set -euo pipefail
# ENV="$HOME/miniconda3/envs/mxnet_gpu"
# export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
# export MXNET_CUDNN_LIB_CHECKING=0
# "$ENV/bin/python" mxnet_embedding_nan_test.py
# SH
# chmod +x run_mxnet.sh
# ./run_mxnet.sh
# bash: /home/talha/miniconda3/envs/mxnet_gpu/lib/libtinfo.so.6: no version information available (required by bash)
# Test Failed ❌