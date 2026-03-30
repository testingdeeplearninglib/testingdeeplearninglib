# GCFL-AUTOGRADBA-0095
# Graph-mode Jacobian failure for tf.cond; eager mode correct

import sys
import tensorflow as tf

def _passed():
    print("Test Passed ✅")
    sys.exit(0)

def _failed():
    print("Test Failed ❌")
    sys.exit(0)

def main():
    print(f"TF_VERSION: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUS: {gpus}")
    device = "/GPU:0" if gpus else "/CPU:0"
    print(f"DEVICE: {device}")

    @tf.function
    def f_bad(x):
        if x < 1.0:
            return x * x
        else:
            return x * x

    @tf.function
    def f_bad_cond(x):
        return tf.cond(
            x < 1.0,
            lambda: x * x,
            lambda: x * x,
        )

    x = tf.constant(0.5, dtype=tf.float32)

    def run(func, eager: bool):
        tf.config.run_functions_eagerly(eager)
        try:
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(x)
                with tf.GradientTape() as t1:
                    t1.watch(x)
                    y = func(x)
                g = t1.gradient(y, x)
            h = t2.jacobian(g, x, experimental_use_pfor=True)
            return ("ok", float(h.numpy()))
        except Exception as e:
            return ("err", str(e))

    r_fn_1 = run(f_bad, eager=False)
    r_ea_1 = run(f_bad, eager=True)
    r_fn_2 = run(f_bad_cond, eager=False)
    r_ea_2 = run(f_bad_cond, eager=True)

    print("MODE=function NAME=f_bad     RESULT=", r_fn_1)
    print("MODE=eager    NAME=f_bad     RESULT=", r_ea_1)
    print("MODE=function NAME=f_bad_cond RESULT=", r_fn_2)
    print("MODE=eager    NAME=f_bad_cond RESULT=", r_ea_2)

    if r_fn_1[0] == "err" and r_ea_1 == ("ok", 2.0):
        _passed()
    if r_fn_2[0] == "err" and r_ea_2 == ("ok", 2.0):
        _passed()

    _failed()

if __name__ == "__main__":
    main()



# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Tensorflow


# Commands
# *****************
# source ~/.venvs/dl_testing/bin/activate
# cd ~/dl_testing
# CUDA_VISIBLE_DEVICES=0 python testcases/tf_cases/tensorflow_testcase.py \
#   2>&1 | tee logs/tf_autogradba_0095_run.log



# Output:
# *****************
# TF_VERSION: 2.20.0
# GPUS: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
# DEVICE: /GPU:0

# MODE=function NAME=f_bad RESULT= (
#   'err',
#   'Encountered an exception while vectorizing the jacobian computation'
# )

# MODE=eager    NAME=f_bad RESULT= ('ok', 2.0)

# MODE=function NAME=f_bad_cond RESULT= (
#   'err',
#   'Encountered an exception while vectorizing the jacobian computation'
# )

# MODE=eager    NAME=f_bad_cond RESULT= ('ok', 2.0)

# Test Passed ✅


# ******************************************************************************

# Reported ✅
# Link: 
# https://github.com/tensorflow/tensorflow/issues/108936