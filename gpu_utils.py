import os
import sys

def set_gpu():
    try:
        gpu_argument_index = sys.argv.index("--gpu")
        gpu = sys.argv[gpu_argument_index + 1]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        sys.argv.pop(gpu_argument_index)
        sys.argv.pop(gpu_argument_index)
    except ValueError:
        print("No GPU specified. Using all available GPUs.")