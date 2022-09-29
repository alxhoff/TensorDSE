import sys
sys.path[0]="/home/tensorDSE" # need to overwrite working directory, so imports can work

import argparse
from utils.model import Model
from deploy import TPUDeploy
from analysis import AnalyzeModelResults

model="models/compiled/quant_CONV_2D_edgetpu.tflite"
count=1

results = {}
results["cpu"]   = []
results["gpu"]   = []
results["tpu"]   = []
results["count"] = count

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-c", "--count", type=int, default=1, help="Number of times to run inference."
    )

    parser.add_argument(
        "-t", "--timeout", type=int, default=30, help="Number of seconds to wait timeout."
    )

    args = parser.parse_args()
    return args


args = get_args()
results["tpu"].append(TPUDeploy(Model(model, "tpu"), args.count, args.timeout))
# AnalyzeModelResults("TEST", results)
