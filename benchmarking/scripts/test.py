import sys
sys.path[0]="/home/tensorDSE" # need to overwrite working directory, so imports can work

from utils.model import Model
from deploy import TPUDeploy
from analysis import AnalyzeModelResults

from main import log
from utils.log import Log

log = Log("results/debug.log")

model="models/compiled/quant_CONV_2D_edgetpu.tflite"
count=1

results = {}
results["cpu"]   = []
results["gpu"]   = []
results["tpu"]   = []
results["count"] = count

results["tpu"].append(TPUDeploy(Model(model, "tpu"), count))
# AnalyzeModelResults("TEST", results)
