import sys
sys.path.append('/home/tensorDSE')

from utils.model import Model
from deploy import TPUDeploy
from analysis import AnalyzeModelResults

model="models/compiled/quant_CONV_2D_edgetpu.tflite"
count=100

results = {}
results["cpu"]   = []
results["gpu"]   = []
results["tpu"]   = []
results["count"] = count

results["tpu"].append(TPUDeploy(Model(model, "tpu"), count))
AnalyzeModelResults("TEST", results)
