# import sys
# sys.path.insert(0,'utils')

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

m = TPUDeploy(Model(model, "tpu"), count)
AnalyzeModelResults("TEST", results)
