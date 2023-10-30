import sys
import copy
import json
import argparse
import random
import numpy as np

from utils.logging.logger import log
from utils.model import Model

def GetInputData(m: Model):
    from utils.benchmark import GetArraySizeFromShape

    input_size = GetArraySizeFromShape(m.input_shape)
    m.input_vector = np.array(
            np.random.random_sample(input_size), dtype=m.get_np_dtype(m.input_datatype)
            )


def GetInputTestDataModule(m: Model, dataset_module: str):
    module = __import__(dataset_module)
    for comp in dataset_module.split(".")[1:]:
        module = getattr(module, comp)
    dataset = module.GetData()["test_data"]
    input_shape = module.GetInputShape()
    m.input_vector = random.choice(dataset)


def SplitForDeployment(model_summary: dict, platform: str, native: bool, hardware: str):
    from utils.splitter.split import Splitter
    from utils.splitter.utils import ReadJSON

    model_layer_sequences = []

    if native is True:
        for model in model_summary["models"]:
            for layer in model["layers"]:
                layer["mapping"] = hardware

    splitter = Splitter(model_summary)
    if (platform == "desktop"):
    # Create operation models/layers from the operations in the provided model
        try:
            log.info("Running Model Splitter ...")
            splitter.Run(sequences=True)
            log.info("Splitting Process Complete!\n")
        except Exception as e:
            splitter.Clean(True)
            log.error("Failed to run splitter! {}".format(str(e)))
            sys.exit(1)

        # Compiles created models/layers into Coral models for execution
        splitter.CompileForEdgeTPU(bm=False)
        log.info("Models successfully compiled!")
        model_layer_sequences = splitter.model_layer_sequences
    else:
        model_layer_sequences = ReadJSON("utils/splitter/model_layer_sequences.json")
        

    return model_layer_sequences, model_summary


def DeployLayer(m: Model, platform: str, usbmon: int=None):
    from utils.splitter.split import MODELS_DIR
    from utils.benchmark import GetArraySizeFromShape, StandardDeploy, TPUDeploy
    from backend.distributed_inference import distributed_inference

    delegate_type  = m.delegate[:3]
    delegate_index = int(m.delegate[-1])

    if delegate_type == "tpu":
        m.model_path = os.path.join(
                MODELS_DIR,
                m.parent,
                "sub",
                "compiled",
                "{}_edgetpu.tflite".format(
                    m.model_name),
                )
    else:
        m.model_path = os.path.join(
                MODELS_DIR,
                m.parent,
                "sub",
                "tflite",
                "{0}.tflite".format(
                    m.model_name
                    ),
                )

    if ((platform == "desktop") or (platform == "rpi")):
        if os.path.isfile(m.model_path):
            if (delegate_type == "tpu"):
                if usbmon == None:
                    raise Exception("usbmon interface not provided to Deploy Layer for TPU device")
                else:
                    m = TPUDeploy(m=m, count=1, usbmon=usbmon, platform=platform)
            else:
                m = StandardDeploy(m=m, count=1, hardware_target=delegate_type, platform=platform)

        else:
            m.results = [-1]
            m.timers = {
                    "error": {
                        "name": "file_not_found",
                        "reason": "Cannot find model {}".format(m.model_path),
                        },
                    }
    else:
        m = StandardDeploy(m=m, count=1, hardware_target=delegate_type, platform=platform)

    return m


def AnalyzeDeploymentResults(models: list) -> None:

    RESULTS_FOLDER = os.path.join(os.getcwd(), "resources/deployment_results")
    results_path = os.path.join(RESULTS_FOLDER, f"{models[0].parent}.json")

    if not os.path.isdir(RESULTS_FOLDER):
        import sys
        log.error(f"{RESULTS_FOLDER} is not a valid folder to store results!")
        os.mkdir(RESULTS_FOLDER)
        if not os.path.isdir(RESULTS_FOLDER):
            sys.exit(-1)

    result = dict()
    result["models"] = []

    for i, model in enumerate(models):
        submodels_result = dict()
        submodels_result["model_name"] = model[i].parent
        submodels_result["submodels"] = []
        submodels_result["total_inference_time (s)"] = 0
        for m in model:
            submodel = dict()
            submodel["name"] = m.model_name
            submodel["layers"] = m.details
            submodel["inference_time (s)"] = m.results[0] / 1000000000.0
            submodels_result["submodels"].append(submodel)
            submodels_result["total_inference_time (s)"] += m.results[0] / 1000000000.0
        result["models"].append(submodels_result)

    with open(results_path, 'w') as json_file:
        json.dump(result, json_file, indent=4)


def DeployModels(model_summary: dict, platform: str, hardware: str, native: bool = False, data_module: str = None) -> None:

    models = []
    multi_models_sequences = []
    
    multi_models_sequences, model_summary = SplitForDeployment(model_summary=model_summary, platform=platform, native=native, hardware=hardware)
        
    for i, model_sequences in enumerate(multi_models_sequences):
        submodels = []
        model_name = os.path.basename(model_summary["models"][i]["path"]).split(".tflite")[0]
        for j, sequence in enumerate(model_sequences):
            layers = []
            for op in sequence:
                layers.append(model_summary["models"][i]["layers"][op[0]])

            m = Model(layers, layers[0]["mapping"], model_name)

            if len(sequence) == 1:
                ops_range = sequence[0][0]
            else:
                ops_range = '-'.join(map(str, [sequence[0][0], sequence[-1][0]]))

            if native is False:
                m.model_name = "submodel_{0}_{1}_{2}".format(j, f"ops{ops_range}", m.delegate)
            else:
                m.model_name = model_name

            if j == 0:
                #Foced random input for rpi and dev board due to unsupported dataset installation
                if not (platform == "desktop"):
                    data_module = None
                else:
                    if data_module == None:
                        GetInputData(m)
                    else:
                        GetInputTestDataModule(m, dataset_module=data_module)
            else:
                m.input_vector = copy.deepcopy(submodels[-1].output_vector)

            m = DeployLayer(m, platform)
            submodels.append(m)
        models.append(submodels)
    
    AnalyzeDeploymentResults(models=models)


def getArgs():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )

    parser.add_argument(
            "-s",
            "--summarypath",
            default="resources/model_summaries/example_summaries/MNIST/MNIST_full_quanitization_summary_w_mappings.json",
            help="File that contains a model summary with mapping annotations"
            )

    parser.add_argument(
            "-d",
            "--dataset",
            default="utils.datasets.MNIST",
            help="Dataset import module to be used for providing input data"
            )

    parser.add_argument(
            "-p",
            "--platform",
            default="desktop",
            help="Platform supporting the profiling/deployment process",
            )
    
    parser.add_argument(
            "-n",
            "--native",
            default=False,
            help="Native deployment on one specific HW Target",
            )
    
    parser.add_argument(
            "-h",
            "--hardware",
            default="cpu",
            help="HW Target for native deployment",
            )
    
    return parser.parse_args()


if __name__ == "__main__":
    import os

    from utils.splitter.utils import ReadJSON

    args = getArgs()

    if args.summarypath is not None:
        model_summary_json = ReadJSON(args.summarypath)
        if model_summary_json is None:
            log.error("The provided Model Summary is empty!")
            sys.exit(-1)
    else:
        log.error("The provided Model Summary is empty!")
        sys.exit(-1)
    
    log.info("[PROFILER] Starting")

    try:
        DeployModels(
            model_summary=model_summary_json, 
            platform=args.platform,
            hardware=args.hardware, 
            native=args.native, 
            data_module=args.dataset
            )
        
    except Exception as e:
        log.error(f"An error occured in the deployment process. ({e})")

    log.info("[DEPLOY] Finished")
