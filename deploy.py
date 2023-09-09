import sys
import copy
import json
import argparse
import random
import numpy as np

from utils.splitter.logger import log
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


def SplitForDeployment(model_path: str, model_summary: dict):
    from utils.splitter.split import Splitter

    # Create operation models/layers from the operations in the provided model
    splitter = Splitter(model_path, model_summary)
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

    return splitter.model_layer_sequences


def DeployLayer(m: Model):
    from utils.splitter.split import SUB_DIR, COMPILED_DIR
    from utils.benchmark import GetArraySizeFromShape
    from backend.distributed_inference import distributed_inference

    delegate_type  = m.delegate[:3]
    delegate_index = int(m.delegate[-1])

    if delegate_type == "tpu":
        m.model_path = os.path.join(
                COMPILED_DIR,
                "{}_edgetpu.tflite".format(
                    m.model_name),
                )
    else:
        m.model_path = os.path.join(
                SUB_DIR,
                "tflite",
                m.model_name,
                "{0}.tflite".format(
                    m.model_name
                    ),
                )

    output_size = GetArraySizeFromShape(m.output_shape)
    output_data_vector = np.zeros(output_size).astype(m.get_np_dtype(m.output_datatype))

    inference_times_vector = np.zeros(1).astype(np.uint32)

    mean_inference_time = distributed_inference(
            m.model_path,
            m.input_vector,
            output_data_vector,
            inference_times_vector,
            len(m.input_vector),
            len(output_data_vector),
            delegate_type,
            1,
            delegate_index
            )

    m.output_vector = output_data_vector
    m.results = inference_times_vector.tolist()

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
    result["model_name"] = models[0].parent
    result["submodels"] = []
    result["total_inference_time (s)"] = 0

    for m in models:
        submodel = dict()
        submodel["name"] = m.model_name
        submodel["layers"] = m.details
        submodel["inference_time (s)"] = m.results[0] / 1000000000.0
        result["submodels"].append(submodel)
        result["total_inference_time (s)"] += m.results[0] / 1000000000.0

    with open(results_path, 'w') as json_file:
        json.dump(result, json_file, indent=4)


def DeployModel(model_path: str, model_summary_path: str, hw_summary_path: str, data_module : str = None) -> None:
    from utils.splitter.utils import ReadJSON
    from utils.splitter.logger import log

    model_name = (model_path.split("/")[-1]).split(".tflite")[0]

    if model_summary_path is not None:
        try:
            model_summary = ReadJSON(model_summary_path)
            hardware_summary = ReadJSON(hw_summary_path)
        except Exception as e:
            log.error(f"Exception occured while trying to read Model and HW Summary!")

    multi_models_sequences = SplitForDeployment(model_path=model_path, model_summary=model_summary)

    models = []

    for i, model_sequences in enumerate(multi_models_sequences):
        for j, sequence in enumerate(model_sequences):
            layers = []
            for op in sequence:
                layers.append(model_summary["models"][0]["layers"][op[0]])

            m = Model(layers, layers[0]["mapping"], model_name)
            m.model_name = "submodel_{0}_ops{1}-{2}_{3}".format(j, sequence[0][0], sequence[-1][0], m.delegate)

            if j == 0:
                data_module = None
                if data_module == None:
                    GetInputData(m)
                else:
                    GetInputTestDataModule(m, dataset_module=data_module)
            elif j < len(model_sequences):
                m.input_vector = copy.deepcopy(models[len(models) - 1].output_vector)

            m = DeployLayer(m)
            models.append(m)
    AnalyzeDeploymentResults(models=models)


parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

parser.add_argument(
        "-m",
        "--model",
        default="resources/models/example_models/MNIST_full_quanitization.tflite",
        help="File path to the SOURCE .tflite file.",
        )

parser.add_argument(
        "-s",
        "--summary",
        default="resources/model_summaries/example_summaries/MNIST/MNIST_full_quanitization_summary_with_mappings.json",
        help="File that contains a model summary with mapping annotations"
        )

parser.add_argument(
        "-d",
        "--dataset",
        default="utils.datasets.MNIST",
        help="Dataset import module to be used for providing input data"
        )

parser.add_argument(
        "-o",
        "--summaryoutputdir",
        default="resources/model_summaries/example_summaries/MNIST",
        help="Directory where model summary should be saved",
        )

parser.add_argument(
        "-n",
        "--summaryoutputname",
        default="MNIST_full_quanitization_summary_with_mappings",
        help="Name that the model summary should have",
        )

parser.add_argument(
        "-hs",
        "--hardwaresummary",
        type=str,
        default="resources/architecture_summaries/example_output_architecture_summary.json",
        help="Hardware summary file to tell benchmarking which devices to benchmark, by default all devices will be benchmarked",
        )

if __name__ == "__main__":
    import os
    from profiler import GetArgs

    args = parser.parse_args()

    if args.summary:
        DeployModel(args.model, args.summary, args.hardwaresummary, args.dataset)
    else:
        DeployModel(
                args.model,
                os.path.join(args.summaryoutputdir, "{}.json".format(args.summaryoutputname)),
                args.hardwaresummary,
                args.dataset
                )

    print("Model Deployed")
