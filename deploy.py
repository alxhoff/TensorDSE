import copy
import numpy as np
from utils.model import Model
import argparse
import random

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


def DeployLayer(m: Model):
    from utils.splitter.split import SUB_DIR, COMPILED_DIR
    from utils.benchmark import GetArraySizeFromShape
    from backend.distributed_inference import distributed_inference

    if m.delegate == "tpu":
        m.model_path = os.path.join(
            COMPILED_DIR,
            "submodel_{0}_{1}_{2}_edgetpu.tflite".format(
                m.details["index"], m.details["type"], m.delegate
            ),
        )
    else:
        m.model_path = os.path.join(
            SUB_DIR,
            "tflite",
            "submodel_{0}_{1}_{2}".format(
                m.details["index"], m.details["type"], m.delegate
            ),
            "submodel_{0}_{1}_{2}.tflite".format(
                m.details["index"], m.details["type"], m.delegate
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
        m.delegate,
        1,
    )

    m.output_vector = output_data_vector
    m.results = inference_times_vector.tolist()

    return m


def DeployModel(model_path: str, model_summary_path: str, data_module : str = None) -> None:
    from utils.splitter.utils import ReadJSON
    from utils.splitter.logger import log

    model_name = (model_path.split("/")[-1]).split(".tflite")[0]

    if model_summary_path is not None:
        try:
            model_summary = ReadJSON(model_summary_path)
        except Exception as e:
            log.error(f"Exception occured while trying to read Model Summary!")

    models = []

    for i, model in enumerate(model_summary["models"]):
        for idx, layer in enumerate(model["layers"]):
            m = Model(layer, layer["mapping"].upper(), model_name)

            if idx == 0:
                if data_module == None:
                    GetInputData(m)
                else:
                    GetInputTestDataModule(m, dataset_module=data_module)
            elif idx < len(model_summary["layers"]):
                m.input_vector = copy.deepcopy(models[len(models) - 1].output_vector)

            m = DeployLayer(m)
            models.append(m)

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


if __name__ == "__main__":
    import os
    from profiler import GetArgs

    args = parser.parse_args()

    if args.summary:
        print("Deploying: {}".format(args.summary))
        DeployModel(args.model, args.summary, args.dataset)
    else:
        print("Deploying: {}".format(args.summaryoutputdir + "/" + args.summaryoutputname))
        DeployModel(
            args.model,
            os.path.join(args.summaryoutputdir, "{}.json".format(args.summaryoutputname)),
            args.dataset
        )

    print("Model Deployed")
