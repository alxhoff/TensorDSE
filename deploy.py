import numpy as np
from utils.model import Model

def GetInputData():
    pass


def DeployLayer(m: Model):
    
    from utils.splitter.split import SUB_DIR, COMPILED_DIR
    from utils.benchmark import GetArraySizeFromShape
    from backend.distributed_inference import distributed_inference

    if (m.delegate == "tpu"):
        m.model_path = os.path.join(COMPILED_DIR, "submodel_{0}_{1}_bm_edgetpu.tflite".format(m.details["index"], m.details["type"]))
    else:
        m.model_path = os.path.join(SUB_DIR, "tflite", "submodel_{0}_{1}_bm".format(m.details["index"], m.details["type"]),"submodel_{0}_{1}_bm.tflite".format(m.details["index"], m.details["type"]))

    
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
        1
    )

    m.output_vector = output_data_vector
    m.results = inference_times_vector.tolist()


def DeployModel(model_path: str, model_summary_path: str) -> None:

    from utils.benchmark import GetArraySizeFromShape
    from utils.splitter.utils import ReadJSON
    from utils.splitter.logger import log
    
    model_name = (model_path.split("/")[-1]).split(".tflite")[0]

    if model_summary_path is not None:
        try:
            model_summary = ReadJSON(model_summary_path)
        except Exception as e:
            log.error(f"Exception occured while trying to read Model Summary!")

    models = []

    for idx, layer in enumerate(model_summary["layers"]):
        m = Model(layer, layer["mapping"], model_name)

        input_size  = GetArraySizeFromShape(m.input_shape)
        m.input_vector = np.zeros(input_size).astype(m.get_np_dtype(m.input_datatype))

        if (idx == 0):
            print("Read Input Data")
        elif (idx < len(model_summary["layers"])):
            print("Pass Output Vector")
        else:
            print("Return Output Data Vector")

        m = DeployLayer(m)
        models.append(m)


if __name__ == "__main__":
    import os
    from profiler import GetArgs

    args = GetArgs()

    DeployModel(args.model, args.count, args.hardwaresummary, os.path.join(args.summaryoutputdir, "{}.json".format(args.summaryoutputname)))

    print("Model Deployed")
