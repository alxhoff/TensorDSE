import numpy as np

from backend.distributed_inference import distributed_inference

def get_array_size_from_shape(shape: list) -> int:
    """
    Missing  Docstring: TODO
    """
    size = 1
    for dim in shape:
        size *= int(dim)
    return size

def test():

    model_path = "resources/models/example_models/resnet/pretrainedResnet_quant.tflite"
    count = 20
    platform = "desktop"
    data = {
        "model_path": model_path,
        "count": count,
        "platform": platform,
        "cpu":{
            "mean_inf_time": 0,
            "backend_inf_time": 0
        },
        "gpu":{
            "mean_inf_time": 0,
            "backend_inf_time": 0
        },
        "tpu":{
            "mean_inf_time": 0,
            "backend_inf_time": 0
        }
    }
    input_size = get_array_size_from_shape([
                                "32",
                                "32",
                                "3"
                            ])
    output_size = get_array_size_from_shape([
                                "10"
                            ])

    input_data = np.zeros(input_size).astype(np.int8)        

    output_data = np.zeros(output_size).astype(np.int8)

    inference_times = np.zeros(count).astype(np.uint32)
    data_times = np.zeros(2).astype(np.uint32)

    for hardware_target in ["cpu", "gpu", "tpu"]:
        mean_inf_time, backend_inf_time = distributed_inference(
                                            model_path,
                                            input_data,
                                            output_data,
                                            inference_times,
                                            data_times,
                                            input_size,
                                            output_size,
                                            hardware_target,
                                            platform,
                                            count,
                                            0,
                                            'int8'
                                        )
        data[hardware_target]["mean_inf_time"] = mean_inf_time
        data[hardware_target]["backend_inf_time"] = backend_inf_time / 1000000000.0

    print(data)

if __name__ == "__main__":
    test()
