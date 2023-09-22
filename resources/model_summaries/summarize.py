def SummarizeGraph(model: str, output_dir: str, output_name: str) -> None:
    from mltk.core import summarize_model
    import json, re
    from tensorflow.lite.python.analyzer_wrapper import (
        _pywrap_analyzer_wrapper as _analyzer_wrapper,
    )

    analysis = _analyzer_wrapper.ModelAnalyzer(model, True, False).split("\n")

    summary = summarize_model(model, tflite=True)
    summary = [x.strip().split("|") for x in summary.split("\n")][3:-7]

    tensor_lines = []
    starting_tensor = "0"
    finishing_tensor = "0"

    for line in [x.strip() for x in analysis]:
        if line[:2] == "Op":
            tensor_lines.append(line)
        elif line[:8] == "Subgraph":
            starting_tensor = re.findall(r"Subgraph#[\d] main\(T#(\d+)\)", line)[0]
            finishing_tensor = re.findall(
                r"Subgraph#[\d] main\(T#\d+\) -> \[T#(\d+)\]", line
            )[0]

    ops_tensors = {}

    for line in tensor_lines:
        index = re.findall(r"Op#([0-9]+)", line)[0]
        input_tensors = re.findall(r"\((.+)\)\s->", line)[0]
        input_tensors_list = re.findall(r"T#(\d+)", input_tensors)
        output_tensors = re.findall(r"\[((?:T#\d+(?:,\s)?)*)\]", line)[0]
        output_tensors_list = re.findall(r"T#(\d+)", output_tensors)
        input_tensors = (
            [input_tensors_list]
            if not isinstance(input_tensors_list, list)
            else input_tensors_list
        )
        output_tensors = (
            [output_tensors_list]
            if not isinstance(output_tensors_list, list)
            else output_tensors_list
        )
        ops_tensors[index] = {
            "input_tensors": input_tensors,
            "output_tensors": output_tensors,
        }

    layer_lines = [i for i, x in enumerate(summary) if x[1].strip() != ""]

    layer_number_index = 1
    layer_type_index = 2
    inputs_index = 3
    outputs_index = 4
    arguments_index = 5

    layers = []

    for i in range(len(layer_lines)):
        layer_number = summary[layer_lines[i]][layer_number_index].replace(" ", "")
        layer_type = summary[layer_lines[i]][layer_type_index].replace(" ", "")
        layer_args = summary[layer_lines[i]][arguments_index].strip()

        parsed_args = []

        if layer_args != "":
            for arg in layer_args.split(" "):
                arg = re.split(":|=", arg)
                parsed_args.append({"key": arg[0], "value": arg[1]})

        layer = {
            "index": layer_number,
            "type": layer_type,
            "args": parsed_args,
            "mapping": "",
            "inputs": [],
            "outputs": [],
        }

        if i != len(layer_lines) - 1:
            lines = summary[layer_lines[i] : layer_lines[i + 1]]
        else:
            lines = summary[layer_lines[i] :]

        for i, line in enumerate(lines):
            input = re.findall(r"[a-z0-9]+", line[inputs_index].replace(" ", ""))
            output = re.findall(r"[a-z0-9]+", line[outputs_index].replace(" ", ""))
            if input != []:
                input_shape = input[0].split("x")
                input_type = input[1]
                input_tensor = ops_tensors.get(layer_number).get("input_tensors")[i]
                layer.get("inputs", []).append(
                    {"shape": input_shape, "type": input_type, "tensor": input_tensor}
                )
            if output != []:
                try:
                    output_shape = output[0].split("x")
                    output_type = output[1]
                    output_tensor = ops_tensors.get(layer_number).get("output_tensors")[
                        i
                    ]
                    layer.get("outputs", []).append(
                        {
                            "shape": output_shape,
                            "type": output_type,
                            "tensor": output_tensor,
                        }
                    )
                except Exception as e:
                    print(e)

        layers.append(layer)

    ret = {
        "models": [
            {
                "name": output_name,
                "deadline" : 0.0,
                "starting_tensor": starting_tensor,
                "finishing_tensor": finishing_tensor,
                "layers": layers,
            }
        ]
    }

    try:
        with open(output_dir + "/" + output_name + ".json", "w+") as outfile:
            json.dump(ret, outfile, indent=4)
    except Exception as e:
        print(e)
        return 1

    return 0
