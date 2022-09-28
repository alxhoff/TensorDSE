#!/bin/python
import sys
import argparse

parser = argparse.ArgumentParser(description="Pass in the model file to be summarized")
parser.add_argument(
    "--model",
    type=str,
    required=True,
    default="src/main/resources/examplemodels/MNIST.tflite",
    help="Path to the TFLite model that is to be loaded and summarized",
)
parser.add_argument("--outputdir", type=str, default="src/main/resources/modelsummaries",
                    help="Output directory for JSON file, defults to modelsummaries folder in java resources")
parser.add_argument("--output", type=str, required=True, default="output.json",
                    help="Filename of output JSON file")

args = parser.parse_args()


def main() -> int:
    from mltk.core import summarize_model
    import json, re
    from tensorflow.lite.python.analyzer_wrapper import _pywrap_analyzer_wrapper as _analyzer_wrapper

    analysis = _analyzer_wrapper.ModelAnalyzer(args.model, True, False).split("\n")

    summary = summarize_model(args.model, tflite=True)
    summary = [x.strip().split('|') for x in summary.split('\n')][3:-11]

    tensor_lines = []
    starting_tensor = "0"
    finishing_tensor = "0"

    for line in [x.strip() for x in analysis]:
        if line[:2] == "Op":
            tensor_lines.append(line)
        elif line[:8] == "Subgraph":
            starting_tensor = re.findall(r'Subgraph#[\d] main\(T#(\d+)\)', line)[0]
            finishing_tensor = re.findall(r'Subgraph#[\d] main\(T#\d+\) -> \[T#(\d+)\]', line)[0]

    ops_tensors = {}

    for line in tensor_lines:
        index = re.findall(r'Op#([0-9]+)', line)[0]
        input_tensors = re.findall(r'\(((?:T#\d+(?:,\s)?)*)\)', line)[0]
        input_tensors_list = re.findall(r'T#(\d+)', input_tensors)
        output_tensors = re.findall(r'\[((?:T#\d+(?:,\s)?)*)\]', line)[0]
        output_tensors_list = re.findall(r'T#(\d+)', output_tensors)
        input_tensors = [input_tensors_list] if not isinstance(input_tensors_list, list) else input_tensors_list
        output_tensors = [output_tensors_list] if not isinstance(output_tensors_list, list) else output_tensors_list
        ops_tensors[index] = {"input_tensors": input_tensors,
                              "output_tensors": output_tensors}

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
                arg = re.split(':|=', arg)
                parsed_args.append({"key": arg[0], "value": arg[1]})

        layer = {"index": layer_number, "type": layer_type, "args": parsed_args, "inputs": [], "outputs": []}

        if i != len(layer_lines) - 1:
            lines = summary[layer_lines[i]:layer_lines[i + 1]]
        else:
            lines = summary[layer_lines[i]:]

        for i, line in enumerate(lines):
            input = re.findall(r"[a-z0-9]+", line[inputs_index].replace(" ", ""))
            output = re.findall(r"[a-z0-9]+", line[outputs_index].replace(" ", ""))
            if input != []:
                input_shape = input[0].split("x")
                input_type = input[1]
                input_tensor = ops_tensors.get(layer_number).get("input_tensors")[i]
                layer.get("inputs", []).append({"shape": input_shape, "type": input_type, "tensor": input_tensor})
            if output != []:
                output_shape = output[0].split("x")
                output_type = output[1]
                output_tensor = ops_tensors.get(layer_number).get("output_tensors")[i]
                layer.get("outputs", []).append({"shape": output_shape, "type": output_type, "tensor": output_tensor})

        layers.append(layer)

    ret = {"name": args.output, "starting_tensor": starting_tensor,
           "finishing_tensor": finishing_tensor, "layers": layers}

    with open(args.outputdir + "/" + args.output, "w+") as outfile:
        json.dump(ret, outfile, indent=4)

    return 0


if __name__ == "__main__":
    sys.exit(main())
