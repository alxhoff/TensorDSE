import re
import tflite

# tflite folder generated using tflite schema and the flattbuffer compiler
# See  https://google.github.io/flatbuffers/flatbuffers_guide_tutorial.html

model_filename = "MNIST_model.tflite"


def class_code_to_name(cls, code):
    for name, value in cls.__dict__.items():
        if value == code:
            return name
    return None


def process_io_lengths(op):
    return (op.InputsLength(), op.OutputsLength())


def process_io(op):
    (input, output) = process_io_lengths(op)
    inputs = []
    outputs = []
    for i in range(input):
        inputs.append(op.Inputs(i))
    for i in range(output):
        outputs.append(op.Outputs(i))

    return (inputs, outputs)


def process_io_numpy(op):
    return (op.InputsAsNumpy(), op.OutputsAsNumpy())


def process_options(op, options):
    op_type = class_code_to_name(sys.modules['tflite'].BuiltinOptions.BuiltinOptions, op.BuiltinOptionsType())
    opt = eval("sys.modules['tflite'].{}.{}()".format(op_type, op_type))
    opt.Init(options.Bytes, options.Pos)

    methods = [func for func in dir(opt) if
               callable(getattr(opt, func)) and re.search(r'^((?!Init)(?!__)(?!{}).)*$'.format(op_type), func)]

    # Store options in locals
    for method in methods:
        locals()[method] = eval("opt.{}()".format(method))
        # exec("{} = opt.{}()".format(method, method))

    return locals()


# Functions to process each operation take the form of "process_" + the builtin opcode name that can
# be found in the TFLite schema under `BuiltinOperator`. This way the functions can be resolved using `eval` and
# the resolved builtin operator name.

def process_CONV_2D(op, options, io):
    conv_options = process_options(op, options)

    print("Done")


def process_MAX_POOL_2D(op, options, io):
    pass


def process_RESHAPE(op, options, io):
    pass


def process_FULLY_CONNECTED(op, options, io):
    pass


def process_SOFTMAX(op, options, io):
    pass


def process_operation(model, graph, op):
    opcode_builtin = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
    op_name = class_code_to_name(sys.modules["tflite"].BuiltinOperator.BuiltinOperator, opcode_builtin)
    op_opts = op.BuiltinOptions()
    io_lengths = process_io_lengths(op)
    io = process_io(op)

    input_tensors = []
    for i, index in enumerate(io[0]):
        tensor = graph.Tensors(index)
        input_tensors.append(
            (tensor.ShapeAsNumpy(), class_code_to_name(sys.modules["tflite"].TensorType.TensorType, tensor.Type())))
    output_tensors = []
    for i, index in enumerate(io[1]):
        tensor = graph.Tensors(index)
        output_tensors.append(
            (tensor.ShapeAsNumpy(), class_code_to_name(sys.modules["tflite"].TensorType.TensorType, tensor.Type())))

    print("Processing {}, OP code: {}, options: {}".format(op_name, opcode_builtin,
                                                           class_code_to_name(
                                                               sys.modules["tflite"].BuiltinOptions.BuiltinOptions,
                                                               op.BuiltinOptionsType())))
    print("Input tensors: {}, output tensors: {}".format(io_lengths[0], io_lengths[1]))
    print("Input: {}, output: {}".format(input_tensors, output_tensors))

    eval("process_" + op_name)(op, op_opts, (input_tensors, output_tensors))


def main():
    with open(model_filename, "rb") as f:
        model = sys.modules["tflite"].Model.Model.GetRootAsModel(f.read(), 0)

        graph = model.Subgraphs(0)

        for i in range(graph.OperatorsLength()):
            process_operation(model, graph, graph.Operators(i))


if __name__ == '__main__':
    import os, sys

    path = os.path.join(os.path.dirname(__file__), "tflite")

    for py in [f[:-3] for f in os.listdir(path) if f.endswith('.py') and f != '__init__.py']:
        mod_name = '.'.join(["tflite", py])
        mod = __import__(mod_name, fromlist=[py])
        classes = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]
        for cls in classes:
            setattr(sys.modules[__name__], cls.__name__, cls)

    main()
