from tflite.ActivationFunctionType import ActivationFunctionType
from tflite.BuiltinOperator import BuiltinOperator
from tflite.BuiltinOptions import BuiltinOptions
from tflite.FullyConnectedOptionsWeightsFormat import FullyConnectedOptionsWeightsFormat
from tflite.Model import Model
from tflite.Padding import Padding
from tflite.TensorType import TensorType

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


# Functions to process each operation take the form of "process_" + the builtin opcode name that can
# be found in the TFLite schema under `BuiltinOperator`. This way the functions can be resolved using `eval` and
# the resolved builtin operator name.

def process_CONV_2D(op, options, io):
    from tflite import Conv2DOptions as c2opt

    assert (op.BuiltinOptionsType() == BuiltinOptions.Conv2DOptions)

    opt = c2opt.Conv2DOptions()
    opt.Init(options.Bytes, options.Pos)

    padding = class_code_to_name(Padding, opt.Padding())
    stride_w = opt.StrideW()
    stride_h = opt.StrideH()
    dilation_w = opt.DilationWFactor()
    dilation_h = opt.DilationHFactor()
    activation_func = class_code_to_name(ActivationFunctionType, opt.FusedActivationFunction())

    print("Pad: {}, Stride: [{},{}], Dilation: [{},{}], Activ: {}".format(padding, stride_w, stride_h, dilation_w,
                                                                          dilation_h, activation_func))


    ## We know options so we could go and create out single layer network for benchmarking here and save it

def process_MAX_POOL_2D(op, options, io):
    from tflite import Pool2DOptions as p2opt

    assert (op.BuiltinOptionsType() == BuiltinOptions.Pool2DOptions)

    opt = p2opt.Pool2DOptions()
    opt.Init(options.Bytes, options.Pos)

    padding = class_code_to_name(Padding, opt.Padding())
    stride_w = opt.StrideW()
    stride_h = opt.StrideH()
    filter_w = opt.FilterWidth()
    filter_h = opt.FilterHeight()
    activation_func = class_code_to_name(ActivationFunctionType, opt.FusedActivationFunction())

    print("Pad: {}, Stride: [{},{}], Filter: [{},{}], Activ: {}".format(padding, stride_w, stride_h, filter_w,
                                                                        filter_h, activation_func))

def process_RESHAPE(op, options, io):
    from tflite import ReshapeOptions as ropt

    # TODO why is this sometimes NONE?
    if op.BuiltinOptionsType() == BuiltinOptions.ReshapeOptions:

        opt = ropt.ReshapeOptions()
        opt.Init(options.Bytes, options.Pos)

        new_shape = opt.NewShape()

        print("New shape: {}".format(new_shape))


def process_FULLY_CONNECTED(op, options, io):
    from tflite import FullyConnectedOptions as fcopt

    assert (op.BuiltinOptionsType() == BuiltinOptions.FullyConnectedOptions)

    opt = fcopt.FullyConnectedOptions()
    opt.Init(options.Bytes, options.Pos)

    activation_func = class_code_to_name(ActivationFunctionType, opt.FusedActivationFunction())
    weights_format = class_code_to_name(FullyConnectedOptionsWeightsFormat, opt.WeightsFormat())
    keep_num_dim = opt.KeepNumDims()


def process_SOFTMAX(op, options, io):
    from tflite import SoftmaxOptions as smopt

    assert (op.BuiltinOptionsType() == BuiltinOptions.SoftmaxOptions)

    opt = smopt.SoftmaxOptions()
    opt.Init(options.Bytes, options.Pos)

    beta = opt.Beta()


def process_operation(model, graph, op):
    opcode_builtin = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
    op_name = class_code_to_name(BuiltinOperator, opcode_builtin)
    op_opts = op.BuiltinOptions()
    io_lengths = process_io_lengths(op)
    io = process_io(op)

    input_tensors = []
    for i, index in enumerate(io[0]):
        tensor = graph.Tensors(index)
        input_tensors.append((tensor.ShapeAsNumpy(), class_code_to_name(TensorType, tensor.Type())))
    output_tensors = []
    for i, index in enumerate(io[1]):
        tensor = graph.Tensors(index)
        output_tensors.append((tensor.ShapeAsNumpy(), class_code_to_name(TensorType, tensor.Type())))

    print("Processing {}, OP code: {}, options: {}".format(op_name, opcode_builtin,
                                            class_code_to_name(BuiltinOptions, op.BuiltinOptionsType())))
    print("Input tensors: {}, output tensors: {}".format(io_lengths[0], io_lengths[1]))
    print("Input: {}, output: {}".format(input_tensors, output_tensors))

    eval("process_" + op_name)(op, op_opts, (input_tensors, output_tensors))


def main():
    with open(model_filename, "rb") as f:
        model = Model.GetRootAsModel(f.read(), 0)
        graph = model.Subgraphs(0)

        for i in range(graph.OperatorsLength()):
            process_operation(model, graph, graph.Operators(i))


if __name__ == '__main__':
    main()
