from tflite.ActivationFunctionType import ActivationFunctionType
from tflite.BuiltinOperator import BuiltinOperator
from tflite.BuiltinOptions import BuiltinOptions
from tflite.Model import Model
from tflite.Padding import Padding

# tflite folder generated using tflite schema and the flattbuffer compiler
# See  https://google.github.io/flatbuffers/flatbuffers_guide_tutorial.html

model_filename = "MNIST_model.tflite"


def class_code_to_name(cls, code):
    for name, value in cls.__dict__.items():
        if value == code:
            return name
    return None


# Functions to process each operation take the form of "process_" + the builtin opcode name that can
# be found in the TFLite schema under `BuiltinOperator`. This way the functions can be resolved using `eval` and
# the resolved builtin operator name.

def process_CONV_2D(op, options):
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


def process_MAX_POOL_2D(op, options):
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

def process_RESHAPE(op, options):
    from tflite import ReshapeOptions as ropt

    #TODO why is this sometimes NONE?
    if op.BuiltinOptionsType() == BuiltinOptions.ReshapeOptions:

        opt = ropt.ReshapeOptions()
        opt.Init(options.Bytes, options.Pos)

        new_shape = opt.NewShape()

        print("New shape: {}".format(new_shape))


def process_FULLY_CONNECTED(op, options):
    pass


def process_SOFTMAX(op, options):
    pass


def process_operation(model, op):
    opcode_builtin = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
    op_name = class_code_to_name(BuiltinOperator, opcode_builtin)
    op_opts = op.BuiltinOptions()

    print("Processing {}, OP code: {}, option:{}".format(op_name, opcode_builtin, class_code_to_name(BuiltinOptions, op.BuiltinOptionsType())))

    eval("process_" + op_name)(op, op_opts)


def main():
    with open(model_filename, "rb") as f:
        model = Model.GetRootAsModel(f.read(), 0)
        graph = model.Subgraphs(0)

        for i in range(graph.OperatorsLength()):
            process_operation(model, graph.Operators(i))


if __name__ == '__main__':
    main()
