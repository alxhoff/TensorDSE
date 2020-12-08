import os, sys
import tensorflow as tf

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

def print_options(options):
    if options:
        for key, item in options.items():
            print("{} -> {}".format(key, item))


def process_options(op, options):
    op_type = class_code_to_name(sys.modules['tflite'].BuiltinOptions.BuiltinOptions, 
                                 op.BuiltinOptionsType())

    if op_type != "NONE":
        import re

        opt = eval("sys.modules['tflite'].{}.{}()".format(op_type, op_type))
        opt.Init(options.Bytes, options.Pos)

        methods = [func for func in dir(opt) if
                   callable(getattr(opt, func)) and re.search(r'^((?!Init)(?!__)(?!{}).)*$'.format(op_type), func)]

        opts = {}
        for method in methods:
            opts[method] = eval("opt.{}()".format(method))

        return opts
    return None

def get_strides(options):
    return [1, options['StrideH'], options['StrideW'], 1]

def get_padding(options):
    return class_code_to_name(sys.modules['tflite'].Padding.Padding, options['Padding'])

# Pool size
def get_filter(options):
    return [options['StrideH'], options['StrideW']]

def get_input_tensor_shape(io):
    return io[0][0][0]

def get_kernel_shape(io):
    return io[0][1][0][1:]

def get_output_tensor_shape(io):
    return io[1][0]

def get_activation_function(options):
    return class_code_to_name(sys.modules['tflite'].ActivationFunctionType.ActivationFunctionType, 
                              options['FusedActivationFunction'])

def get_num_dims(options):
    return options['KeepNumDims']

def get_weights_format(options):
    return options['WeightsFormat']

