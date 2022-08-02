import argparse
from ast import operator
import logging
from typing import Dict
from tflite_helper import *
import tensorflow as tf

log = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

COMPILED_MODELS_FOLDER = "models/compiled/"
MODELS_FOLDER = "models/layers/"

# Functions with the format get_'object' simply retreive said object given a
# overloaded variable.


def get_strides(options):
    return [1, options["StrideH"], options["StrideW"], 1]


def get_padding(options):

    return ClassCodeToName(GetTFLiteClass("Padding"), options["Padding"])


def get_filter(options):
    # Pool Size
    return [options["StrideH"], options["StrideW"]]


def get_input_tensor_shape(io):
    return io[0][0][0]


def get_kernel_shape(io):
    return io[0][1][0][1:]


def get_output_tensor_shape(io):
    return io[1][0]


def get_activation_function(options):
    return ClassCodeToName(
        GetTFLiteClass("ActivationFunctionType"),
        options["FusedActivationFunction"],
    )


def get_num_dims(options):
    return options["KeepNumDims"]


def get_weights_format(options):
    return options["WeightsFormat"]


def get_activation_id(activ_func):

    options_dict = {"RELU": "relu", "SOFTMAX": "softmax", "DROPOUT": None}

    default = None
    return options_dict.get(activ_func, default)


def process_CONV_2D(operator_name, out_dir, options, input_tensors, output_tensors):
    """Processes the overloaded arguments to recreate the wished Conv 2D model."""
    import numpy as np
    from utils import tflite_conversion, save_session

    # The input tensors for a Conv2D layer are as follows:
    # [0] : The input shape to the Conv2D layer
    # [1] : The filter shape, ie. no of filters x kernel width x kernel height x kernel depth
    # [2] : The number of filters

    # Retrieving operation relevant variables.
    filter_count = input_tensors[2][0][0]

    input_shape = get_input_tensor_shape(io_tensors)
    test_input = np.array(np.random.random_sample(input_shape))

    kernel_shape = get_kernel_shape(io_tensors)

    padding = get_padding(options)
    strides = get_strides(options)
    activ_func = get_activation_function(options)

    conv_graph = tf.Graph()
    with conv_graph.as_default(), tf.compat.v1.Session() as sess:

        kernel_place = tf.Variable(
            tf.random.normal([3, 3, 1, filter_count], dtype="float32"), dtype=tf.float32
        )

        input_place = tf.raw_ops.Placeholder(
            dtype=tf.float32, shape=input_shape, name=operator_name + "_input"
        )

        conv_2d = tf.nn.conv2d(
            input_place,
            filters=kernel_place,
            strides=strides,
            padding=padding,
            name=operator_name + "_op",
        )

        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        output_place = tf.identity(conv_2d, name=operator_name + "_output")
        output_place = sess.run(conv_2d, feed_dict={input_place: test_input})

        tmp_model_saved_dir = save_session(
            sess, operator_name, conv_2d, out_dir, input_place
        )

    tflite_conversion(out_dir, tmp_model_saved_dir, operator_name, input_place)


def process_MAX_POOL_2D(options, io):
    """Processes the overloaded arguments to recreate the wished Max Pool 2D model."""
    import numpy as np
    from utils import tflite_conversion, save_session

    op_name = "MAX_POOL_2D"
    pool_dir = f"{MODELS_FOLDER}{op_name}/"

    pool_size = get_filter(options)
    padding = get_padding(options)
    strides = get_strides(options)
    activ_func = get_activation_function(options)

    input_shape = get_input_tensor_shape(io)
    test_input = np.array(np.random.random_sample(input_shape))

    pool_2d = tf.Graph()

    with pool_2d.as_default(), tf.compat.v1.Session() as sess:

        input_place = tf.raw_ops.Placeholder(
            dtype=tf.float32, shape=input_shape, name=op_name + "_input"
        )

        pool_2d = tf.nn.max_pool2d(
            input_place, 2, strides, padding, data_format="NHWC", name=op_name + "_op"
        )

        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        output_place = tf.identity(pool_2d, name=op_name + "_output")
        output_place = sess.run(pool_2d, feed_dict={input_place: test_input})

        tmp_model_saved_dir = save_session(
            sess, op_name, pool_2d, pool_dir, input_place
        )

    tflite_conversion(pool_dir, tmp_model_saved_dir, op_name, input_place)


def process_RESHAPE(options, io):
    """Processes the overloaded arguments to recreate the wished Reshape model."""
    import numpy as np
    from utils import tflite_conversion, save_session

    op_name = "RESHAPE"
    reshape_dir = f"{MODELS_FOLDER}{op_name}/"

    input_shape = get_input_tensor_shape(io)
    test_input = np.array(np.random.random_sample(input_shape), dtype=np.int32)

    output_shape = get_output_tensor_shape(io)

    reshape_graph = tf.Graph()
    with reshape_graph.as_default(), tf.compat.v1.Session() as sess:

        input_place = tf.raw_ops.Placeholder(
            dtype=tf.int32, shape=input_shape, name=op_name + "_input"
        )

        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        reshape = tf.reshape(
            input_place, (output_shape[0][0], output_shape[0][1]), name=op_name + "_op"
        )

        output_place = tf.identity(reshape, name=op_name + "_output")
        output_place = sess.run(reshape, feed_dict={input_place: test_input})

        tmp_model_saved_dir = save_session(
            sess, op_name, reshape, reshape_dir, input_place
        )

    tflite_conversion(reshape_dir, tmp_model_saved_dir, op_name, input_place)


def process_FULLY_CONNECTED(options, io):
    """Processes the overloaded arguments to recreate the wished FC model."""
    import numpy as np
    from utils import tflite_conversion, save_session

    activ_func = get_activation_function(options)
    activ_func = get_activation_id(activ_func)

    op_name = "FULLY_CONNECTED_" + activ_func if activ_func else "FULLY_CONNECTED"
    fcl_dir = f"{MODELS_FOLDER}{op_name}/"

    input_shape = get_input_tensor_shape(io)
    test_input = np.array(np.random.random_sample(input_shape))

    output_shape = get_output_tensor_shape(io)
    units = output_shape[0][1]

    weights_format = get_weights_format(options)

    keep_num_dim = get_num_dims(options)

    fully_connected_graph = tf.Graph()

    with fully_connected_graph.as_default(), tf.compat.v1.Session() as sess:

        input_place = tf.raw_ops.Placeholder(
            dtype=tf.float32, shape=input_shape, name=op_name + "_input"
        )

        fcl = tf.compat.v1.layers.dense(
            input_place, units, activ_func, use_bias=keep_num_dim, name=op_name + "_op"
        )

        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        output_place = tf.identity(fcl, name=op_name + "_output")
        output_place = sess.run(fcl, feed_dict={input_place: test_input})

        tmp_model_saved_dir = save_session(sess, op_name, fcl, fcl_dir, input_place)

    tflite_conversion(fcl_dir, tmp_model_saved_dir, op_name, input_place)


def process_SOFTMAX(options, io):
    """Processes the overloaded arguments to recreate the wished softmax model."""
    import numpy as np
    from utils import tflite_conversion, save_session

    op_name = "SOFTMAX"
    softmx_dir = f"{MODELS_FOLDER}{op_name}/"

    input_shape = get_input_tensor_shape(io)
    test_input = np.array(np.random.random_sample(input_shape))

    output_shape = get_output_tensor_shape(io)
    units = output_shape[0][1]

    softmx_graph = tf.Graph()
    with softmx_graph.as_default(), tf.compat.v1.Session() as sess:

        input_place = tf.raw_ops.Placeholder(
            dtype=tf.float32, shape=input_shape, name=op_name + "_input"
        )

        Soft = tf.nn.softmax(input_place, None, op_name + "_op")

        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        output_place = tf.identity(Soft, name=op_name + "_output")
        output_place = sess.run(Soft, feed_dict={input_place: test_input})

        tmp_model_saved_dir = save_session(sess, op_name, Soft, softmx_dir, input_place)

    tflite_conversion(softmx_dir, tmp_model_saved_dir, op_name, input_place)


def print_options(options):
    if options:
        for key, item in options.items():
            print("{} -> {}".format(key, item))


def ProcessOptions(operator) -> Dict:

    """Handles/Processes all operator types.

    This is done  by generically calling into the flattbuffer compiler generated
    module containing the options class for the target operator. Once the appropriate
    options class has been instantiated and initialized, all option get methods
    are called by filtering all callable methods of the options class using negative
    look-around regular expressions. Operator options are stored using their get
    method names as keys.

    :type operation: tflite.Operator.Operator
    :param operation: The operation whos opertions are to be returned

    :raises:

    :rtype: Dict
    """

    # flatbuffers.table.Table representation of the operation's options
    options_flatbuffer_table = operator.BuiltinOptions()

    options_class_name = ClassCodeToName(
        GetTFLiteClass("BuiltinOptions"),
        operator.BuiltinOptionsType(),
    )

    if options_class_name != "NONE":
        options_class = InitOptionsClass(
            options_class_name=options_class_name,
            options_fb_table=options_flatbuffer_table,
        )

        return GetOptions(options_class)
    return None


def _ProcessTensorID(id, graph):

    """Gets the input shape and type of a tensor given its ID
    :type id: str
    :param id: String ID of the tensor whos input shape and type should be returned

    :type graph: tflite.SubGraph.SubGraph
    :param graph: The graph from which the tensors should be taken

    :raises:

    :rtype: tuple
    """
    tensor = graph.Tensors(id)
    shape = tensor.ShapeAsNumpy()
    tensor_type = ClassCodeToName(GetTFLiteClass("TensorType"), tensor.Type())

    return shape, tensor_type


def ProcessIO(operator, graph):
    """Processes the overloaded arguments to recreate the wished operation's model."""
    input_tensor_count, output_tensor_count = GetOperatorInputOutputLengths(operator)
    inputs, outputs = [], []
    for i in range(input_tensor_count):
        inputs.append(_ProcessTensorID(id=operator.Inputs(i), graph=graph))
    for i in range(output_tensor_count):
        outputs.append(_ProcessTensorID(id=operator.Outputs(i), graph=graph))

    return inputs, outputs


def ProcessLayer(layer_name, options, input_tensors, output_tensors) -> None:

    out_dir = f"{MODELS_FOLDER}{layer_name}/"
    eval("process_" + layer_name)(layer_name, out_dir, options, input_tensors, output_tensors)


def ProcessOperation(model, graph, operator) -> None:

    """Each operator is parsed from the schema to obtain the operations's options,

    This is required for creating the operation in tensorflow, given the appropriate
    shapes (input, output, kernels etc). Parsing and processing of operations is done
    through the use of the operation's builtin name/options name which are resolved
    using the flattbuffer compiler's python classes, because of this, function naming
    for the processing of each operation type must be adhered to.

    Processing functions should be named `process_` + the operator name as per the
    BuiltinOperator enum in the TFlite schema.

    Similarly, operation options are parsed by calling all available get methods
    pulled from the appropriate schema options objects. This is done in process_options.
    The options for each operation are returned in a dictionary where the keys for
    each option are the strings used in the schema generated python classes, eg.
    for a fully connected operation, the keep_num_dims option (from the flattbuffer schema)
    is stored using the key found in FullyConnectedOptions.py, namely KeepNumDims.
    :type model: tflite.Model.Model
    :param model: The schema generated tflite helper model representation of the target model

    :type graph: tflite.SubGraph.SubGraph
    :param graph: The target model's graph

    :type operator: tflite.Operator.Operator
    :param operator: The operator currently being processed from the target model's graph

    :raises:

    :rtype: None
    """
    operator_name = GetOpNameFromOperator(model=model, operator=operator)
    log.info(f"Processing the operation: {operator_name}")

    operator_options = ProcessOptions(operator=operator)
    input_tensors, output_tensors = ProcessIO(operator=operator, graph=graph)

    ProcessLayer(operator_name, operator_options, input_tensors, output_tensors)


def split_tflite_model(model) -> None:

    """Splits/Processes the given as input tflite model into its individual operations.

    Individually calls onto the process_operation function that processes and
    eventually recreates the operation overloaded onto 'process_operation(...)'
    as 'graph.Operators(i)'.
    """
    with open(model, "rb") as f:
        # Load our model from file and create the flatbuffer parser object for the root node
        model = GetTFLiteClass("Model").GetRootAsModel(f.read(), 0)
        graph = model.Subgraphs(0)

        for op in [graph.Operators(i) for i in range(graph.OperatorsLength())]:
            ProcessOperation(model, graph, op)


def args() -> argparse.Namespace:

    """Argument parser, returns the Namespace containing all of the arguments.
    :raises: None

    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-m",
        "--model",
        default="models/source/MNIST.tflite",
        help="File path to the SOURCE .tflite file.",
    )

    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=1000,
        help="Number of times to measure inference.",
    )

    args = parser.parse_args()

    return args


def import_tflite_modules() -> None:

    """The automatically generated schema helper modules, generated by the flatc flatbuffer compiler,
    need to be imported. As they sit together in the `tflite` folder one must import all modules found
    withing that folder. Modules are imported using __import__ and then added to system modules.

    :raises: None
    :rtype: None
    """
    import os, sys

    tflite_path = os.path.join(os.path.dirname(__file__), "tflite")

    log.info("Importing flatbuffers API...")

    # For each .py stripped file in the tflite folder
    for py in [
        f[:-3]
        for f in os.listdir(tflite_path)
        if f.endswith(".py") and f != "__init__.py"
    ]:
        mod_name = ".".join(["tflite", py])
        mod_imported = __import__(mod_name, fromlist=[py])
        mod_classes = [
            getattr(mod_imported, x)
            for x in dir(mod_imported)
            if isinstance(getattr(mod_imported, x), type)
        ]

        # Make all of the module's classes available from sys.modules
        for cls in mod_classes:
            setattr(sys.modules[__name__], cls.__name__, cls)


if __name__ == "__main__":

    """Entry point to execute this script.

    Flags
    ---------
    -t or --target
        Target input tflite model to be processed and splitted.

    -c or --count
        Used in the tflite deployment that may occur directly after conversion.
        With count it is set the number of deployments done.
    """

    # from docker import remove_project
    # from deploy import tflite_deployment
    # from cmpile import tflite_compilation
    # from analyze import tflite_results_analysis

    args = args()
    import_tflite_modules()

    split_tflite_model(model=args.model)

    tflite_compilation()
    tflite_deployment("models/compiled/", "models/layers/", count=args.count)
    tflite_results_analysis()
    remove_project()
