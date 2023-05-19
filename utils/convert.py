# Global variables needed to generate the correct values
# for the freezing of each model.
OP_NAME     = ""
OP_IN_SHAPE = []
OP_IN_TYPE  = "float32"

def generator_init(op_name, input_place):
    """Sets global variables that will be used at the generator() function."""
    import numpy as np
    import tensorflow as tf

    tf_type = str(input_place.dtype).split(" '")[1].split("'>")[0]

    global OP_NAME
    global OP_IN_SHAPE
    global OP_IN_TYPE

    OP_IN_SHAPE = []
    OP_NAME     = op_name
    OP_IN_TYPE  = np.float32 if tf_type == 'float32' else \
                  np.int32   if tf_type == 'int32' else None

    for i in range(len(input_place.shape)):
        OP_IN_SHAPE.append(input_place.shape[i])

    OP_IN_SHAPE = tuple(OP_IN_SHAPE)


def generator():
    """Generates input samples to possibilitate quantization before conversion
    and compilation for the edge TPU.
    """
    import numpy as np
    import tensorflow as tf

    for _ in range(100):
        input_data = np.array(
                    np.random.random_sample(OP_IN_SHAPE),
                    dtype=OP_IN_TYPE
                )
        input_data = tf.convert_to_tensor(input_data)
        yield [input_data]


def SaveSession(session, operation_name, operation, op_dir, input_placeholder):
    """Quantizes a converter object so that the compilation on the edge tpu is
    made possible.

    Parameters
    ----------
    session : Session object
    Object created during the session creation when recreating single operations.

    operation_name : String
    The name of the operation.

    operation : Object
    The tenorflow layer creating during when recreating single operations.
    Ex: conv_2d = tf.nn.conv2d(...) -> conv_2d will be passed here as operation.

    op_dir : String
    Path to the directory in which the session will be saved at.

    input_place : tensor
    Input Placeholder created when recreating single operations.

    Returns
    -------
    converter : tf.lite.TFLiteConverter.from_saved_model()
    """
    from utils import extend_directory
    import tensorflow as tf

    # Clears saved model directory.
    export_dir = extend_directory(op_dir, "tmp")

    # Saving Model into the saved model directory.
    tf.compat.v1.saved_model.simple_save(session,
                                         export_dir,
                                         inputs={operation_name +
                                                 "_input": input_placeholder},
                                         outputs={operation_name+"_op": operation})
    return export_dir


def TFLiteQuantization(converter):
    """Quantizes a converter object so that the compilation on the edge tpu is
    made possible.

    Parameters
    ----------
    converter : tf.lite.TFLiteConverter.from_saved_model() Object
    Object created during the conversion of a tensorflow session, which in its
    place was exported to a specific folder.

    Returns
    -------
    converter : tf.lite.TFLiteConverter.from_saved_model()
    """
    import tensorflow as tf

    # True enables MLIR-based conversion
    # Pro of turning on/True:
    #   - faster execution
    #   - less warning to stdout
    #   - no polluting of logger with warnings from tf
    # Pro of turning on/True:
    #   - I dont know if thats what we want
    converter.experimental_new_converter = True

    # This enables Quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # This sets the representative dataset for quantization
    converter.representative_dataset = tf.lite.RepresentativeDataset(generator)

    # This ensures that if OPS cant be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # For full integer quantization, through supported types defaults to int8, declared for clarity
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.uint8  # or tf.uint8
    converter.inference_output_type = tf.uint8  # or tf.uint8

    return converter


def TFLiteConverter(op_dir, model_saved_dir, operation_name, input_place):
    """Manager function responsible for converting a tensorflow session into a
    tflite model.

    Parameters
    ----------
    op_dir : String
    Path to the directory where the tflite model of this operation will be
    saved.

    model_saved_dir : String
    Path to the directory where the exported tensorflow session has been saved
    to. Needed for the converter to be able to read the model and convert it to
    a tflite format.

    operation_name : String
    Name of the operation.

    input_place : Tensorflow Tensor
    Shape and type of input tenosr necessary to generate samples to quantize the
    to be produced tflite model.
    """
    import tensorflow as tf
    from utils import extend_directory, remove_directory
    from os.path import join

    generator_init(operation_name, input_place)

    tf_model_filename = join(op_dir, operation_name + ".tflite")
    edge_tf_model_filename = extend_directory(op_dir, "quant") + \
                             "quant_" + operation_name + ".tflite"

    # Creates Converter Object.
    converter = tf.lite.TFLiteConverter.from_saved_model(model_saved_dir)
    # Creates another Converter Object.
    edge_converter = tf.lite.TFLiteConverter.from_saved_model(model_saved_dir)
    # Quanitzation of tflite model, necessary for edge
    edge_converter = TFLiteQuantization(edge_converter)
    # Performs tflite conversion with it.
    tflite_model = converter.convert()

    try:
        # Performs quantized tflite conversion with it.
        edge_tflite_model = edge_converter.convert()

        # Writes Conversion to pre-defined tflite folder
        open(edge_tf_model_filename, "wb").write(edge_tflite_model)

    except ValueError:
        print(f"Model: {operation_name}'s input was not able to be quantized.")

    # Writes Conversion to pre-defined tflite folder.
    open(tf_model_filename, "wb").write(tflite_model)

    remove_directory(model_saved_dir)

from array import array
from ast import operator

from typing import Dict
from utils.tflite_helper import *

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
    return {}


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

    """Returns a tuple of lists containing the input and output tensors, resolved into Numpy arrays
    :type operator: tflite.Operator.Operator
    :param operator: Operator who's input and output tensors should be retrieved

    :type graph: tflite.SubGraph.SubGraph
    :param graph: Graph from which the operator should be retrieved

    :raises:

    :rtype: tuple
    """
    input_tensor_count, output_tensor_count = GetOperatorInputOutputLengths(operator)
    inputs, outputs = [], []
    for i in range(input_tensor_count):
        inputs.append(_ProcessTensorID(id=operator.Inputs(i), graph=graph))
    for i in range(output_tensor_count):
        outputs.append(_ProcessTensorID(id=operator.Outputs(i), graph=graph))

    return inputs, outputs


def ProcessLayer(layer_name, options, input_tensors, output_tensors) -> None:
    """ Calls the appropriate 'process_$LAYER_NAME' function
    :type layer_name: str
    :param layer_name: Name of the layer to be processed, eg. CONV_2D

    :type options: Class containing the layers options
    :param options: Options class for the layer of interest

    :type input_tensors: array
    :param input_tensors: Array of all the input tensors

    :type output_tensors: array
    :param output_tensors: Array of all the output tensors

    :raises:

    :rtype: None
    """
    from os.path import join
    import utils.tflite_helper.process_layers as pl

    from main import log, LAYERS_FOLDER

    processors = {
            "CONV_2D"           : pl.process_CONV_2D,
            "FULLY_CONNECTED"   : pl.process_FULLY_CONNECTED,
            "MAX_POOL_2D"       : pl.process_MAX_POOL_2D,
            "RESHAPE"           : pl.process_RESHAPE,
            "SOFTMAX"           : pl.process_SOFTMAX,
    }

    out_dir = join(LAYERS_FOLDER, layer_name)
    p = processors.get(layer_name, None) # returns None if no matching key in dict

    if p:
        p(layer_name, out_dir, options, input_tensors, output_tensors)
        return

    log.error(f"{layer_name} DOES NOT HAVE CORRESPONDING LAYER PROCESSING FUNCTION!")


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
    from main import log

    operator_name = GetOpNameFromOperator(model=model, operator=operator)

    operator_options = ProcessOptions(operator=operator)
    input_tensors, output_tensors = ProcessIO(operator=operator, graph=graph)

    log.info(f"Processing the operation: {operator_name}")
    ProcessLayer(operator_name, operator_options, input_tensors, output_tensors)


def SplitTFLiteModel(model) -> list:

    """Splits/Processes the given as input tflite model into its individual operations.

    Individually calls onto the process_operation function that processes and
    eventually recreates the operation overloaded onto 'process_operation(...)'
    as 'graph.Operators(i)'.

    :rtype: list of strins
        each entry is one of the layers that compose the
        to-be-benchmarked model
    """
    from main import log
    layers = []
    with open(model, "rb") as f:
        # Load our model from file and create the flatbuffer parser object for the root node
        model = GetTFLiteClass("Model").GetRootAsModel(f.read(), 0)
        graph = model.Subgraphs(0)
        log.info(f"Model subdivided into {graph.OperatorsLength()} operators")

        for operator in [graph.Operators(i) for i in range(graph.OperatorsLength())]:
            layers.append(GetOpNameFromOperator(model=model, operator=operator))
            ProcessOperation(model, graph, operator)

    return layers

def ImportTFLiteModules() -> None:

    """The automatically generated schema helper modules, generated by the flatc flatbuffer compiler,
    need to be imported. As they sit together in the `tflite` folder one must import all modules found
    withing that folder. Modules are imported using __import__ and then added to system modules.

    :raises: None
    :rtype: None
    """
    import os, sys
    from main import log

    tflite_path = os.path.join(os.getcwd(), "tflite")

    log.info("Importing flatbuffers API")

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



