from array import array
from ast import operator

from typing import Dict
from tflite_helper import *

import tensorflow as tf

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
    import tflite_helper.process_layers as pl

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


def SplitTFLiteModel(model) -> None:

    """Splits/Processes the given as input tflite model into its individual operations.

    Individually calls onto the process_operation function that processes and
    eventually recreates the operation overloaded onto 'process_operation(...)'
    as 'graph.Operators(i)'.
    """
    from main import log
    with open(model, "rb") as f:
        # Load our model from file and create the flatbuffer parser object for the root node
        model = GetTFLiteClass("Model").GetRootAsModel(f.read(), 0)
        graph = model.Subgraphs(0)
        log.info(f"Model subdivided into {graph.OperatorsLength()} operators")

        for operator in [graph.Operators(i) for i in range(graph.OperatorsLength())]:
            ProcessOperation(model, graph, operator)


def ImportTFLiteModules() -> None:

    """The automatically generated schema helper modules, generated by the flatc flatbuffer compiler,
    need to be imported. As they sit together in the `tflite` folder one must import all modules found
    withing that folder. Modules are imported using __import__ and then added to system modules.

    :raises: None
    :rtype: None
    """
    import os, sys
    from main import log

    tflite_path = os.path.join(os.path.dirname(__file__), "tflite")

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


