# Global variables needed to generate the correct values
# for the freezing of each model.
OP_NAME     = ""
OP_IN_SHAPE = []
OP_IN_TYPE  = "float32"

def generator_init(op_name, input_place):
    """Sets global variables that will be used at the generator() function."""
    import numpy as np

    tf_type = str(input_place.dtype).split(" '")[1].split("'>")[0]

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
    import tensorflow as tf
    import numpy as np

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
    import tensorflow as tf
    from utils import extend_directory

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

    converter.experimental_new_converter = False

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

    generator_init(operation_name, input_place)

    tf_model_filename = op_dir + operation_name + ".tflite"
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
        print(f"Model {operation_name}'s input was not able to be quantized.")

    # Writes Conversion to pre-defined tflite folder.
    open(tf_model_filename, "wb").write(tflite_model)

    remove_directory(model_saved_dir)
