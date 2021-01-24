# Global variables needed to generate the correct values for the freezing
# of each model.
gen_op_name = ""

gen_op_in_shape = []

gen_op_in_type = "float32"


def get_numpy_type(tensor_type):
    import numpy as np

    tf_type_dict = {
        'float32': np.float32,
        'int32': np.int32
    }

    default = None
    return tf_type_dict.get(tensor_type, default)


def place_within_quotes(string):
    from shlex import quote
    return "".join(quote(string))


def concat_args(args):
    summed_args = ""
    for arg in range(len(args)):
        summed_args += args[arg]
    return summed_args



def parse_csv(filename):
    import csv

    samples = []

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            samples.append(float(row[1]))

    return samples


def create_csv_file(path_file, folder_name, results):
    import os
    import csv

    csv_dir = path_file + folder_name + "/"
    csv_file = csv_dir + "/" + "Results.csv"

    if (os.path.exists(path_file)):
        if (not os.path.exists(csv_dir)):
            mkdir_cmd = "mkdir " + csv_dir
            os.system(mkdir_cmd)
        else:
            clean_up_cmd = "rm -r " + csv_dir + "*"
            os.system(clean_up_cmd)
    else:
        raise NotImplementedError

    with open(csv_file, 'w') as csvfile:
        fw = csv.writer(csvfile, delimiter=',', quotechar='|',
                        quoting=csv.QUOTE_MINIMAL)

        for i in range(len(results)):
            fw.writerow([results[i][0], results[i][1]])


def deduce_operation_from_file(tflite_file, beginning=None, ending=None):
    f = tflite_file
    op = ""

    if(beginning and not ending):
        if beginning in tflite_file:
            op = f.split(beginning)[1]

    elif(beginning and ending):
        if beginning in tflite_file and ending in tflite_file:
            op = f.split(beginning)[1]
            op = op.split(ending)[0]

    elif(ending and not beginning):
        if ending in tflite_file:
            op = f.split(ending)[0]

    return op


def deduce_operations_from_folder(models_folder, beginning=None, ending=None):
    import os
    from os import listdir
    from os.path import isfile, isdir, join

    tflite_models_info = []

    for f_1 in listdir(models_folder):
        f_1_path = models_folder + f_1
        if isdir(f_1_path):
            for f_2 in listdir(f_1_path):
                f_2_path = f_1_path + "/" + f_2
                if (isfile(f_2_path) and f_2.endswith(".tflite")):
                    op = deduce_operation_from_file(
                        f_2, beginning=beginning, ending=ending)
                    tflite_models_info.append([f_2_path, op])

        elif isfile(f_1_path):
            if (isfile(f_1_path) and f_1.endswith(".tflite")):
                op = deduce_operation_from_file(
                    f_1, beginning=beginning, ending=ending)
                tflite_models_info.append([f_1_path, op])

    return tflite_models_info


def retrieve_folder_path(path, folder):
    return path.split(folder + "/")[0] + folder + "/"


def extend_directory(path_to_dir, extended_dir, parent_dir=""):
    import os

    if (os.path.exists(path_to_dir)):
        ext_dir = path_to_dir + extended_dir
        if (not os.path.exists(ext_dir)):
            mkdir_cmd = "mkdir " + ext_dir
            os.system(mkdir_cmd)
        else:
            rm_cmd = "rm -r " + ext_dir
            mkdir_cmd = "mkdir " + ext_dir
            os.system(rm_cmd)
            os.system(mkdir_cmd)
    else:
        ext_dir = path_to_dir + extended_dir
        mkdir_cmd = "mkdir -p" + ext_dir
        os.system(mkdir_cmd)

    return ext_dir + "/"


def clean_directory(path_to_dir):
    import os

    if (os.path.exists(path_to_dir)):
        if (os.listdir(path=path_to_dir) != 0):
            rm_cmd = "rm -r " + path_to_dir
            os.system(rm_cmd)
    else:
        raise NotImplementedError


def prep_dataset_generator(op_name, input_place):
    import random

    global gen_op_name
    global gen_op_in_shape
    global gen_op_in_type

    gen_op_in_shape = []
    gen_op_name = op_name
    gen_op_in_type = get_numpy_type(
        str(input_place.dtype).split(" '")[1].split("'>")[0])

    for i in range(len(input_place.shape)):
        gen_op_in_shape.append(input_place.shape[i])

    gen_op_in_shape = tuple(gen_op_in_shape)


def generator():
    import tensorflow as tf
    import numpy as np

    for _ in range(100):
        input_data = np.array(np.random.random_sample(
            gen_op_in_shape), dtype=gen_op_in_type)
        input_data = tf.convert_to_tensor(input_data)
        yield [input_data]


def save_session(session, operation_name, operation, op_dir, input_place):
    import tensorflow as tf

    # Clears saved model directory.
    tmp_model_saved_dir = extend_directory(op_dir, "tmp", operation_name)

    # Saving Model into the saved model directory.
    tf.compat.v1.saved_model.simple_save(session,
                                         tmp_model_saved_dir,
                                         inputs={operation_name +
                                                 "_input": input_place},
                                         outputs={operation_name+"_op": operation})
    return tmp_model_saved_dir


def tflite_quantization(converter):
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


def tflite_conversion(op_dir, model_saved_dir, operation_name, input_place):
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

    tf_model_filename = op_dir + operation_name + ".tflite"
    prep_dataset_generator(operation_name, input_place)

    edge_dir = extend_directory(op_dir, "quant", operation_name)
    edge_tf_model_filename = edge_dir + "quant_" + operation_name + ".tflite"

    # Creates Converter Object.
    converter = tf.lite.TFLiteConverter.from_saved_model(model_saved_dir)
    # Creates another Converter Object.
    edge_converter = tf.lite.TFLiteConverter.from_saved_model(model_saved_dir)
    # Quanitzation of tflite model, necessary for edge
    edge_converter = tflite_quantization(edge_converter)
    # Performs tflite conversion with it.
    tflite_model = converter.convert()
    # Performs quantized tflite conversion with it.
    edge_tflite_model = edge_converter.convert()

    # Writes Conversion to pre-defined tflite folder.
    open(tf_model_filename, "wb").write(tflite_model)
    # Writes Conversion to pre-defined tflite folder
    open(edge_tf_model_filename, "wb").write(edge_tflite_model)

    clean_directory(model_saved_dir)
