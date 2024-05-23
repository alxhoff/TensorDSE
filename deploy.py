"""
Missing  Docstring: TODO
"""

import multiprocessing.pool
import os
import sys
import copy
import argparse
import traceback
import threading
import multiprocessing

import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np

from utils.model import Model
from utils.logging.logger import log
from utils.splitter.utils import read_json_file
from utils.splitter.split import Splitter
from utils.splitter.split import MODELS_DIR
from utils.analysis import analyse_model_results_deploy

from utils.inference import Inference

from status_verifier import StatusVerifierSingleton as verifier


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def set_start_method():
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass


def preprocess_image_mnist(image, label, image_size, data_type):
    """
    Missing  Docstring: TODO
    """

    image = tf.image.resize(image, image_size)  # Resize to the input size required by the model
    image = tf.image.grayscale_to_rgb(image)  # Convert grayscale to RGB
    image = tf.cast(image, tf.float32) / 255.0  # Normalize the image to [0, 1] range
    if data_type == np.int8:
        image = tf.cast(image * 255.0 - 128, data_type)  # Convert to int8
    elif data_type == np.uint8:
        image = tf.cast(image * 255.0, data_type)  # Convert to uint8
    return image, label


def preprocess_image_imagenet(image, label, image_size, data_type):
    """
    Missing  Docstring: TODO
    """

    image = tf.image.resize(image, image_size)  # Resize to the input size required by the model
    image = tf.cast(image, tf.float32) / 255.0  # Normalize the image to [0, 1] range
    if data_type == np.int8:
        image = tf.cast(image * 255.0 - 128, data_type)  # Convert to int8
    elif data_type == np.uint8:
        image = tf.cast(image * 255.0, data_type)  # Convert to uint8
    return image, label


def preprocess_image_cifar10(image, label, image_size, data_type):
    """
    Missing  Docstring: TODO
    """

    image = tf.image.resize(image, image_size)  # Resize to the input size required by the model
    image = tf.cast(image, tf.float32) / 255.0  # Normalize the image to [0, 1] range
    if data_type == np.int8:
        image = tf.cast(image * 255.0 - 128, data_type)  # Convert to int8
    elif data_type == np.uint8:
        image = tf.cast(image * 255.0, data_type)  # Convert to uint8
    return image, label


def load_and_preprocess_dataset(dataset_name, image_size, data_type, preprocess_fn):
    """
    Missing  Docstring: TODO
    """

    dataset, info = tfds.load(dataset_name, with_info=True, as_supervised=True)
    test_dataset = dataset['test']

    test_dataset = test_dataset.map(lambda image, label: preprocess_fn(image,
                                                                       label,
                                                                       image_size,
                                                                       data_type))

    images = []
    labels = []

    for image, label in tfds.as_numpy(test_dataset):
        images.append(image)
        labels.append(label)

    images_np = np.array(images)
    labels_np = np.array(labels)

    return images_np, labels_np


def interpret_output(output_data, data_type, labels_np):
    """
    Missing  Docstring: TODO
    """

    result = True
    output_data = output_data.astype(np.float32)
    if data_type == np.int8:
        output_data = (output_data + 128) / 255.0  # Dequantize int8
    elif data_type == np.uint8:
        output_data = output_data / 255.0  # Dequantize uint8
    predicted_label = np.argmax(output_data)

    probabilities = softmax(output_data)
    predicted_label = np.argmax(probabilities)

    if predicted_label == labels_np[0]:
        result = True
    return result


class Deployer:
    """
    Missing  Docstring: TODO
    """

    HARDWARE_LOCKS = {
        'cpu': multiprocessing.Lock(),
        'gpu': multiprocessing.Lock(),
        'tpu': multiprocessing.Lock(),
    }


    def __init__(self,
                 native_deployment: bool,
                 data_module: str,
                 hardware: str) -> None:

        self.model_summary = verifier.get_model_summary(verifier)
        self.platform = verifier.get_platform()
        self.native_deployment = native_deployment
        self.data_module = data_module
        self.native_hardware = hardware

        self.models = []
        self.multi_models_sequences = []
        self.images = None
        self.labels = None


    def debug_active_threads_and_processes(self):
        log.info("Active threads and processes:")
        log.info(f"Active threads: {[thread.name for thread in threading.enumerate()]}")
        log.info(f"Active children: {multiprocessing.active_children()}")
        for p in multiprocessing.active_children():
            log.info(f"Terminating process: {p.pid}")
            p.terminate()
            p.join()


    def split(self):
        """
        Missing  Docstring: TODO
        """

        model_layer_sequences = []

        if self.native_deployment is True:
            for model in self.model_summary["models"]:
                for layer in model["layers"]:
                    layer["mapping"] = self.native_hardware

        splitter = Splitter(self.model_summary)

        if self.platform == "desktop":
            try:
                log.info("Running Model Splitter ...")
                splitter.run(sequences=True)
                log.info("Splitting Process Complete!\n")
            except RuntimeError:
                splitter.clean(True)
                log.fatal("Failed to run splitter!")

            # Compiles created models/layers into Coral models for execution
            splitter.compile_for_edge_tpu(bm=False)
            log.info("Models successfully compiled!")
            model_layer_sequences = splitter.model_layer_sequences
        else:
            model_layer_sequences = read_json_file("utils/splitter/model_layer_sequences.json")

        self.multi_models_sequences = model_layer_sequences


    def analyse_results(self):
        """
        Missing  Docstring: TODO
        """
        analyse_model_results_deploy(self.models, self.platform)


    def set_random_input_data(self, m: Model):
        """
        Missing  Docstring: TODO
        """
        input_size = m.get_array_size_from_shape(m.input_shape)
        m.input_vector = np.array(
                np.random.random_sample(input_size), dtype=m.get_np_dtype(m.input_datatype)
                )


    def set_input_test_data_module(self, m: Model):
        """
        Missing  Docstring: TODO
        """
        if self.data_module == 'MNIST':
            dataset_name = 'mnist'
            image_size = [224, 224]
            preprocess_fn = preprocess_image_mnist

        elif self.data_module == 'IMAGENET':
            dataset_name = 'imagenet2012'
            image_size = [224, 224]
            preprocess_fn = preprocess_image_imagenet

        elif self.data_module == 'CIFAR10':
            dataset_name = 'cifar10'
            image_size = [224, 224]
            preprocess_fn = preprocess_image_cifar10

        data_type = np.int8 if m.input_datatype == 'int8' else np.uint8

        images_np, labels_np = load_and_preprocess_dataset(dataset_name,
                                                           image_size,
                                                           data_type,
                                                           preprocess_fn)

        self.images = images_np
        self.labels = labels_np

        m.input_vector = images_np[0]


    def layer_deploy(self, m: Model, model_index):
        """
        Missing  Docstring: TODO
        """

        delegate_type  = m.delegate[:3]

        if delegate_type == "tpu":
            m.model_path = os.path.join(
                    MODELS_DIR,
                    f"model_{model_index}_{m.parent}",
                    "sub",
                    "compiled",
                    f"{m.model_name}_edgetpu.tflite"
                    )
        else:
            m.model_path = os.path.join(
                    MODELS_DIR,
                    f"model_{model_index}_{m.parent}",
                    "sub",
                    "tflite",
                    m.model_name,
                    f"{m.model_name}.tflite"
                    )

        inference_instance = Inference(model=m, count=1, platform=self.platform)
        m = inference_instance.invoke(force_a=True if m.delegate == "tpu" else False)

        return m


    def sequences_deploy(self, model_sequence, model_index):
        """
        Missing  Docstring: TODO
        """

        submodels = []
        model_name = os.path.basename(
            self.model_summary["models"][model_index]["path"]).split(".tflite")[0]

        for j, sequence in enumerate(model_sequence):
            layers = [self.model_summary["models"][model_index]["layers"][op[0]] for op in sequence]
            m = Model(layers, layers[0]["mapping"], model_name)

            if len(sequence) == 1:
                ops_range = sequence[0][0]
            else:
                ops_range = '-'.join(map(str, [sequence[0][0], sequence[-1][0]]))

            if self.native_deployment is False:
                m.model_name = f"submodel_{j}_{f'ops{ops_range}'}_{m.delegate}"
            else:
                m.model_name = model_name

            if j == 0:
                #Foced random input for rpi and dev board due to unsupported dataset installation
                if not self.platform == "desktop":
                    self.data_module = None
                else:
                    if self.data_module is None:
                        self.set_random_input_data(m)
                    else:
                        self.set_input_test_data_module(m)
            else:
                m.input_vector = copy.deepcopy(submodels[-1].output_vector)

            hardware_lock = self.HARDWARE_LOCKS[m.delegate]

            with hardware_lock:
                m = self.layer_deploy(m, model_index)

            if j == (len(model_sequence) - 1):
                if interpret_output(m.output_vector,
                                    m.get_np_dtype(m.output_datatype),
                                    self.labels) is True:
                    print("Correct Prediction for Image")
                else:
                    print("False Prediction for Image")


            submodels.append(m)
            log.info("Model %s | Layer %s deployed on %s", model_index, j, m.delegate)

        return submodels


    def multi_model_deploy(self):
        self.split()
        pool = None
        try:
            pool = multiprocessing.Pool(processes=4)
            async_results = [
                pool.apply_async(self.sequences_deploy, (model_sequences, i))
                for i, model_sequences in enumerate(self.multi_models_sequences)
            ]
            self.models = [async_result.get() for async_result in async_results]
        except Exception as e:
            log.error(f"An error occurred during deployment: {e}")
        finally:
            if pool is not None:
                pool.close()
                pool.join()
            self.debug_active_threads_and_processes()
        log.info("All models deployed!")
        self.analyse_results()


def get_arguments():
    """
    Missing  Docstring: TODO
    """

    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )

    parser.add_argument(
            "-s",
            "--summarypath",
            default="resources/model_summaries/example_summaries/\
                MNIST/MNIST_full_quanitization_summary_w_mappings.json",
            help="File that contains a model summary with mapping annotations"
            )

    parser.add_argument(
            "-d",
            "--dataset",
            default="MNIST",
            help="Dataset to be used for providing input data"
            )

    parser.add_argument(
            "-p",
            "--platform",
            default="desktop",
            help="Platform supporting the profiling/deployment process",
            )

    parser.add_argument(
            "-n",
            "--native",
            default=False,
            help="Native deployment on one single HW Target",
            )

    parser.add_argument(
            "-hw",
            "--hardware",
            default="cpu0",
            help="HW Target for native deployment",
            )


    return parser.parse_args()


if __name__ == "__main__":

    set_start_method()
    args = get_arguments()

    deployer = None
    if verifier.verify_args_for_deployer(verifier, args=args) is True:
        deployer = Deployer(args.native, args.dataset, args.hardware)

        try:
            deployer.multi_model_deploy()
        except RuntimeError:
            log.error("An error occured in the deployment process.")
            TRACE_STR = traceback.format_exc()
            log.info("Traceback to the Excepton: %s", TRACE_STR)

        log.info("[DEPLOY] Finished")
        sys.exit(0)
    else:
        log.fatal("Problem in Deployment!")
