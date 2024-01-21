"""
Module Docstring: TODO
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue

import sys
import time
import numpy as np

from utils.model import Model
from utils.logging.log import Log
from utils.logging.logger import log
from utils.usb import END_DEPLOYMENT
from utils.usb.usb import capture_stream
from backend.distributed_inference import distributed_inference


class InferneceContext():
    """
    The Context defines the interface of interest to clients.
    """

    def __init__(self, strategy: InferenceStrategy) -> None:
        self._strategy = strategy

    @property
    def strategy(self) -> InferenceStrategy:
        """
        Missing  Docstring: TODO
        """
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: InferenceStrategy) -> None:
        self._strategy = strategy

    def initiate(self) -> Model:
        """
        Missing  Docstring: TODO
        """
        return self._strategy.invoke()


class InferenceStrategy(ABC):
    """
    The Inference Strategy
    """
    def __init__(self, count: int, platform: str) -> None:
        super().__init__()
        self.count = count
        self.platform = platform

    @abstractmethod
    def invoke(self, model: Model):
        """
        Missing  Docstring: TODO
        """


class InferenceScenarioA(InferenceStrategy):
    """
    Missing  Docstring: TODO
    """

    def invoke(self, model: Model) -> Model:
        input_size = model.get_array_size_from_shape(model.input_shape)
        output_size = model.get_array_size_from_shape(model.output_shape)

        input_data_vector = np.zeros(input_size).astype(
                model.get_np_dtype(model.input_datatype)
                )
        output_data_vector = np.zeros(output_size).astype(
                model.get_np_dtype(model.output_datatype)
                )
        inference_times_vector = np.zeros(self.count).astype(np.uint32)

        mean_inference_time = 0
        try:
            mean_inference_time = distributed_inference(
                    model.model_path,
                    input_data_vector,
                    output_data_vector,
                    inference_times_vector,
                    len(input_data_vector),
                    len(output_data_vector),
                    model.delegate,
                    self.platform,
                    self.count,
                    0
                    )
        except RuntimeError:
            log.error("An error was encountered while trying to invoke \
                      the Standard Infernce Strategy A. \
                      Mean inference Time is: %s", mean_inference_time)

        model.output_vector = output_data_vector
        model.results = [t/1000000000.0 for t in inference_times_vector.tolist()]

        return model


class InferenceScenarioB(InferenceStrategy):
    """
    Missing  Docstring: TODO
    """

    TIMEOUT = 10
    CORE_INDEX = 0
    USBMON = 0
    DEPLOY_WAIT_TIME = 3

    def invoke(self, model: Model) -> Model:
        """
        Missing  Docstring: TODO
        """

        input_size = model.get_array_size_from_shape(model.input_shape)
        output_size = model.get_array_size_from_shape(model.output_shape)
        input_data_vector = None
        output_data_vector = None  # or some default value

        time.sleep(5)

        for i in range(self.count):
            signals_q = Queue()
            data_q = Queue()

            p = Process(
                    target=capture_stream,
                    args=(
                        signals_q,
                        data_q,
                        self.TIMEOUT,
                        Log(f"resources/logs/layer_{model.index}_{model.model_name}_USB.log"),
                        self.USBMON,
                        ),
                    )

            p.start()

            sig = signals_q.get()
            if sig == END_DEPLOYMENT:
                p.join()
                break


            if model.input_vector is None:
                input_data_vector = np.array(
                        np.random.random_sample(input_size),
                        dtype=model.get_np_dtype(model.input_datatype),
                        )
            else:
                input_data_vector = model.input_vector

            output_data_vector = np.zeros(output_size).astype(
                    model.get_np_dtype(model.output_datatype)
                    )

            inference_times_vector = np.zeros(self.count).astype(np.uint32)

            mean_inference_time = distributed_inference(
                    model.model_path,
                    input_data_vector,
                    output_data_vector,
                    inference_times_vector,
                    len(input_data_vector),
                    len(output_data_vector),
                    "tpu",
                    "rpi_test" if self.platform == "rpi" else self.platform,
                    1,
                    self.CORE_INDEX
                    )

            time.sleep(self.DEPLOY_WAIT_TIME)

            model.results.append(mean_inference_time)

            try:
                data = data_q.get(timeout=10)
            except RuntimeError:
                log.error("No data is received from USB analysis")
                data = None

            if not data == {}:
                model.timers.append(data)

            if p.is_alive():
                p.join()

            sys.stdout.write(f"\r {i+1}/{self.count} for TPU ran -> {model.model_name}")
            sys.stdout.flush()

        sys.stdout.write("\n")

        model.output_vector = output_data_vector

        return model


class Inference:
    """
    Missing  Docstring: TODO
    """

    def __init__(self, model: Model, count: int, platform: str) -> None:
        self.model = model
        self.count = count
        self.platform = platform

    def identify_strategy(self) -> InferenceStrategy:
        """
        Missing  Docstring: TODO
        """
        if ((self.platform == "desktop") or (self.platform == "rpi")):
            if self.model.delegate == "tpu":
                return InferenceScenarioB(self.count, self.platform)
            return InferenceScenarioA(self.count, self.platform)
        elif self.platform == "coral":
            return InferenceScenarioA(self.count, self.platform)


    def invoke(self) -> Model:
        """
        Missing  Docstring: TODO
        """
        strategy = self.identify_strategy()
        context = InferneceContext(strategy)
        return context.initiate()
