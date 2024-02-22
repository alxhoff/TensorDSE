"""
Module Docstring: TODO
"""

import argparse

from typing import Tuple

import utils

from utils.logging.logger import log
from utils.splitter.utils import read_json_file


class SingletonMeta(type):
    """
    Singleton Meta Class
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class StatusVerifierSingleton(metaclass=SingletonMeta):
    """
    StatusVerifier Docstring
    """
    _platform = ""
    _multi_model_summary = None
    _hardware_summary = None
    _requested_hardware = None
    _active_process = None

    def __init__(self):
        pass


    def verify_hardware_for_profiling(self, hardware_summary: dict) -> bool:
        """
        Verifies and identifies the hardware used to run the profiling
        """

        req_hw = []
        if int(hardware_summary["CPU_count"]) > 0:
            pe_vailable = self.verify_pe(pe="cpu")[0]
            if pe_vailable:
                req_hw.append("cpu")
            else:
                log.fatal("The CPU requested for Profiling is not available. Abort")

        if int(hardware_summary["GPU_count"]) > 0:
            pe_vailable= self.verify_pe(pe="gpu")[0]
            if pe_vailable:
                req_hw.append("gpu")
            else:
                log.fatal("The GPU requested for Profiling is not available. Abort")

        if int(hardware_summary["TPU_count"]) > 0:
            pe_vailable = self.verify_pe(pe="tpu")[0]
            if pe_vailable:
                req_hw.append("tpu")
            else:
                log.fatal("The TPU requested for Profiling is not available. Abort")

        if len(req_hw) == 0:
            return False

        self._requested_hardware = req_hw
        return True


    def verify_args_for_profiler(self, args: argparse.Namespace) -> bool:
        """
        Verifies the arguments given to the profiler.py script 
        """

        if args.count < 2:
            log.error("The Benchmarking Count must be greater than 2. \
                      Current Count is %s.", args.count)
            return False

        if args.summarypath is not None:
            model_summary_json = read_json_file(args.summarypath)
            if model_summary_json is None:
                log.fatal("The provided Model Summary is empty!")
                return False
        else:
            log.fatal("The provided path to the Model Summary is not valid!")
            return False

        if args.hardwaresummary is not None:
            hardware_summary_json = read_json_file(args.hardwaresummary)
            if hardware_summary_json is None:
                log.fatal("The provided Hardware Summary is empty!")
                return False
        else:
            log.fatal("The provided path to the Hardware Summary is not valid!")
            return False

        self._platform = args.platform
        self._multi_model_summary = model_summary_json
        self._hardware_summary = hardware_summary_json
        self._active_process = "Profiling"

        return True


    def verify_args_for_deployer(self, args: argparse.Namespace) -> bool:
        """
        Verifies the arguments given to the deploy.py script 
        """
        if args.summarypath is not None:
            summary = read_json_file(args.summarypath)
            if summary is None:
                log.fatal("The provided Model Summary is empty!")
        else:
            log.fatal("The provided Path to the Summary is not valid!")

        self._multi_model_summary = summary


    def set_model_summary(self, summary: dict) -> None:
        """
        setter function for multi_model_summary
        """
        self._multi_model_summary = summary


    def get_model_summary(self) -> dict:
        """
        getter function for multi_model_summary
        """
        return self._multi_model_summary


    def set_hardware_summary(self, summary) -> None:
        """
        setter function for hardware_summary
        """
        self._hardware_summary = summary


    def get_hardware_summary(self) -> dict:
        """
        getter function for hardware_summary
        """
        return self._hardware_summary


    def set_requested_hardware(self, hw_list: list) -> None:
        """
        setter function for requested_hardware
        """
        self._requested_hardware = hw_list


    def get_requested_hardware(self) -> list:
        """
        getter function for hardware_summary
        """
        return self._requested_hardware

    @staticmethod
    def get_platform() -> str:
        """
        getter function for platform
        """
        return StatusVerifierSingleton._platform


    def set_platform(self, platform: str) -> None:
        """
        setter function for platform
        """
        self._platform = platform

    @staticmethod
    def verify_pe(pe: str) -> Tuple[bool, str]:
        """
        Verifies if a specific PE is available or not.
        """
        result = None
        if pe == "cpu":
            result = True, "cpu0"
        elif pe == "gpu":
            if StatusVerifierSingleton.get_platform() != "desktop":
                result = True, ""
            else:
                out = utils.run("lshw -numeric -C display").split("\n")
                for line in out:
                    if "vendor" in line:
                        gpu = line.split()[1].lower()
                        if "intel" in line.lower():
                            result = False, gpu
                            break  # Ensures exit after the first match
                        result = True, gpu
                if result is None:  # If no vendor was found
                    result = False, ""
        elif pe == "tpu":
            if StatusVerifierSingleton.get_platform() == "coral":
                result = True, "tpu0"
            else:
                out = utils.run("lsusb").split("\n")
                for device in out:
                    if ("Global" in device) or ("Google" in device):
                        result = True, "tpu0"
                        break  # Ensures exit after the first match
                if result is None:  # If no device matched
                    result = False, ""
        else:
            log.error("Invalid PE type.")
            result = False, "Invalid PE type."  # Provide a default result for invalid PE

        return result


    def get_active_process(self) -> str:
        """
        getter function for active_process
        """
        return self._active_process


    def set_active_process (self, active_process: str) -> None:
        """
        setter function for active_process
        """
        self._active_process = active_process
