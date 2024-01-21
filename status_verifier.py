"""
Module Docstring: TODO
"""

import argparse

from typing import Tuple

import utils

from utils.logging.logger import log
from utils.splitter.utils import ReadJSON


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

    def __init__(self):
        self._platform = None
        self._multi_model_summary = None
        self._hardware_summary = None
        self._requested_hardware = None
        self._active_process = None


    def verify_hardware_for_profiling(self, hardware_summary: dict) -> bool:
        """
        Verifies and identifies the hardware used to run the profiling
        """

        req_hw = []
        if int(hardware_summary["CPU_count"]) > 0:
            pe_vailable = self.verify_pe("cpu")[0]
            if pe_vailable:
                req_hw.append("cpu")
            else:
                log.fatal("The CPU requested for Profiling is not available. Abort")

        if int(hardware_summary["GPU_count"]) > 0:
            pe_vailable= self.verify_pe("gpu")[0]
            if pe_vailable:
                req_hw.append("gpu")
            else:
                log.fatal("The GPU requested for Profiling is not available. Abort")

        if int(hardware_summary["TPU_count"]) > 0:
            pe_vailable = self.verify_pe("tpu")[0]
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
            model_summary_json = ReadJSON(args.summarypath)
            if model_summary_json is None:
                log.fatal("The provided Model Summary is empty!")
                return False
        else:
            log.fatal("The provided path to the Model Summary is not valid!")
            return False

        if args.hardwaresummary is not None:
            hardware_summary_json = ReadJSON(args.hardwaresummary)
            if hardware_summary_json is None:
                log.fatal("The provided Hardware Summary is empty!")
                return False
        else:
            log.fatal("The provided path to the Hardware Summary is not valid!")
            return False

        self._multi_model_summary = model_summary_json
        self._hardware_summary = hardware_summary_json
        self._active_process = "Profiling"

        return True


    def verify_args_for_deployer(self, args: argparse.Namespace) -> bool:
        """
        Verifies the arguments given to the deploy.py script 
        """
        if args.summarypath is not None:
            summary = ReadJSON(args.summarypath)
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


    def get_platform(self) -> str:
        """
        getter function for platform
        """
        return self._platform


    def set_platform(self, platform: str) -> None:
        """
        setter function for platform
        """
        self._platform = platform


    def verify_pe(self, pe: str) -> Tuple[bool, str]:
        """
        Verifies if a specific PE is available or not.
        """
        result = None
        match pe:
            case "cpu":
                result =  True, "cpu0"
            case "gpu":
                if self.get_platform() != "desktop":
                    result =  True, ""
                out = utils.run("lshw -numeric -C display").split("\n")
                for line in out:
                    if "vendor" in line:
                        gpu = line.split()[1].lower()
                        if "intel" in line.lower():
                            result =  False, gpu
                        result =  True, gpu
                result =  False, ""
            case "tpu":
                if self.get_platform() == "coral":
                    result =  True, ""
                out = utils.run("lsusb").split("\n")
                for device in out:
                    if ("Global" in device) or ("Google" in device):
                        result =  True
                result =  False, ""
            case _:
                log.error("Invalid PE type.")
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
