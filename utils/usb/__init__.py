"""
Missing  Docstring: TODO
"""

import os
from utils.logging.logger import log

def init_usbmon(usb_bus: int) -> bool:
    """
    Missing  Docstring: TODO
    """
    dirs = os.listdir("/dev/")
    if f"usbmon{usb_bus}" in dirs:
        return False
    log.error("usbmon module has to be loaded!!!")
    return True
