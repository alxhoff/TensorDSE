from utils.usb.usb import capture_stream, START_DEPLOYMENT, END_DEPLOYMENT

import sys
from .detect_tpu_bus import detect
#from main import log
from ..splitter.logger import log

def init_usbmon() -> bool:
    import os
    dirs = os.listdir("/dev/")
    #interface_index = detect()
    #interface = f"usbmon{interface_index}"
    if "usbmon0" in dirs:
        return False

    log.error("usbmon module has to be loaded!!!")
    print("usbmon module has to be loaded!!!")
    sys.exit(1)
