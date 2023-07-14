from utils.usb.usb import capture_stream, START_DEPLOYMENT, END_DEPLOYMENT

import sys
#from main import log
from ..model_lab.logger import log

def init_usbmon() -> bool:
    import os
    dirs = os.listdir("/dev/")
    if "usbmon0" in dirs:
        return False

    log.error("usbmon module has to be loaded!!!")
    print("usbmon module has to be loaded!!!")
    sys.exit(1)
