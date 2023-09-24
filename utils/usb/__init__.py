from utils.usb.usb import START_DEPLOYMENT, END_DEPLOYMENT

import sys
from utils.logging.logger import log

def init_usbmon(usb_bus: int) -> bool:
    import os
    dirs = os.listdir("/dev/")
    if "usbmon{}".format(usb_bus) in dirs:
        return False
    else:
        log.error("usbmon module has to be loaded!!!")
        return True
