from usb.usb import capture_stream, START_DEPLOYMENT, END_DEPLOYMENT

def init_usbmon() -> bool:
    import os
    dirs = os.listdir("/dev/")
    if "usbmon0" in dirs:
        return False

    os.system("modprobe usbmon")
    return True
