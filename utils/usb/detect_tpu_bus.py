#!/usr/bin/python3
import sys

def detect() -> int:
    import subprocess, re

    try:
        bus = re.findall('\/sys\/bus\/usb\/devices\/([0-9]+)-[0-9]+\/idVendor:.+', 
                subprocess.check_output('grep 18d1 /sys/bus/usb/devices/*/idVendor', shell=True).decode('utf-8'))[0]
    except Exception:
        bus = re.findall('\/sys\/bus\/usb\/devices\/([0-9]+)-[0-9]+\/idVendor:.+', 
            subprocess.check_output('grep 1a6e /sys/bus/usb/devices/*/idVendor', shell=True).decode('utf-8'))[0]
    return bus

if __name__ == "__main__":
    sys.exit(detect())

