#!/usr/bin/python3
import sys

def main() -> int:
    import subprocess, re

    return re.findall('\/sys\/bus\/usb\/devices\/([0-9]+)-[0-9]+\/idVendor:.+', 
            subprocess.check_output('grep 18d1 /sys/bus/usb/devices/*/idVendor', shell=True).decode('utf-8'))[0]

if __name__ == "__main__":
    sys.exit(main())

