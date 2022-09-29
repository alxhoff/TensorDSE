import sys
import os
sys.path[0]=f"{os.getcwd()}/../" # need to overwrite working directory, so imports can work

def capture_stream() -> None:
    """
    """
    import pyshark
    from usb.usb import get_tpu_ids, START_DEPLOYMENT, END_DEPLOYMENT
    from usb.packet import UsbPacket

    id, addr = get_tpu_ids()
    FILTER = (
    f"usb.transfer_type==URB_BULK || usb.transfer_type==URB_INTERRUPT && usb.device_address=={addr}"
    )

    capture = pyshark.LiveCapture(interface='usbmon0', display_filter=FILTER)

    for raw_packet in capture.sniff_continuously():
        p = UsbPacket(raw_packet, id, addr)
        d = raw_packet.usb.data_len
        if int(d) > 0:
            print(d)
        continue

capture_stream()
