from multiprocessing import Queue
from typing import List, Tuple

from usb.usb import get_tpu_ids, START_DEPLOYMENT, END_DEPLOYMENT
from usb.packet import UsbPacket

def color_text(text, fg, bg):
    from termcolor import colored
    return colored(text, fg, bg)

def color_define(p:UsbPacket) -> Tuple[str, str]:
    fg = bg = ""
    if p.transfer_type == "INTERRUPT":
        fg = "red"
        bg = "none"
    elif p.transfer_type == "BULK OUT":
        fg = "green"
        bg = "none"
    elif p.transfer_type == "BULK IN":
        fg = "blue"
        bg = "none"
    else:
        fg = "yellow"
        bg = "none"

    return fg, bg

def show_stream(packets:List[UsbPacket], identification:str) -> None:
    from tabulate import tabulate
    header = [
        ["Nr", f"Direction (host,{identification})", "Type", "URB Type", "TS", "Present Data", "Data Size", "Valid Comm."]
    ]
    table = []

    for i,p in enumerate(packets):
        fg, bg = color_define(p)
        bg = None if bg == "none" else bg

        tb = [
            color_text(f"{i + 1}", fg, bg),
            color_text(f"[{p.direction}]{p.src} -> {p.dest}", fg, bg),
            color_text(f"{p.transfer_type}", fg, bg),
            color_text(f"{p.urb_type}", fg, bg),
            color_text(f"{p.ts:.2f}", fg, bg),
            color_text(f"{p.present_data}", fg, bg),
            color_text(f"{p.data_size}", fg, bg),
            color_text(f"{p.valid_comms}", fg, bg),
        ]
        table.append(tb)

    print("USB PACKETS")
    print(tabulate(header))
    print(tabulate(table))
    return

def capture_stream(signalsQ:Queue, dataQ:Queue) -> None:
    """
    """
    import pyshark
    import queue
    id, addr = get_tpu_ids()

    FILTER = (
    f"usb.transfer_type==URB_BULK || usb.transfer_type==URB_INTERRUPT && usb.device_address=={addr}"
    )

    if (not id) or (not addr):
        signalsQ.put(END_DEPLOYMENT)
        return

    packets = []
    capture = pyshark.LiveCapture(interface='usbmon0', display_filter=FILTER)

    signalsQ.put(START_DEPLOYMENT)
    for raw_packet in capture.sniff_continuously():
        p  = UsbPacket(raw_packet, id, addr)
        packets.append(p)

        try:
            sig = signalsQ.get(False)
        except queue.Empty:
            sig = False
            pass

        if sig == END_DEPLOYMENT:
            break

    show_stream(packets, f"{id}.{addr}")
    dataQ.put({})


