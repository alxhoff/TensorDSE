from multiprocessing import Queue
from typing import List, Tuple

from utils.log import Log

from usb.usb import get_tpu_ids, get_filter, MAX_TIME_CAPTURE
from usb.usb import START_DEPLOYMENT, END_DEPLOYMENT, SUCCESSFULL_DEPLOYMENT, ERRONEOUS_DEPLOYMENT
from usb.packet import UsbPacket
from usb.stream import StreamContext

def color_text(text, fg, bg):
    from termcolor import colored
    return colored(text, fg, bg)

def color_define(p:UsbPacket) -> Tuple[str, str]:
    fg = bg = ""
    if p.transfer_type == "INTERRUPT":
        fg = "red"
        bg = "none"
    elif p.transfer_type == "BULK OUT":
        if p.empty_data and p.present_data:
            fg = "magenta"
            bg = "none"
        else:
            fg = "green"
            bg = "none"
    elif p.transfer_type == "BULK IN":
        if p.empty_data and p.present_data:
            fg = "magenta"
            bg = "none"
        else:
            fg = "blue"
            bg = "none"
    else:
        fg = "yellow"
        bg = "none"

    return fg, bg

def show_stream(packets:List[UsbPacket], identification:str) -> None:
    from tabulate import tabulate
    hdr = [
        "Nr", f"Direction (host,{identification})", "Type", "URB Type", "TS", "Present Data", "Data Size", "Valid Comm."
    ]
    table = []

    for i,p in enumerate(packets):
        fg, bg = color_define(p)
        bg = None if bg == "none" else bg

        tb = [
            color_text(f"{i + 1}/{len(packets)}", fg, bg),
            color_text(f"[{p.direction}]{p.src} -> {p.dest}", fg, bg),
            color_text(f"{p.transfer_type}", fg, bg),
            color_text(f"{p.urb_type}", fg, bg),
            color_text(f"{p.ts:.2f}", fg, bg),
            color_text(f"{p.present_data}", fg, bg),
            color_text(f"{p.data_size}", fg, bg),
            color_text(f"{p.valid_communication}", fg, bg),
        ]
        table.append(tb)

    print(tabulate(table, headers=hdr))
    return

def capture_stream(signalsQ:Queue, dataQ:Queue, timeout:int, l:Log) -> None:
    """
    """
    import pyshark
    import time

    initial = start = time.perf_counter()
    context = StreamContext()
    packets = []

    id, addr = get_tpu_ids()
    FILTER = get_filter(addr)
    if (not id) or (not addr):
        signalsQ.put(END_DEPLOYMENT)
        return

    capture = pyshark.LiveCapture(interface='usbmon0', display_filter=FILTER)
    signalsQ.put(START_DEPLOYMENT)

    for raw_packet in capture.sniff_continuously():
        print("PACKET")
        p = UsbPacket(raw_packet, id, addr)
        diff = (time.perf_counter() - start)
        total_diff = (time.perf_counter() - initial)

        if diff >= timeout:
            print("TIME")
            break
        elif total_diff >= MAX_TIME_CAPTURE:
            print("MAX TIME")
            break
        else:
            if context.stream_valid(p):
                packets.append(p)
                if context.stream_started(p):
                    pass

    show_stream(packets, f"{id}.{addr}")
    signalsQ.put(ERRONEOUS_DEPLOYMENT)
    dataQ.put({})


