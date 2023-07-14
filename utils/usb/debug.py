from multiprocessing import Queue
from typing import List, Tuple

from utils.log import Log

from utils.usb.usb import get_tpu_ids, get_filter, MAX_TIME_CAPTURE
from utils.usb.usb import START_DEPLOYMENT, END_DEPLOYMENT
from utils.usb.packet import UsbPacket
from utils.usb.stream import StreamContext

def color_text(text, fg, bg):
    from termcolor import colored
    return colored(text, fg, bg)

def color_define(p:UsbPacket) -> Tuple[str, str]:
    fg = bg = ""
    if p.transfer_type == "INTERRUPT":
        fg = "red"
        bg = "none"
    elif p.transfer_type == "BULK OUT":
        if p.empty_data:
            fg = "magenta"
            bg = "none"
        else:
            fg = "green"
            bg = "none"
    elif p.transfer_type == "BULK IN":
        if p.empty_data:
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
    from utils.timer import Timer,ConditionalTimer

    initial = start = time.perf_counter()

    id, addr = get_tpu_ids()
    if (not id) or (not addr):
        signalsQ.put(END_DEPLOYMENT)
        return

    packets = []
    context = StreamContext()
    capture = pyshark.LiveCapture(interface='usbmon0', display_filter=get_filter(addr))
    signalsQ.put(START_DEPLOYMENT)

    max_timer = Timer(MAX_TIME_CAPTURE, start_now=True)
    timer = Timer(timeout, start_now=True)
    for raw_packet in capture.sniff_continuously():
        p = UsbPacket(raw_packet, id, addr)

        if timer.reached_timeout():
            context.timed_out()
            break

        elif max_timer.reached_timeout():
            context.maxed_out()
            break

        else:
            if context.stream_valid(p):
                packets.append(p)
                timer.restart()

    show_stream(packets, f"{id}.{addr}")
    dataQ.put(context.conclude())
