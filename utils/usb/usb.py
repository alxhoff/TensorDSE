"""
Missing  Docstring: TODO
"""

from multiprocessing import Queue, Event

import queue
import pyshark

from utils.logging.log import Log

from utils.usb.stream import StreamContext
from utils.usb.packet import UsbPacket

import utils

START_DEPLOYMENT        = 0
END_DEPLOYMENT          = 1

MAX_TIME_CAPTURE=90 # minute and a half

def get_filter(addr:str) -> str:
    """
    Missing  Docstring: TODO
    """
    return (
            #f"usb.transfer_type==URB_BULK || \
            #usb.transfer_type==URB_INTERRUPT && usb.device_address=={addr}"
            f"usb.device_address=={addr}"
            )


def get_tpu_ids():
    """
    Missing  Docstring: TODO
    """

    out = utils.run("lsusb").split("\n")
    line = ""
    for device in out:
        if ("Global" in device) or ("Google" in device):
            line = device
            break
    if not line:
        return "",""

    bus = line.split()[1]
    device = line.split()[3].split(":")[0]

    while device.startswith("0"):
        device = device[-1]

    return bus, device


def peek_queue(q:Queue):
    """
    Missing  Docstring: TODO
    """

    try:
        sig = q.get(False)
        return sig
    except queue.Empty:
        return False


def capture_packets(signals_q:Queue, data_q:Queue, stop_event: Event, usbmon:int, log: Log) -> None:
    """
    Missing  Docstring: TODO
    """

    tpu_id, addr = get_tpu_ids()
    if (not tpu_id) or (not addr):
        signals_q.put(END_DEPLOYMENT)
        return

    context = StreamContext()
    capture = pyshark.LiveCapture(interface=f'usbmon{usbmon}', display_filter=get_filter(addr))


    packet_list = []
    for i, raw_packet in enumerate(capture.sniff_continuously()):
        packet_list.append(UsbPacket(raw_packet, id, addr))
        if stop_event.is_set():
            break

    for i, packet in enumerate(packet_list):
        context.set_phase(packet)
        #print("Packet Nr: {0} | Transfer Type: {1} | Communication Phase: {2}". \
        #format(i, packet.transfer_type, context.current_phase))
        log.info("   - Current Communication phase is: %s -   ", context.current_phase)

        if context.stream_valid(packet):
            if context.contains_host_data(packet):
                log.info("        - Host is communicating Data to Edge TPU -        ")
                context.timestamp_host_data(packet)

            if context.contains_tpu_data(packet):
                log.info("        - Edge TPU is communicating Data to Host -        ")
                context.timestamp_tpu_data(packet)
        else:
            log.error("Packet %s is invalid", i)

    data_q.put(context.conclude())
    log.info("- Packet Capture is complete -")


def capture_stream(signals_q:Queue, data_q:Queue, usbmon:int, log: Log) -> None:
    """
    Missing  Docstring: TODO
    """

    tpu_id, addr = get_tpu_ids()
    if (not tpu_id) or (not addr):
        signals_q.put(END_DEPLOYMENT)
        return

    context = StreamContext()
    capture = pyshark.LiveCapture(interface=f'usbmon{usbmon}', display_filter=get_filter(addr))
    signals_q.put(START_DEPLOYMENT)

    log.info("- Packet Capture is started -")
    for i, raw_packet in enumerate(capture.sniff_continuously()):
        log.info("   - Frame Nr. %s -   ", i)
        p = UsbPacket(raw_packet, id, addr)
        context.set_phase(p)
        log.info("   - Current Communication phase is: %s -   ", context.current_phase)
        log.info("   - Current Transfer Type is: %s -   ", p.transfer_type)

        if context.is_inference_ended():
            break
        else:
            if context.stream_valid(p):
                if context.contains_host_data(p):
                    log.info("        - Host is communicating Data to Edge TPU -        ")
                    context.timestamp_host_data(p)

                if context.contains_tpu_data(p):
                    log.info("        - Edge TPU is communicating Data to Host -        ")
                    context.timestamp_tpu_data(p)

    log.info("- Packet Capture is complete -")
    data_q.put(context.conclude())
