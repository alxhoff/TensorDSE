from multiprocessing import Queue

from utils.log import Log

from usb.stream import StreamContext
from usb.packet import UsbPacket

START_DEPLOYMENT        = 0
END_DEPLOYMENT          = 1

MAX_TIME_CAPTURE=90 # minute and a half

def get_filter(addr:str) -> str:
    return (
    f"usb.transfer_type==URB_BULK || usb.transfer_type==URB_INTERRUPT && usb.device_address=={addr}"
    )

def get_tpu_ids():
    import utils
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
        device = device[1:]

    return bus, device

def peek_queue(q:Queue):
    import queue
    try:
        sig = q.get(False)
        return sig
    except queue.Empty:
        return False

def capture_stream(signalsQ:Queue, dataQ:Queue, timeout:int, l:Log) -> None:
    """
    """
    from utils.timer import Timer,ConditionalTimer
    import pyshark

    id, addr = get_tpu_ids()
    if (not id) or (not addr):
        signalsQ.put(END_DEPLOYMENT)
        return

    context = StreamContext()
    capture = pyshark.LiveCapture(interface='usbmon0', display_filter=get_filter(addr))
    signalsQ.put(START_DEPLOYMENT)

    timer = Timer(MAX_TIME_CAPTURE, start_now=True)
    ctimer = ConditionalTimer(timeout)
    for raw_packet in capture.sniff_continuously():
        p = UsbPacket(raw_packet, id, addr)

        if ctimer.reached_timeout():
            context.timed_out()
            break

        elif timer.reached_timeout():
            context.maxed_out()
            break

        else:
            if context.stream_valid(p):
                if context.has_data_trafficked(): # only returns true once at most
                    ctimer.set_conditional_flag()
                    ctimer.start()

                if context.contains_host_data(p):
                    context.timestamp_host_data(p)
                    continue

                if context.contains_tpu_data(p):
                    context.timestamp_tpu_data(p)
                    ctimer.restart()
                    continue

    dataQ.put(context.conclude())
