from multiprocessing import Queue

from utils.log import Log

from utils.usb.stream import StreamContext
from utils.usb.packet import UsbPacket

START_DEPLOYMENT        = 0
END_DEPLOYMENT          = 1

MAX_TIME_CAPTURE=90 # minute and a half

def get_filter(addr:str) -> str:
    return (
            #f"usb.transfer_type==URB_BULK || usb.transfer_type==URB_INTERRUPT && usb.device_address=={addr}"
            f"usb.device_address=={addr}"
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
        device = device[-1]

    return bus, device

def peek_queue(q:Queue):
    import queue
    try:
        sig = q.get(False)
        return sig
    except queue.Empty:
        return False

def capture_stream(signalsQ:Queue, dataQ:Queue, timeout:int, l:Log, usbmon:int) -> None:
    """
    """
    from utils.timer import Timer,ConditionalTimer
    from .detect_tpu_bus import detect
    import pyshark

    id, addr = get_tpu_ids()
    if (not id) or (not addr):
        signalsQ.put(END_DEPLOYMENT)
        return

    context = StreamContext()
    capture = pyshark.LiveCapture(interface='usbmon{}'.format(usbmon), display_filter=get_filter(addr))
    signalsQ.put(START_DEPLOYMENT)

    l.info("- Packet Capture is started -")
    for i, raw_packet in enumerate(capture.sniff_continuously()):
        l.info("   - Frame Nr. {} -   ".format(i))
        p = UsbPacket(raw_packet, id, addr)
        context.set_phase(p)
        l.info("   - Current Communication phase is: {} -   ".format(context.current_phase))

        if (context.is_inference_ended()):
            break

        else:
            if context.stream_valid(p):
                if context.contains_host_data(p):
                    l.info("        - Host is communicating Data to Edge TPU -        ")
                    context.timestamp_host_data(p)

                if context.contains_tpu_data(p):
                    l.info("        - Edge TPU is communicating Data to Host -        ")
                    context.timestamp_tpu_data(p)

    l.info("- Packet Capture is complete -")
    dataQ.put(context.conclude())
