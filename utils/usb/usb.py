from multiprocessing import Queue, Event

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


def capture_packets(signalsQ:Queue, dataQ:Queue, stopEvent: Event, l:Log, usbmon:int) -> None:
    import pyshark

    id, addr = get_tpu_ids()
    if (not id) or (not addr):
        signalsQ.put(END_DEPLOYMENT)
        return
    
    context = StreamContext()
    capture = pyshark.LiveCapture(interface='usbmon{}'.format(usbmon), display_filter=get_filter(addr))


    packet_list = []
    for i, raw_packet in enumerate(capture.sniff_continuously()):
        packet_list.append(UsbPacket(raw_packet, id, addr))
        if stopEvent.is_set():
            break

    for i, packet in enumerate(packet_list):
        context.set_phase(packet)
        print("Packet Nr: {0} | Transfer Type: {1} | Communication Phase: {2}".format(i, packet.transfer_type, context.current_phase))
        l.info("   - Current Communication phase is: {} -   ".format(context.current_phase))

        #if (context.is_inference_ended()):
        #    break
        #else:
        if context.stream_valid(packet):
            if context.contains_host_data(packet):
                l.info("        - Host is communicating Data to Edge TPU -        ")
                context.timestamp_host_data(packet)

            if context.contains_tpu_data(packet):
                l.info("        - Edge TPU is communicating Data to Host -        ")
                context.timestamp_tpu_data(packet)
        else:
            print(f"Packet {i} is invalid")

    dataQ.put(context.conclude())
    l.info("- Packet Capture is complete -")


def capture_stream(signalsQ:Queue, dataQ:Queue, timeout:int, l:Log, usbmon:int) -> None:
    """
    """
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
        l.info("   - Current Transfer Type is: {} -   ".format(p.transfer_type))

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
