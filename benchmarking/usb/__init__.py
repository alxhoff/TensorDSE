from multiprocessing import Queue
from os import wait
from queue import Empty
from usb.usb import UsbPacket
from usb.timer import UsbTimer
from utils.model import Model
from typing import Dict

START_DEPLOYMENT    = 0
END_DEPLOYMENT      = 1

def analyze(t:UsbTimer) -> Dict:
    d = {}

    d["host_submission"]    = float(t.ts_end_submission - t.ts_begin_submission)
    d["host_communication"] = float(t.ts_end_host_send_request - t.ts_begin_host_send_request)
    d["tpu_communication"]  = float(t.ts_begin_return - t.ts_end_submission)
    d["tpu_return"]         = float(t.ts_end_return - t.ts_end_submission)
    d["inference"]          = float(t.ts_end_return - t.ts_begin_submission)
    d["total"]              = float(t.ts_end_return - t.ts_begin_submission)

    for k in d:
        if d[k] < 0:
            return {}
    return d

def get_tpu_id() -> str:
    bus = "001"
    device = "013"
    return f"{bus}.{device}"

def get_tpu_address(id) -> str:
    return id.split(".")[1]

def capture_stream(model:Model, signalsQ:Queue) -> list:
    """
    """
    import os
    import pyshark

    # initializing usb module
    os.system("modprobe usbmon")

    id      = get_tpu_id()
    addr    = get_tpu_address(id)

    BEGIN               = False
    END                 = False
    TPU_REQUEST_SENT    = False
    SUBMISSION_BEGUN    = False
    HOST_REQUEST_SENT   = False
    RETURN_BEGUN        = False

    FILTER = (
    f"usb.transfer_type==URB_BULK || usb.transfer_type==URB_INTERRUPT && usb.device_address=={addr}"
    )

    def deploy_begun(packet):
        return (
        packet.transfer_type == "INTERRUPT" and
        packet.verify_src("begin")
        )

    def valid_communication(packet):
        return packet.verify_src(id, "all")

    def valid_data(packet):
        return (
        packet.verify_data_validity(raw_packet)
        )

    def present_data(packet):
        return (
        packet.verify_data_presence(raw_packet)
        )

    def sent_from_tpu(packet):
        return (
        packet.id in packet.src
        )

    def sent_from_host(packet):
        return (
        packet.src == "host"
        )

    def deploy_ended(packet):
        return (
        packet.transfer_type == "INTERRUPT" and
        packet.verify_src("begin")
        )

    signalsQ.put(START_DEPLOYMENT)
    capture = pyshark.LiveCapture(interface='usbmon0', display_filter=FILTER)

    timer   = UsbTimer()
    for raw_packet in capture.sniff_continuously():
        try:
            sig = signalsQ.get(False)
            if sig == END_DEPLOYMENT:
                break
        except Exception:
            pass

        packet  = UsbPacket(raw_packet, id)

        if deploy_begun(packet) and not BEGIN:
            timer.stamp_beginning(packet)
            BEGIN = True
            continue

        if deploy_ended(packet) and not END:
            timer.stamp_ending(raw_packet)
            END = True
            break

        if valid_communication(packet) and BEGIN:
            if (packet.transfer_type == "BULK OUT"):

                # Token packets from edge (non-data),
                # describing return
                if (not present_data(packet) and
                        sent_from_tpu(packet) and
                        packet.urb_type == "COMPLETE"):

                    if (not TPU_REQUEST_SENT and
                            not RETURN_BEGUN):
                        timer.stamp_begin_tpu_send_request(raw_packet)
                        TPU_REQUEST_SENT = True
                        continue

                    if (not RETURN_BEGUN):
                        timer.stamp_end_tpu_send_request(raw_packet)
                        continue

                # Data packets from host
                if (present_data(packet) and
                        sent_from_host(packet) and
                        packet.urb_type == "SUBMIT"):

                    if not SUBMISSION_BEGUN:
                        timer.stamp_beginning_submission(raw_packet)
                        SUBMISSION_BEGUN = True
                        continue

                    if (SUBMISSION_BEGUN and
                            valid_data(packet)):
                        timer.stamp_src_host(raw_packet)
                        continue

            if (packet.transfer_type == "BULK IN"):

                # Token packets from host (non-data)
                # asking for data
                if (not present_data(packet) and
                        sent_from_host(packet) and
                        packet.urb_type == "SUBMIT"):

                    # Initial packets of submission of input data
                    # Stamp initial packets
                    if (not HOST_REQUEST_SENT and
                            not SUBMISSION_BEGUN):
                        timer.stamp_begin_host_send_request(raw_packet)
                        HOST_REQUEST_SENT = True
                        continue

                    if not SUBMISSION_BEGUN:
                        timer.stamp_end_host_send_request(raw_packet)
                        continue

                # Data packets from edge
                if (present_data(packet) and
                    sent_from_tpu(packet) and
                    packet.urb_type == "COMPLETE"):

                    # Stamp initial packets
                    if (not RETURN_BEGUN
                            and valid_data(packet)
                            and packet.src != f"{packet.id}.2"):
                        timer.stamp_beginning_return(raw_packet)
                        RETURN_BEGUN = True
                        continue

                    if (valid_data(packet) and
                            RETURN_BEGUN):
                        timer.stamp_src_device(raw_packet)
                        continue

    if END :
        signalsQ.put(analyze(timer))

