from multiprocessing import Queue

from utils.log import Log

from usb.process import process_timestamps
from usb.timer import UsbTimer
from usb.packet import UsbPacket

START_DEPLOYMENT    = 0
END_DEPLOYMENT      = 1

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

    if device.startswith("0"):
        device = device[1:]

    return bus, device


def peek_queue(q:Queue):
    import queue
    try:
        sig = q.get(False)
        return sig
    except queue.Empty:
        return False

def capture_stream(signalsQ:Queue, dataQ:Queue, l:Log) -> None:
    """
    """
    import pyshark
    id, addr = get_tpu_ids()

    BEGIN               = False
    END                 = False
    TPU_REQUEST_SENT    = False
    SUBMISSION_BEGUN    = False
    HOST_REQUEST_SENT   = False
    RETURN_BEGUN        = False

    FILTER = (
    # f"usb.transfer_type==URB_BULK || usb.transfer_type==URB_INTERRUPT && usb.device_address=={addr}"
    f"(usb.transfer_type==URB_BULK || usb.transfer_type==URB_INTERRUPT) && usb.device_address=={addr}"
    )

    if (not id) or (not addr):
        signalsQ.put(END_DEPLOYMENT)
        return

    timer   = UsbTimer()
    capture = pyshark.LiveCapture(interface='usbmon0', display_filter=FILTER)

    signalsQ.put(START_DEPLOYMENT)
    for raw_packet in capture.sniff_continuously():
        packet  = UsbPacket(raw_packet, id, addr)

        if peek_queue(signalsQ) == END_DEPLOYMENT:
            break

        # BEGIN
        if (packet.transfer_type == "INTERRUPT"
            and packet.is_host_src()
            and packet.is_comms_valid()
            and not BEGIN):
            timer.stamp_beginning(raw_packet)
            BEGIN = True
            l.info("BEGIN PACKET")
            continue

        # END
        if (packet.transfer_type == "INTERRUPT"
            and packet.is_tpu_src()
            and packet.is_comms_valid()
            and BEGIN
            and not END):
            timer.stamp_ending(raw_packet)
            END = True
            l.info("END PACKET")
            break

        # TRAFFIC
        if packet.is_comms_valid() and BEGIN:
            if (packet.transfer_type == "BULK OUT"):

                # Token packets from edge (non-data),
                # describing return
                if (not packet.is_data_present() and
                        packet.is_tpu_src() and
                        packet.urb_type == "COMPLETE"):

                    if (not TPU_REQUEST_SENT
                        and not RETURN_BEGUN):
                        timer.stamp_begin_tpu_send_request(raw_packet)
                        TPU_REQUEST_SENT = True
                        l.info("TPU COMM PACKET BEGIN")
                        continue

                    if (not RETURN_BEGUN):
                        timer.stamp_end_tpu_send_request(raw_packet)
                        l.info("TPU COMM PACKET END")
                        continue

                # Data packets from host
                if (packet.is_data_present()
                    and packet.is_host_src()
                    and packet.urb_type == "SUBMIT"):

                    if not SUBMISSION_BEGUN:
                        timer.stamp_beginning_submission(raw_packet)
                        SUBMISSION_BEGUN = True
                        l.info("SUBMISSION PACKET")
                        continue

                    if (SUBMISSION_BEGUN
                        and packet.is_data_valid()):
                        timer.stamp_src_host(raw_packet)
                        l.info("HOST ACK PACKET")
                        continue

            if (packet.transfer_type == "BULK IN"):

                # Token packets from host (non-data)
                # asking for data
                if (not packet.is_data_present() and
                        packet.is_host_src() and
                        packet.urb_type == "SUBMIT"):

                    # Initial packets of submission of input data
                    # Stamp initial packets
                    if (not HOST_REQUEST_SENT and
                            not SUBMISSION_BEGUN):
                        timer.stamp_begin_host_send_request(raw_packet)
                        HOST_REQUEST_SENT = True
                        l.info("HOST REQUEST PACKET")
                        continue

                    if not SUBMISSION_BEGUN:
                        timer.stamp_end_host_send_request(raw_packet)
                        l.info("HOST REQUEST PACKET END")
                        continue

                # Data packets from edge
                if (packet.is_data_present() and
                    packet.is_tpu_src() and
                    packet.urb_type == "COMPLETE"):

                    # Stamp initial packets
                    if (not RETURN_BEGUN
                            and packet.is_data_valid()):
                        timer.stamp_beginning_return(raw_packet)
                        RETURN_BEGUN = True
                        l.info("RETURN PACKET BEGIN")
                        continue

                    if (packet.is_data_valid() and
                            RETURN_BEGUN):
                        timer.stamp_src_device(raw_packet)
                        l.info("RETURN PACKET END")
                        continue

    if END :
        dataQ.put(process_timestamps(timer))
    else:
        dataQ.put({})
    l.info("RETURN")
    return

