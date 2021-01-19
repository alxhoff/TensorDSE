import threading

event = threading.Event()

edge_tpu_id = ""

count = 0


class UsbPacket:
    def __init__(self):
        ts_absolute_begin = 0
        ts_begin_inference = 0
        ts_begin_return = 0
        ts_absolute_end = 0

    def create_copy(self, packet):
        import copy
        self.raw_packet = copy.deepcopy(packet)

    def find_direction(self, packet):
        self.direction = packet.usb.endpoint_address_direction

    def find_scr_dest(self, packet):
        self.src = packet.usb.src
        self.dest = packet.usb.dst

    def find_transfer_type(self, hexa_transfer_type):
        default = None
        transfer_dict = {
                "0x00000001"    :   "INTERRUPT",
                "0x00000002"    :   "CONTROL",
                "0x00000003"    :   self.find_bulk_type(),
                ""              :   None
                }

        self.transfer_type = transfer_dict.get(hexa_transfer_type, default)

    def find_bulk_type(self):
        if self.direction == '1':
            return "BULK IN"
        elif self.direction == '0':
            return "BULK OUT"
        else:
            raise ValueError("Wrong attribute type in packet.endpoint_address_direction.")

    def stamp_beginning(self, packet):
        self.ts_absolute_begin = packet.frame_info.time_relative

def prep_capture_file():
    import os
    cap_file = "/home/duclos/Documents/work/TensorDSE/shark/capture.pcap"
    os.system("[ -f " + cap_file + " ] || touch " + cap_file)


def shark_deploy_edge(count, folder):
    import os
    import utils
    from deploy import deduce_operations_from_folder
    import docker
    from docker import TO_DOCKER, FROM_DOCKER, home

    path_to_tensorDSE = utils.retrieve_folder_path(os.getcwd(), "TensorDSE")
    docker.docker_copy(path_to_tensorDSE, TO_DOCKER)
    docker.set_globals(count)

    models_info = deduce_operations_from_folder(folder,
                                                beginning="quant_",
                                                ending="_edgetpu.tflite")

    for m_i in models_info:
        docker.docker_exec("shark_single_edge_deploy", m_i[0])

        ans = input("Continue [y/n]?")
        if ans != "y":
            break


def count_packets(cap):
    lngth = 0
    try:
        for c in cap:
            lngth += 1
    except:
        pass

    return lngth


def lsusb_identify():
    import os

    global edge_tpu_id
    print("IDing usb entry...")

    lsusb_cmd = "lsusb | grep Google > temp.txt"
    os.system(lsusb_cmd)
    with open("temp.txt", "r") as f:
        for line in f.readlines():
            if "Google" in line:
                bus = int(line.split()[1])
                device = int((line.split()[3]).split(":")[0])
                edge_tpu_id = str(bus) + "." + str(device)

    os.system("[ -f temp.txt ] && rm temp.txt")


def shark_usbmon_init():
    import os
    usbmon_cmd = "sudo modprobe usbmon"
    os.system(usbmon_cmd)


def shark_capture_cont():
    import os
    import asyncio
    import pyshark

    global event

    usb_array = []

    beginning_of_comms = False
    end_of_comms = False
    end_of_capture = False

    capture_filter = "usb.transfer_type==URB_BULK || usb.transfer_type==URB_INTERRUPT"
    capture = pyshark.LiveCapture(interface='usbmon0', display_filter=capture_filter)

    for packet in capture.sniff_continuously():
        usb_packet = UsbPacket()
        usb_packet.find_direction(packet)
        usb_packet.find_transfer_type(packet.usb.transfer_type)
        usb_packet.find_scr_dest(packet)

        if (usb_packet.transfer_type == "INTERRUPT"): # Check for beginning of capture
            beginning_of_comms = True
            usb_packet.stamp_beginning(packet)
            usb_packet.create_copy(packet)
            usb_array.append(usb_packet)

        elif (usb_packet.transfer_type == "BULK IN"):
            usb_packet.create_copy(packet)
            usb_array.append(usb_packet)
            
            if packet.usb.data_flag != '<':
                pass

        elif (usb_packet.transfer_type == "BULK OUT"):
            usb_packet.create_copy(packet)
            usb_array.append(usb_packet)

            if packet.usb.data_flag != '<':
                pass

        else:
            pass

        if (not event.is_set() and end_of_capture):
            event.set()
            break


def shark_capture_init(timeout, op_name):
    """
    Function responsible for initializing the capture/listen socket on the usb
    interface.
    """
    import os
    import asyncio
    import pyshark

    prep_capture_file()
    capture_filter = "usb.transfer_type==URB_BULK"
    out_file = "shark/" + op_name + "capture.pcap"

    param_dict = {
        "--param": "value",
        "--param": "value"
    }

    capture = pyshark.LiveCapture(interface='usbmon0', output_file=out_file)

    try:
        capture.sniff(timeout=timeout)

    except asyncio.exceptions.TimeoutError:  # Pyshark is a bit broken.
        pass

    try:
        capture.close()

    except RuntimeError:
        pass
    except pyshark.capture.capture.TSharkCrashException:
        pass


def shark_read_captures():
    import os
    import pyshark
    from os import listdir
    from deploy import deduce_operation_from_file

    capture_dir = "shark/"
    read_filter = "usb.transfer_type==URB_BULK"

    for capture_file in listdir(capture_dir):

        usb_timer = UsbPacket()
        cap_file_path = capture_dir + capture_file
        op = deduce_operation_from_file(capture_file, ending="_capture.pcap")

        capture = pyshark.FileCapture(input_file=cap_file_path,
                                      display_filter=read_filter)

        host_first = 0
        edge_first = 0

        try:
            for pkt in capture:
                if ("host" in pkt.usb.src or edge_tpu_id in pkt.usb.src):
                    if ("host" in pkt.usb.src):
                        if (host_first == 0):
                            usb_timer.ts_absolute_begin = pkt.frame_info.time_relative
                            host_first = 1

                        if (edge_first == 0):
                            usb_timer.ts_begin_inference = pkt.frame_info.time_relative

                    if (edge_tpu_id in pkt.usb.src):
                        if (edge_first == 0):
                            usb_timer.ts_begin_return = pkt.frame_info.time_relative
                            edge_first = 1

                    usb_timer.ts_absolute_end = pkt.frame_info.time_relative

        except pyshark.capture.capture.TSharkCrashException:
            pass

        try:
            capture.close()

        except RuntimeError:
            pass
        except pyshark.capture.capture.TSharkCrashException:
            pass

        os.system("pgrep tshark && killall -9 tshark")
        export_analysis(usb_timer, op)


def shark_manager(folder):
    import os
    import time
    import threading
    import pyshark
    from docker import TO_DOCKER, FROM_DOCKER, home, docker_exec, docker_copy
    from deploy import deduce_operations_from_folder
    from utils import retrieve_folder_path

    global check

    path_to_tensorDSE = retrieve_folder_path(os.getcwd(), "TensorDSE")
    docker_copy(path_to_tensorDSE, TO_DOCKER)

    models_info = deduce_operations_from_folder(folder,
                                                beginning="quant_",
                                                ending="_edgetpu.tflite")

    for m_i in models_info:
        t_1 = threading.Thread(target=shark_capture_cont,
                              args=())

        t_2 = threading.Thread(target=docker_exec, args=(
            "shark_single_edge_deploy", m_i[0],))

        t_1.start()
        t_2.start()

        event.wait()
        event.clear()

        t_1.join()
        t_2.join()

        print("Ended capture.")

        time.sleep(1)

    # shark_read_captures()


def export_analysis(usb_timer, op):
    import csv
    from utils import extend_directory

    out_dir = "results/shark/"
    extend_directory(out_dir, op)
    extended_dir = out_dir + op
    results_file = extended_dir + "/Results.csv"

    with open(results_file, 'w') as csvfile:
        fw = csv.writer(csvfile, delimiter=',', quotechar='|',
                        quoting=csv.QUOTE_MINIMAL)

        fw.writerow(["begin", "inference", "return", "end"])
        fw.writerow([usb_timer.ts_absolute_begin, usb_timer.ts_begin_inference,
                     usb_timer.ts_begin_return, usb_timer.ts_absolute_end])


if __name__ == '__main__':
    from docker import docker_start
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-c', '--count', required=False,
                        default=1000,
                        help='Count of the number of times of edge deployment.')

    parser.add_argument('-t', '--timeout', required=False, type=int,
                        default=60,
                        help='Timeout used to enforce time employed on capture/listening of usb packets.')

    parser.add_argument('-m', '--mode', required=False,
                        default="Both",
                        help='Mode in which the script will run: All, Read, Capture or Deploy.')

    parser.add_argument('-f', '--folder', required=False,
                        default="",
                        help='Folder.')

    args = parser.parse_args()

    if (args.mode == "Capture"):
        shark_usbmon_init()
        shark_capture_init(args.timeout, "TEST")

    elif(args.mode == "Deploy"):
        docker_start()
        shark_deploy_edge(args.count, args.folder)

    elif (args.mode == "Read"):
        lsusb_identify()
        shark_read_captures()

    elif (args.mode == "All" and args.folder != ""):
        shark_usbmon_init()
        lsusb_identify()
        docker_start()
        shark_manager(args.folder)

        print(f"Count: {count}")

    else:
        print("Invaild arguments.")
