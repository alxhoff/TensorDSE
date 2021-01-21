import threading

event = threading.Event()

edge_tpu_id = ""

usb_array = []

class UsbTimer:
    def __init__(self):
        self.ts_absolute_begin = 0

        # Last CPU sent request before actual data was sent to device(TPU).
        self.ts_end_host_send_request = 0 

        # Last CPU sent of actual data and where the TPU begins to compute the result.
        self.ts_begin_inference = 0

        # Last TPU sent request before actual data was sent to the CPU.
        self.ts_end_tpu_send_request = 0

        # First Data present packet sent from device(TPU) to CPU.
        self.ts_end_inference = 0

        # Receiving Data
        self.ts_absolute_end = 0


    def print_stamps(self):
        print(f"ABSOLUTE BEGINNING/BEGINNING OF REQUESTS (HOST): {self.ts_absolute_begin}")
        print(f"END OF REQUESTS (HOST): {self.ts_end_host_send_request}")
        print(f"END OF HOST SENT DATA: {self.ts_end_submission}")
        print(f"INFERENCE BEGIN/LAST DATA FROM HOST SENT/BEGIN OF REQUESTS (TPU): {self.ts_begin_inference}")
        print(f"END OF REQUESTS (TPU)/BEGIN OF TPU SENT DATA: {self.ts_end_tpu_send_request}")
        print(f"ABSOLUTE END: {self.ts_absolute_end}")

    def stamp_beginning(self, packet):
        self.ts_absolute_begin = packet.frame_info.time_relative

    def stamp_ending(self, packet):
        self.ts_absolute_end = packet.frame_info.time_relative

    def stamp_inference(self, packet):
        self.ts_begin_inference = self.ts_end_submission
        self.ts_end_inference = float(packet.frame_info.time_relative) - float(self.ts_begin_inference)

    def stamp_end_host_send_request(self, packet):
        self.ts_end_host_send_request = packet.frame_info.time_relative
    
    def stamp_beginning_submission(self, packet):
        self.ts_begin_submission = packet.frame_info.time_relative

    def stamp_beginning_return(self, packet):
        self.ts_begin_return = packet.frame_info.time_relative

    def stamp_end_tpu_send_request(self, packet):
        self.ts_end_tpu_send_request = packet.frame_info.time_relative

    def stamp_src_host(self, packet):
        self.ts_end_submission = packet.frame_info.time_relative

    def stamp_src_device(self, packet):
        self.ts_end_return = packet.frame_info.time_relative

class UsbPacket:
    def __init__(self):
        pass

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

    def find_urb_type(self, urb_type):
        default = None
        transfer_dict = {
                "'S'"    :   "SUBMIT",
                "'C'"    :   "COMPLETE",
                ""              :   None
                }

        self.urb_type = transfer_dict.get(urb_type, default)

    def find_bulk_type(self):
        if self.direction == '1':
            return "BULK IN"
        elif self.direction == '0':
            return "BULK OUT"
        else:
            raise ValueError("Wrong attribute type in packet.endpoint_address_direction.")

    def find_data_presence(self, packet):
        tmp = packet.usb.data_flag
        if tmp == '>' or tmp == '<':
            return False
        elif tmp == 'present (0)':
            return True
        else:
            return False

    def verify_src(self):
        global edge_tpu_id
        return True if ("host" == self.src and edge_tpu_id in self.dest) else False


def export_analysis(usb_timer, op, append):
    import csv
    from utils import extend_directory

    results_file = f"results/shark/{op}/Results.csv"

    if (append == True):
        with open(results_file, 'a+') as csvfile:
            fw = csv.writer(csvfile, delimiter=',', quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)

            fw.writerow([usb_timer.ts_absolute_begin, usb_timer.ts_end_host_send_request, 
                         usb_timer.ts_end_submission, usb_timer.ts_begin_inference,
                         usb_timer.ts_end_tpu_send_request, usb_timer.ts_absolute_end])
    else:
        with open(results_file, 'w') as csvfile:
            fw = csv.writer(csvfile, delimiter=',', quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)

            fw.writerow(["absolute_begin", "end_requests_host", 
                        "end_host_submissions", "inference_begin", 
                        "end_requests_tpu", "absolute_end"])

            fw.writerow([usb_timer.ts_absolute_begin, usb_timer.ts_end_host_send_request, 
                         usb_timer.ts_end_submission, usb_timer.ts_begin_inference,
                         usb_timer.ts_end_tpu_send_request, usb_timer.ts_absolute_end])


def prep_capture_file():
    import os
    cap_file = "/home/duclos/Documents/work/TensorDSE/shark/capture.pcap"
    os.system("[ -f " + cap_file + " ] || touch " + cap_file)


def lsusb_identify():
    import os

    print("IDing usb entry...")

    lsusb_cmd = "lsusb | grep Google > temp.txt"
    os.system(lsusb_cmd)
    with open("temp.txt", "r") as f:
        for line in f.readlines():
            if "Google" in line:
                global edge_tpu_id

                bus = int(line.split()[1])
                device = int((line.split()[3]).split(":")[0])
                edge_tpu_id = str(bus) + "." + str(device)

    os.system("[ -f temp.txt ] && rm temp.txt")


def shark_usbmon_init():
    import os
    usbmon_cmd = "sudo modprobe usbmon"
    os.system(usbmon_cmd)


def shark_capture_cont(op, cnt):
    import os
    import pyshark

    global event
    global usb_array

    usb_timer = UsbTimer()

    beginning_of_comms = False
    beginning_of_submission = False
    beginning_of_return = False

    end_host_send_request = False
    end_tpu_send_request = False

    end_of_capture = False
    end_of_comms = False

    capture_filter = "usb.transfer_type==URB_BULK || usb.transfer_type==URB_INTERRUPT"
    capture = pyshark.LiveCapture(interface='usbmon0', display_filter=capture_filter)

    for packet in capture.sniff_continuously():
        usb_packet = UsbPacket()
        usb_packet.find_direction(packet)
        usb_packet.find_transfer_type(packet.usb.transfer_type)
        usb_packet.find_scr_dest(packet)
        usb_packet.find_urb_type(packet.usb.urb_type)

        data_is_present = usb_packet.find_data_presence(packet)

        # Checks for beginning and ending of captures, as interrupts appear at these point.
        if (usb_packet.transfer_type == "INTERRUPT"):
            usb_array.append(usb_packet)

            if (beginning_of_comms == False and usb_packet.verify_src()):
                usb_timer.stamp_beginning(packet)
                beginning_of_comms = True

            elif (end_of_comms == False and beginning_of_comms == True):
                end_of_comms = True
                end_of_capture = True
                usb_timer.stamp_ending(packet)
            else:
                pass

        elif ((usb_packet.transfer_type == "BULK IN" 
               or usb_packet.transfer_type == "BULK OUT")):

            if (usb_packet.urb_type == "SUBMIT" and data_is_present == False):
                assert (usb_packet.src == "host"), "Submit packets should only be from host!"

                if (end_host_send_request == False):
                    usb_timer.stamp_end_host_send_request(packet)

            elif (usb_packet.urb_type == "SUBMIT" and data_is_present == True):
                assert (usb_packet.src == "host"), "Submit packets should only be from host!"
                end_host_send_request = True

                if beginning_of_submission == False:
                    usb_timer.stamp_beginning_submission(packet)
                    beginning_of_submission = True

                usb_timer.stamp_src_host(packet)
                usb_array.append(usb_packet)

            elif usb_packet.urb_type == "COMPLETE" and data_is_present == False:
                assert (usb_packet.src != "host"), "Complete packets should only be from device!"
                end_tpu_send_request = True

                if (end_tpu_send_request == False):
                    usb_timer.stamp_end_tpu_send_request(packet)
                    end_tpu_send_request = True

                usb_timer.stamp_end_tpu_send_request(packet)
                usb_array.append(usb_packet)

            elif (usb_packet.urb_type == "COMPLETE" and data_is_present == True):
                assert (usb_packet.src != "host"), "Complete packets should only be from device!"

                if beginning_of_return == False:
                    usb_timer.stamp_beginning_return(packet)
                    usb_timer.stamp_inference(packet)
                    beginning_of_return = True

                usb_timer.stamp_src_device(packet)
                usb_array.append(usb_packet)

            else:
                pass

        else:
            pass

        if (not event.is_set() and end_of_capture == True and end_of_comms == True):
            # usb_timer.print_stamps()
            export_analysis(usb_timer, op, cnt!=0)
            event.set()
            break


def shark_manager(folder):
    import os
    import time
    import threading
    import pyshark
    from docker import TO_DOCKER, FROM_DOCKER, home, docker_exec, docker_copy
    from deploy import deduce_operations_from_folder
    from utils import retrieve_folder_path, extend_directory

    global usb_array
    global check

    path_to_tensorDSE = retrieve_folder_path(os.getcwd(), "TensorDSE")
    docker_copy(path_to_tensorDSE, TO_DOCKER)

    models_info = deduce_operations_from_folder(folder,
                                                beginning="quant_",
                                                ending="_edgetpu.tflite")
    out_dir = "results/shark/"
    cnt = 5

    for m_i in models_info:
        for i in range(cnt):
            op = m_i[1]
            print(f"\nOperation: {op}")

            if i == 0:
                extend_directory(out_dir, op)

            print("Begun capture.")
            t_1 = threading.Thread(target=shark_capture_cont,
                                  args=(op, i))

            t_2 = threading.Thread(target=docker_exec, args=(
                    "shark_single_edge_deploy", m_i[0],))

            t_1.start()
            t_2.start()

            event.wait()
            event.clear()

            t_1.join()
            t_2.join()

            print("Ended capture.")
            print("\n")

        print("\n")



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

    if (args.mode == "All" and args.folder != ""):
        shark_usbmon_init()
        lsusb_identify()
        docker_start()
        shark_manager(args.folder)

    else:
        print("Invaild arguments.")
