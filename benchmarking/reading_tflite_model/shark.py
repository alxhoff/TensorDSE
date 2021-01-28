import threading
event = threading.Event()

edge_tpu_id = ""

usb_array = []

class UsbTimer:
    """Class containing all necessary time stamps regarding usb traffic and
    their methods.

    As usb traffic is read during the edge_tpu deployment. These methods(functions)
    will either simply save the overloaded timestamps onto the class instance or 
    perform minor operations to deduce other relevant information.

    Attributes
    -------
    """
    def __init__(self):
        # Absolute Begin
        self.ts_absolute_begin = 0

        # Last CPU sent request before actual data was sent to device(TPU).
        self.ts_end_host_send_request = 0 

        self.ts_end_submission = 0 

        # Last CPU sent of actual data and where the TPU begins to compute the result.
        self.ts_begin_inference = 0

        # Last TPU sent request before actual data was sent to the CPU.
        self.ts_end_tpu_send_request = 0

        # First Data present packet sent from device(TPU) to CPU.
        self.ts_end_inference = 0

        # End of Receiving Data
        self.ts_absolute_end = 0


    def print_stamps(self):
        """Function used to debug timing values, prints all important stamps."""

        print(f"ABSOLUTE BEGINNING/BEGINNING OF REQUESTS (HOST): {self.ts_absolute_begin}")
        print(f"END OF REQUESTS (HOST): {self.ts_end_host_send_request}")
        print(f"END OF HOST SENT DATA: {self.ts_end_submission}")
        print(f"INFERENCE BEGIN/LAST DATA FROM HOST SENT/BEGIN OF REQUESTS (TPU): {self.ts_begin_inference}")
        print(f"END OF REQUESTS (TPU)/BEGIN OF TPU SENT DATA: {self.ts_end_tpu_send_request}")
        print(f"ABSOLUTE END: {self.ts_absolute_end}")

    def stamp_beginning(self, packet):
        """Saves the overloaded packet's timestamp onto ts_absolute_begin.

        Parameters
        ----------
        packet : object
        This object has as attributes all necessary data regarding an incoming
        usb packet. This will be the same attribute for all stamp-like methods.
        """
        self.ts_absolute_begin = packet.frame_info.time_relative

    def stamp_ending(self, packet):
        """Saves the overloaded packet's timestamp onto ts_absolute_end."""
        self.ts_absolute_end = packet.frame_info.time_relative

    def stamp_end_host_send_request(self, packet):
        """Saves the overloaded packet's timestamp onto ts_end_host_send_request."""
        self.ts_end_host_send_request = packet.frame_info.time_relative
    
    def stamp_beginning_submission(self, packet):
        """Saves the overloaded packet's timestamp onto ts_begin_submission."""
        self.ts_begin_submission = packet.frame_info.time_relative

    def stamp_beginning_return(self, packet):
        """Saves the overloaded packet's timestamp onto ts_begin_return."""
        self.ts_begin_return = packet.frame_info.time_relative
        self.ts_begin_inference = self.ts_end_submission
        self.ts_end_inference = self.ts_begin_return

    def stamp_end_tpu_send_request(self, packet):
        """Saves the overloaded packet's timestamp onto ts_end_tpu_send_request."""
        self.ts_end_tpu_send_request = packet.frame_info.time_relative

    def stamp_src_host(self, packet):
        """Saves the overloaded packet's timestamp onto ts_end_submission."""
        self.ts_end_submission = packet.frame_info.time_relative

    def stamp_src_device(self, packet):
        """Saves the overloaded packet's timestamp onto ts_end_return."""
        self.ts_end_return = packet.frame_info.time_relative
        self.ts_end_inference = self.ts_end_return

class UsbPacket:
    """Class containing all necessary methods to decode/retreive human
    understandable information regarding incoming usb packets.

    Not only does these methods decode usb info but also sometimes exposes them
    as easy to use booleans that are useful in conditional statements.
    """
    def __init__(self):
        pass

    def find_direction(self, packet):
        """Method that stores the overloaded packet's flag denoting direction
        of usb transfer.
        """
        self.direction = packet.usb.endpoint_address_direction

    def find_scr_dest(self, packet):
        """Stores source and destination values of overloaded packet."""
        self.src = packet.usb.src
        self.dest = packet.usb.dst

    def find_transfer_type(self, packet):
        """Finds the urb transfer type of the overloaded packet."""
        hexa_transfer_type = packet.usb.transfer_type
        default = None
        transfer_dict = {
                "0x00000001"    :   "INTERRUPT",
                "0x00000002"    :   "CONTROL",
                "0x00000003"    :   self.find_bulk_type(),
                ""              :   None
                }

        self.transfer_type = transfer_dict.get(hexa_transfer_type, default)

    def find_urb_type(self, urb_type):
        """Finds the urb type of the overloaded packet."""
        default = None
        transfer_dict = {
                "'S'"    :   "SUBMIT",
                "'C'"    :   "COMPLETE",
                ""              :   None
                }

        self.urb_type = transfer_dict.get(urb_type, default)

    def find_bulk_type(self):
        """Finds the packet's usb bulk variation with use of its direction var."""
        if self.direction == '1':
            return "BULK IN"
        elif self.direction == '0':
            return "BULK OUT"
        else:
            raise ValueError("Wrong attribute type in packet.endpoint_address_direction.")

    def find_data_presence(self, packet):
        """Finds if the overloaded packet contains actual DATA being sent."""
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
    """Creates CSV file with the relevant usb transfer timestamps.

    The csv file will contain a header exposing the names of the variables
    stored in each row and then corresponding timestamps regarding the current 
    usb traffic of the current instance of edge_tpu deployment.

    Parameters
    ---------
    usb_timer : Object
    Instance of the UsbTimer() class.

    op : String
    Current operation name. 

    append : bool
    True - if the csv file has already been created and the new values are to be
    appended to the existent 'Results.csv' file.

    False - if a csv file has to be created, given one doesnt exist yet and then 
    corresponding headers must then be placed.
    """
    import csv
    from utils import extend_directory

    results_file = f"results/shark/{op}/Results.csv"

    if (append == True):
        with open(results_file, 'a+') as csvfile:
            fw = csv.writer(csvfile, delimiter=',', quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)

            fw.writerow([usb_timer.ts_absolute_begin, usb_timer.ts_end_host_send_request, 
                         usb_timer.ts_end_submission, usb_timer.ts_end_tpu_send_request,
                         usb_timer.ts_end_inference, usb_timer.ts_absolute_end])
    else:
        with open(results_file, 'w') as csvfile:
            fw = csv.writer(csvfile, delimiter=',', quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)

            fw.writerow(["absolute_begin", "end_requests_host", 
                        "end_host_submissions", "end_requests_tpu", 
                        "inference_end", "absolute_end"])

            fw.writerow([usb_timer.ts_absolute_begin, usb_timer.ts_end_host_send_request, 
                         usb_timer.ts_end_submission, usb_timer.ts_end_tpu_send_request,
                         usb_timer.ts_end_inference, usb_timer.ts_absolute_end])


def lsusb_identify():
    """Aims to identify the device ID where the edge tpu is connected. 

    This ID is constituted as the bus Nr it os on concatenated by a device Nr 
    which is created as usb devices are plugged onto to the host. The latter
    value will change/increment every time the edge_tpu is re-plugged onto the 
    host.
    """
    import os
    import logging

    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.INFO)

    log.info("IDing usb entry...")

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
    """Initializes the usbmon driver module."""
    import os
    usbmon_cmd = "sudo modprobe usbmon"
    os.system(usbmon_cmd)


def shark_capture_cont(op, cnt):
    """Continuouly reads usb packet traffic and retreives the necessary
    timestamps within that cycle.

    This Function is called as a separate thread to a edge tpu deployment or is 
    called alone if the user also deploys the edge tpu on a separate instance or
    script. With use of the pyshark module, this function will listen onto the 
    usbmon0 interface for usb traffic and will read packet by packet as they are
    sniffed. By Decoding/Reading packet info it will take note of important
    timestamps that denote significant moments during host <-> edge_tpu
    communication.

    Parameters
    ---------
    op : String
    Characterizes the operation name.

    cnt : Integer
    Indicates which number this instance is of the for loop running in the
    parent process of the 'shark_manager()'. Useful to know if it is the first,
    which then urges the need to create a brand new 'Results.csv' file
    concerning this instance of edge deployment usb traffic analysis.
    """
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
        usb_packet.find_transfer_type(packet)
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

        # Checks for BULK transfers, as they constitute actual data transfers or
        # connection requests..
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


def shark_manager(folder, count):
    """Manages the two threads that take care of deploying and listening to
    edge_tpu <-> host communication.

    This Function is called to manage two simple threads, one will deploy
    edge_tpu tflite models and the other calls on the 'shark_capture_cont'
    function which will listen on usb traffic and retrieve the necessary
    timestamps. With use of the 'check' object one is able to signal flags
    between threads or between child and parent processes. Only when this flag
    is signaled that the 'check.wait()' function will stop blocking and the rest 
    of the code will then continue executing.

    Parameters
    ---------
    folder : String
    Characterizes the folder where the compiled edge tflite models are located.
    """
    import os
    import time
    import threading
    import pyshark
    from docker import TO_DOCKER, FROM_DOCKER, HOME, docker_exec, docker_copy
    from utils import retrieve_folder_path, extend_directory, deduce_operations_from_folder

    global usb_array
    global check

    path_to_tensorDSE = retrieve_folder_path(os.getcwd(), "TensorDSE")
    docker_copy(path_to_tensorDSE, TO_DOCKER)

    models_info = deduce_operations_from_folder(folder,
                                                beginning="quant_",
                                                ending="_edgetpu.tflite")
    out_dir = "results/shark/"
    cnt = int(count)

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
            time.sleep(1)
            print("\n")

        print("\n")


def shark_single_manager(model, count):
    """Manages the two threads that take care of deploying and listening to
    edge_tpu <-> host communication.

    This Function is called to manage two simple threads, one will deploy
    a single edge_tpu tflite model and the other calls on the 'shark_capture_cont'
    function which will listen on usb traffic and retrieve the necessary
    timestamps. With use of the 'check' object one is able to signal flags
    between threads or between child and parent processes. Only when this flag
    is signaled that the 'check.wait()' function will stop blocking and the rest 
    of the code will then continue executing.

    Parameters
    ---------
    folder : String
    Characterizes the folder where the compiled edge tflite models are located.
    """
    import os
    import time
    import threading
    import pyshark
    from docker import TO_DOCKER, FROM_DOCKER, HOME, docker_exec, docker_copy
    from utils import retrieve_folder_path, extend_directory, deduce_operation_from_file, deduce_filename

    global usb_array
    global check

    path_to_tensorDSE = retrieve_folder_path(os.getcwd(), "TensorDSE")
    docker_copy(path_to_tensorDSE, TO_DOCKER)

    filename = deduce_filename(model)
    op = deduce_operation_from_file(f"{filename}.tflite",
                                    beginning=None,
                                    ending="_edgetpu.tflite")
    out_dir = "results/shark/"
    cnt = int(count)

    for i in range(cnt):
        if i == 0:
            extend_directory(out_dir, op)

        print(f"\nOperation: {op}")
        print("Begun capture.")
        t_1 = threading.Thread(target=shark_capture_cont,
                              args=(op, i))

        t_2 = threading.Thread(target=docker_exec, args=(
                "shark_single_edge_deploy", model,))

        t_1.start()
        t_2.start()

        event.wait()
        event.clear()

        t_1.join()
        t_2.join()

        print("Ended capture.")
        time.sleep(1)
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

    parser.add_argument('-m', '--mode', required=False,
                        default="Both",
                        help='Mode in which the script will run: All, Read, Capture or Deploy.')

    parser.add_argument('-f', '--folder', required=False,
                        default="",
                        help='Folder.')

    parser.add_argument('-t', '--target', required=False,
                        default="",
                        help='Model.')

    args = parser.parse_args()

    if (args.mode == "All" and args.folder != ""):
        shark_usbmon_init()
        lsusb_identify()
        docker_start()
        shark_manager(args.folder, args.count)

    elif (args.mode == "Single"):
        shark_usbmon_init()
        lsusb_identify()
        docker_start()
        shark_single_manager(args.target, args.count)
    else:
        print("Invaild arguments.")
