import threading
event = threading.Event()

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

        # Begin host sent request before actual data was sent to device(TPU).
        self.ts_begin_host_send_request = 0

        # Last host sent request before actual data was sent to device(TPU).
        self.ts_end_host_send_request = 0 

        # Beginning of host sent data to device(TPU).
        self.ts_begin_submission = 0

        # End of host sent data to device(TPU).
        self.ts_end_submission = 0 

        # First TPU sent request before actual data was sent to the CPU.
        self.ts_begin_tpu_send_request = 0

        # Last TPU sent request before actual data was sent to the CPU.
        self.ts_end_tpu_send_request = 0

        # End of device(TPU) sent data to host.
        self.ts_begin_return = 0 

        # End of device(TPU) sent data to host.
        self.ts_end_return = 0 

        # End of Receiving Data
        self.ts_absolute_end = 0


    def print_stamps(self):
        """Function used to debug timing values, prints all important stamps."""

        print(f"ABSOLUTE BEGINNING: {self.ts_absolute_begin}")

        print(f"BEGIN OF REQUESTS (HOST): {self.ts_begin_host_send_request}")
        print(f"END OF REQUESTS (HOST): {self.ts_end_host_send_request}")

        print(f"BEGIN OF HOST SENT DATA: {self.ts_begin_submission}")
        print(f"END OF HOST SENT DATA: {self.ts_end_submission}")

        print(f"BEGIN OF REQUESTS (TPU): {self.ts_begin_tpu_send_request}")
        print(f"END OF REQUESTS (TPU):{self.ts_end_tpu_send_request}")

        print(f"INFERENCE END/BEGIN OF SUBMISSION (TPU): {self.ts_begin_return}")
        print(f"END OF SUBMISSION (TPU): {self.ts_end_return}")

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

    def stamp_begin_host_send_request(self, packet):
        """Saves the overloaded packet's timestamp onto ts_begin_host_send_request."""
        self.ts_begin_host_send_request = packet.frame_info.time_relative

    def stamp_end_host_send_request(self, packet):
        """Saves the overloaded packet's timestamp onto ts_end_host_send_request."""
        self.ts_end_host_send_request = packet.frame_info.time_relative
    
    def stamp_beginning_submission(self, packet):
        """Saves the overloaded packet's timestamp onto ts_begin_submission."""
        self.ts_begin_submission = packet.frame_info.time_relative

    def stamp_beginning_return(self, packet):
        """Saves the overloaded packet's timestamp onto ts_begin_return."""
        self.ts_begin_return = packet.frame_info.time_relative

    def stamp_begin_tpu_send_request(self, packet):
        """Saves the overloaded packet's timestamp onto ts_begin_tpu_send_request."""
        self.ts_begin_tpu_send_request = packet.frame_info.time_relative

    def stamp_end_tpu_send_request(self, packet):
        """Saves the overloaded packet's timestamp onto ts_end_tpu_send_request."""
        self.ts_end_tpu_send_request = packet.frame_info.time_relative

    def stamp_src_host(self, packet):
        """Saves the overloaded packet's timestamp onto ts_end_submission."""
        self.ts_end_submission = packet.frame_info.time_relative

    def stamp_src_device(self, packet):
        """Saves the overloaded packet's timestamp onto ts_end_return."""
        self.ts_end_return = packet.frame_info.time_relative

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
        default = "OTHER"
        transfer_dict = {
                "0x00000001"    :   "INTERRUPT",
                "0x00000002"    :   "CONTROL",
                "0x00000003"    :   self.find_bulk_type(),
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

    def verify_src(self, edge_tpu_id, string):
        if string == "begin":
            return True if ("host" == self.src and edge_tpu_id in self.dest) else False

        elif string == "end":
            return True if (edge_tpu_id in self.src and "host" == self.dest) else False

        else:
            raise ValueError("Unacceptable string value.")


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

    results_file = f"results/usb/{op}/Results.csv"

    if (append == True):
        with open(results_file, 'a+') as csvfile:
            fw = csv.writer(csvfile, delimiter=',', quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)

            fw.writerow([usb_timer.ts_absolute_begin, #Same as ts_begin_host_send_request.
                         usb_timer.ts_end_host_send_request, 
                         usb_timer.ts_end_submission, 
                         usb_timer.ts_begin_tpu_send_request,
                         usb_timer.ts_begin_return, 
                         usb_timer.ts_absolute_end]) #Should be same as ts_end_return.
    else:
        with open(results_file, 'w') as csvfile:
            fw = csv.writer(csvfile, delimiter=',', quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)

            fw.writerow(["start_host_request_and_absolute_begin",
                         "end_host_request_and_start_host_submission", 
                         "end_host_submission_and_start_inference",
                         "start_tpu_request", 
                         "end_inference_and_start_tpu_submission",
                         "end_tpu_submission_and_absolute_end"])

            fw.writerow([usb_timer.ts_absolute_begin, #Same as ts_begin_host_send_request.
                         usb_timer.ts_end_host_send_request, 
                         usb_timer.ts_end_submission, 
                         usb_timer.ts_begin_tpu_send_request,
                         usb_timer.ts_begin_return, 
                         usb_timer.ts_absolute_end]) #Should be same as ts_end_return.


def prepare_ulimit():
    import logging
    import os
    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.INFO)

    log.info("Prepping ulimit...")
    os.system("ulimit -n 4096")


def reset_ulimit():
    import logging
    import os
    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.INFO)

    log.info("Resetting ulimit...")
    os.system("ulimit -n 1024")


def lsusb_identify():
    """Aims to identify the device ID where the edge tpu is connected. 

    This ID is constituted as the bus Nr it os on concatenated by a device Nr 
    which is created as usb devices are plugged onto to the host. The latter
    value will change/increment every time the edge_tpu is re-plugged onto the 
    host.
    """
    import os
    import subprocess
    import logging

    edge_tpu_id = ""

    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.INFO)

    log.info("IDing usb entry...")
    lsusb_cmd = "lsusb | grep Google > temp"
    os.system(lsusb_cmd)

    p = subprocess.run(list(["cat", "temp"]), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = p.stdout.decode()
    
    if "Google" in output:
        bus = int(output.split()[1])
        device = int((output.split()[3]).split(":")[0])
        edge_tpu_id = f"{bus}.{device}"
    else:
        lsusb_cmd = "lsusb | grep Global > temp"
        os.system(lsusb_cmd)
        p = subprocess.run(["cat", "temp"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = p.stdout.decode()

        if "Global" in output:
            bus = int(output.split()[1])
            device = int((output.split()[3]).split(":")[0])
            edge_tpu_id = f"{bus}.{device}"
        else:
            raise ValueError("Unable to identidy Google Coral Bus and Device address.")

    os.system("[ -f temp ] && rm temp")
    return edge_tpu_id


def shark_usbmon_init():
    """Initializes the usbmon driver module."""
    import os
    usbmon_cmd = "sudo modprobe usbmon"
    os.system(usbmon_cmd)


def shark_capture_cont(op, cnt, edge_tpu_id):
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

    edge_tpu_id : String
    Characterizes the device and bus address where the tpu is mounted on.
    """
    import os
    import pyshark

    global event
    global usb_array

    usb_timer = UsbTimer()

    beginning_of_comms = False
    end_of_comms = False

    beginning_of_submission = False
    begin_host_send_request = False

    begin_tpu_send_request = False
    beginning_of_return = False

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

            if (beginning_of_comms == False
                    and usb_packet.verify_src(edge_tpu_id, "begin")):

                usb_timer.stamp_beginning(packet)
                beginning_of_comms = True

            elif (beginning_of_comms == True 
                    and usb_packet.verify_src(edge_tpu_id, "end")):

                print(packet.usb.src)
                print(packet.usb.dst)
                end_of_comms = True
                usb_timer.stamp_ending(packet)
            else:
                pass

        # Checks for BULK transfers, as they constitute actual data transfers or
        # connection requests..
        elif ((usb_packet.transfer_type == "BULK IN" 
               or usb_packet.transfer_type == "BULK OUT")):

            if (usb_packet.urb_type == "SUBMIT" and data_is_present == False):
                assert (usb_packet.src == "host"), "Submit packets should only be from host!"

                if (begin_host_send_request == False):
                    usb_timer.stamp_begin_host_send_request(packet)
                    begin_host_send_request = True

                usb_timer.stamp_end_host_send_request(packet)

            elif (usb_packet.urb_type == "SUBMIT" and data_is_present == True):
                assert (usb_packet.src == "host"), "Submit packets should only be from host!"

                if beginning_of_submission == False and beginning_of_comms ==True:
                    usb_timer.stamp_beginning_submission(packet)
                    beginning_of_submission = True

                usb_timer.stamp_src_host(packet)

            elif usb_packet.urb_type == "COMPLETE" and data_is_present == False:
                assert (usb_packet.src != "host"), "Complete packets should only be from device!"

                if (begin_tpu_send_request == False and beginning_of_comms == True):
                    usb_timer.stamp_begin_tpu_send_request(packet)
                    begin_tpu_send_request = True

                usb_timer.stamp_end_tpu_send_request(packet)

            elif (usb_packet.urb_type == "COMPLETE" and data_is_present == True):
                assert (usb_packet.src != "host"), "Complete packets should only be from device!"

                if beginning_of_return == False:
                    usb_timer.stamp_beginning_return(packet)
                    beginning_of_return = True

                usb_timer.stamp_src_device(packet)

            else:
                pass

        else:
            pass

        if (not event.is_set() and end_of_comms == True):
            usb_timer.print_stamps()
            export_analysis(usb_timer, op, cnt!=0)
            event.set()
            break


def shark_manager(folder, count, edge_tpu_id):
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
                                  args=(op, i, edge_tpu_id))

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


def shark_single_manager(model, count, edge_tpu_id):
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
                              args=(op, i, edge_tpu_id))

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
        prepare_ulimit()
        edge_tpu_id = lsusb_identify()
        docker_start()
        shark_manager(args.folder, args.count, edge_tpu_id)
        reset_ulimit()

    elif (args.mode == "Single"):
        shark_usbmon_init()
        prepare_ulimit()
        lsusb_identify()
        docker_start()
        shark_single_manager(args.target, args.count, edge_tpu_id)
        reset_ulimit()
    else:
        print("Invaild arguments.")
