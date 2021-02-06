DEBUG = 0


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

    def stamp_beginning(self, packet):
        """Saves the overloaded packet's timestamp onto ts_absolute_begin.

        Parameters
        ----------
        packet : object
        This object has as attributes all necessary data regarding an incoming
        usb packet. This will be the same attribute for all stamp-like methods.
        """
        self.ts_absolute_begin = float(packet.frame_info.time_relative)
        self.interrupt_begin_src = packet.usb.src
        self.interrupt_begin_dst = packet.usb.dst

    def stamp_ending(self, packet):
        """Saves the overloaded packet's timestamp onto ts_absolute_end."""
        self.ts_absolute_end = float(packet.frame_info.time_relative)
        self.interrupt_end_src = packet.usb.src
        self.interrupt_end_dst = packet.usb.dst

    def stamp_begin_host_send_request(self, packet):
        """Saves the overloaded packet's timestamp onto ts_begin_host_send_request."""
        self.ts_begin_host_send_request = float(packet.frame_info.time_relative)

    def stamp_end_host_send_request(self, packet):
        """Saves the overloaded packet's timestamp onto ts_end_host_send_request."""
        self.ts_end_host_send_request = float(packet.frame_info.time_relative)

    def stamp_beginning_submission(self, packet):
        """Saves the overloaded packet's timestamp onto ts_begin_submission."""
        self.ts_begin_submission = float(packet.frame_info.time_relative)

    def stamp_beginning_return(self, packet):
        """Saves the overloaded packet's timestamp onto ts_begin_return."""
        self.ts_begin_return = float(packet.frame_info.time_relative)

    def stamp_begin_tpu_send_request(self, packet):
        """Saves the overloaded packet's timestamp onto ts_begin_tpu_send_request."""
        self.ts_begin_tpu_send_request = float(packet.frame_info.time_relative)

    def stamp_end_tpu_send_request(self, packet):
        """Saves the overloaded packet's timestamp onto ts_end_tpu_send_request."""
        self.ts_end_tpu_send_request = float(packet.frame_info.time_relative)

    def stamp_src_host(self, packet):
        """Saves the overloaded packet's timestamp onto ts_end_submission."""
        self.ts_end_submission = float(packet.frame_info.time_relative)

    def stamp_src_device(self, packet):
        """Saves the overloaded packet's timestamp onto ts_end_return."""
        self.ts_end_return = float(packet.frame_info.time_relative)


class UsbPacket:
    """Class containing all necessary methods to decode/retreive human
    understandable information regarding incoming usb packets.

    Not only does these methods decode usb info but also sometimes exposes them
    as easy to use booleans that are useful in conditional statements.
    """

    def __init__(self):
        self.data_presence = "Void"

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
            "0x00000001":   "INTERRUPT",
            "0x00000002":   "CONTROL",
            "0x00000003":   self.find_bulk_type(),
        }

        self.transfer_type = transfer_dict.get(hexa_transfer_type, default)

    def find_urb_type(self, urb_type):
        """Finds the urb type of the overloaded packet."""
        default = None
        transfer_dict = {
            "'S'":   "SUBMIT",
            "'C'":   "COMPLETE",
            "":   None
        }

        self.urb_type = transfer_dict.get(urb_type, default)

    def find_bulk_type(self):
        """Finds the packet's usb bulk variation with use of its direction var."""
        if self.direction == '1':
            return "BULK IN"
        elif self.direction == '0':
            return "BULK OUT"
        else:
            raise ValueError(
                "Wrong attribute type in packet.endpoint_address_direction.")

    def find_data_presence(self, packet):
        """Finds if the overloaded packet contains actual DATA being sent."""

        tmp = packet.usb.data_flag
        if tmp == '>' or tmp == '<':
            self.data_presence = False
            return False
        elif 'not present' in tmp:
            self.data_presence = False
            return False
        elif tmp == 'present (0)':
            self.data_presence = True
            return True
        else:
            raise ValueError("Unknown data presence variable.")

    def verify_src(self, edge_tpu_id, string):
        if string == "begin":
            return True if ("host" == self.src and edge_tpu_id in self.dest) else False

        elif string == "end":
            return True if (edge_tpu_id in self.src and "host" == self.dest) else False

        else:
            raise ValueError("Unacceptable string value.")

def debug_stamps(usb_timer):
    """Function used to debug timing values, prints all important stamps."""
    from tabulate import tabulate

    table = [
                [   
                f"INTERRUPT BEGIN", f"{float(usb_timer.ts_absolute_begin)}", f"{usb_timer.interrupt_begin_src} ---> {usb_timer.interrupt_begin_dst}"
                ],

                [
                f"BEGIN OF REQUESTS (HOST)", f"{float(usb_timer.ts_begin_host_send_request)}"
                ],

                [
                f"END OF REQUESTS (HOST)", f"{float(usb_timer.ts_end_host_send_request)}"
                ],

                [
                f"BEGIN OF HOST SENT DATA", f"{float(usb_timer.ts_begin_submission)}"
                ],

                [
                f"END OF HOST SENT DATA", f"{float(usb_timer.ts_end_submission)}"
                ],

                [
                f"BEGIN OF REQUESTS (TPU)", f"{float(usb_timer.ts_begin_tpu_send_request)}"
                ],

                [
                f"END OF REQUESTS (TPU)", f"{float(usb_timer.ts_end_tpu_send_request)}"
                ],

                [
                f"BEGIN OF SUBMISSION (TPU)", f"{float(usb_timer.ts_begin_return)}"
                ],

                [
                f"END OF SUBMISSION (TPU)", f"{float(usb_timer.ts_end_return)}"
                ],

                [
                f"INTERRUPT END", f"{float(usb_timer.ts_absolute_end)}", f"{usb_timer.interrupt_end_src} ---> {usb_timer.interrupt_end_dst}"
                ],

                [
                f"TOTAL TIME: {float(float(usb_timer.ts_absolute_end) - float(usb_timer.ts_absolute_begin))}"
                ]
           ]

    print(f"\nTIMESTAMPS\n{tabulate(table)}")

def debug_usb(usb_array):
    from tabulate import tabulate

    i = 0
    print()
    table = []
    for u,i in zip(usb_array, range(len(usb_array))):
        tmp = [
               f"USB ({i + 1})", 
               f"{u.src} --> {u.dest}", 
               f"{u.transfer_type} - {u.urb_type}",
               f"{u.data_presence}"
               ]

        table.append(tmp)

    print(f"USB PACKETS\n{tabulate(table)}")


def export_analysis(usb_timer, op, append, filesize):
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

    filesize : String
    Size in Bytes of current op file.

    False - if a csv file has to be created, given one doesnt exist yet and then
    corresponding headers must then be placed.
    """
    import csv
    from utils import extend_directory

    results_dir = "results/usb/"
    results_folder = f"{op}_{filesize}"
    results_file = f"results/usb/{op}_{filesize}/Results.csv"

    if (append == True):
        with open(results_file, 'a+') as csvfile:
            fw = csv.writer(csvfile, delimiter=',', quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)

            fw.writerow([usb_timer.ts_absolute_begin,   # Same as ts_begin_host_send_request.
                         usb_timer.ts_begin_submission,  # ts_end_host_requests - negative values
                         usb_timer.ts_end_submission,
                         usb_timer.ts_begin_tpu_send_request,
                         usb_timer.ts_begin_return,  # ts_end_tpu_requests - negative values
                         usb_timer.ts_end_return])  # Sometimes absolute end doesnt match
    else:
        extend_directory(results_dir, results_folder)
        with open(results_file, 'w') as csvfile:

            fw = csv.writer(csvfile, delimiter=',', quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)

            fw.writerow(["start_host_request_and_absolute_begin",
                         "end_host_request_and_start_host_submission",
                         "end_host_submission_and_start_inference",
                         "start_tpu_request",
                         "end_inference_and_start_tpu_submission",
                         "end_tpu_submission_and_absolute_end"])

            fw.writerow([usb_timer.ts_absolute_begin,   # Same as ts_begin_host_send_request.
                         usb_timer.ts_begin_submission,  # ts_end_host_requests - negative values
                         usb_timer.ts_end_submission,
                         usb_timer.ts_begin_tpu_send_request,
                         usb_timer.ts_begin_return,  # ts_end_tpu_requests - negative values
                         usb_timer.ts_end_return])  # Sometimes absolute end doesnt match


def prepare_ulimit():
    import logging
    import os
    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    log.info("Prepping ulimit...")
    os.system("ulimit -n 4096")


def reset_ulimit():
    import logging
    import os
    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

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
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    log.info("IDing usb entry...")
    lsusb_cmd = "lsusb | grep Google > temp"
    os.system(lsusb_cmd)

    p = subprocess.run(list(["cat", "temp"]),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = p.stdout.decode()

    if "Google" in output:
        bus = int(output.split()[1])
        device = int((output.split()[3]).split(":")[0])
        edge_tpu_id = f"{bus}.{device}"
    else:
        lsusb_cmd = "lsusb | grep Global > temp"
        os.system(lsusb_cmd)
        p = subprocess.run(
            ["cat", "temp"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = p.stdout.decode()

        if "Global" in output:
            bus = int(output.split()[1])
            device = int((output.split()[3]).split(":")[0])
            edge_tpu_id = f"{bus}.{device}"
        else:
            raise ValueError(
                "Unable to identify Google Coral Bus and Device address.")

    os.system("[ -f temp ] && rm temp")
    return edge_tpu_id


def shark_usbmon_init():
    """Initializes the usbmon driver module."""
    import os
    usbmon_cmd = "sudo modprobe usbmon"
    os.system(usbmon_cmd)


def shark_capture_cont(op, cnt, edge_tpu_id, op_filesize):
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

    usb_array = []

    usb_timer = UsbTimer()

    beginning_of_comms = False
    end_of_comms = False

    beginning_of_submission = False
    begin_host_send_request = False

    begin_tpu_send_request = False
    beginning_of_return = False

    end_of_capture = False

    # capture_filter = "usb.transfer_type==URB_BULK || usb.transfer_type==URB_INTERRUPT"
    capture_filter = ""
    capture = pyshark.LiveCapture(
        interface='usbmon0', display_filter=capture_filter)

    for raw_packet in capture.sniff_continuously():
        custom_packet = UsbPacket()
        custom_packet.find_direction(raw_packet)
        custom_packet.find_transfer_type(raw_packet)
        custom_packet.find_scr_dest(raw_packet)
        custom_packet.find_urb_type(raw_packet.usb.urb_type)

        data_is_present = custom_packet.find_data_presence(raw_packet)

        # Checks for beginning and ending of captures, as interrupts appear at these point.
        if (custom_packet.transfer_type == "INTERRUPT"):

            if (beginning_of_comms == False
                    and custom_packet.verify_src(edge_tpu_id, "begin")):

                beginning_of_comms = True
                usb_timer.stamp_beginning(raw_packet)

                if DEBUG:
                    usb_array.append(custom_packet)

            elif (beginning_of_comms == True
                    and custom_packet.verify_src(edge_tpu_id, "end")):

                end_of_comms = True
                usb_timer.stamp_ending(raw_packet)

                if DEBUG:
                    usb_array.append(custom_packet)

            else:
                pass

        # Checks for BULK transfers, as they constitute actual data transfers or
        # connection requests..
        elif ((custom_packet.transfer_type == "BULK IN"
               or custom_packet.transfer_type == "BULK OUT")
              and beginning_of_comms == True):

            if (custom_packet.urb_type == "SUBMIT" and not data_is_present):
                assert (custom_packet.src 
                        == "host"), "Submit packets should only be from host!"

                if begin_host_send_request == False:
                    usb_timer.stamp_begin_host_send_request(raw_packet)
                    begin_host_send_request = True

                    if DEBUG:
                        usb_array.append(custom_packet)

                else:
                    usb_timer.stamp_end_host_send_request(raw_packet)

                    if DEBUG:
                        usb_array.append(custom_packet)

            elif (custom_packet.urb_type == "SUBMIT" and data_is_present == True):
                assert (custom_packet.src 
                        == "host"), "Submit packets should only be from host!"

                if beginning_of_submission == False:
                    usb_timer.stamp_beginning_submission(raw_packet)
                    beginning_of_submission = True

                    if DEBUG:
                        usb_array.append(custom_packet)
                else:
                    usb_timer.stamp_src_host(raw_packet)

                    if DEBUG:
                        usb_array.append(custom_packet)

            elif custom_packet.urb_type == "COMPLETE" and not data_is_present:
                assert (custom_packet.src !=
                        "host"), "Complete packets should only be from device!"

                if begin_tpu_send_request == False:
                    usb_timer.stamp_begin_tpu_send_request(raw_packet)
                    begin_tpu_send_request = True

                    if DEBUG:
                        usb_array.append(custom_packet)
                else:
                    usb_timer.stamp_end_tpu_send_request(raw_packet)

                    if DEBUG:
                        usb_array.append(custom_packet)

            elif (custom_packet.urb_type == "COMPLETE" and data_is_present == True):
                assert (custom_packet.src !=
                        "host"), "Complete packets should only be from device!"

                if beginning_of_return == False:
                    usb_timer.stamp_beginning_return(raw_packet)
                    beginning_of_return = True

                    if DEBUG:
                        usb_array.append(custom_packet)
                else:
                    usb_timer.stamp_src_device(raw_packet)

                    if DEBUG:
                        usb_array.append(custom_packet)

            else:
                raise ValueError(
                    "There shouldnt be non Bulk IN/OUT transfers within BULK.")

        else: # NON-BULK/INTERRUPT PACKETS - CONTROL PACKETS
            if end_of_comms == True:
                end_of_capture = True
                if DEBUG:
                    usb_array.append(custom_packet)

        if end_of_capture == True:
            if DEBUG:
                debug_stamps(usb_timer)
                debug_usb(usb_array)
                # export_analysis(usb_timer, op, cnt != 0, op_filesize)
            else:
                export_analysis(usb_timer, op, cnt != 0, op_filesize)

            break

    # capture.close()



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
    from multiprocessing import Process
    from docker import TO_DOCKER, FROM_DOCKER, HOME, docker_exec, docker_copy
    from utils import retrieve_folder_path, extend_directory, deduce_operations_from_folder, deduce_filesize

    global usb_array

    path_to_tensorDSE = retrieve_folder_path(os.getcwd(), "TensorDSE")
    docker_copy(path_to_tensorDSE, TO_DOCKER)

    models_info = deduce_operations_from_folder(folder,
                                                beginning="quant_",
                                                ending="_edgetpu.tflite")
    out_dir = "results/usb/"
    cnt = int(count)

    for m_i in models_info:
        for i in range(cnt):
            filepath = m_i[0]
            op = m_i[1]

            filesize = deduce_filesize(filepath)

            print(f"\nOperation: {op}")
            print("Begun capture.")

            p_1 = Process(target=shark_capture_cont,
                            args=(op, i, edge_tpu_id, filesize))

            p_2 = Process(target=docker_exec, 
                            args=("shark_single_edge_deploy", filepath,))

            p_1.start()
            p_2.start()

            p_2.join()
            p_1.join()
    
        print("Ended capture.")

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
    from multiprocessing import Process
    from docker import TO_DOCKER, FROM_DOCKER, HOME, docker_exec, docker_copy
    from utils import retrieve_folder_path, extend_directory, deduce_operation_from_file, deduce_filename, deduce_filesize

    global usb_array

    path_to_tensorDSE = retrieve_folder_path(os.getcwd(), "TensorDSE")
    docker_copy(path_to_tensorDSE, TO_DOCKER)

    filename = deduce_filename(model)
    filesize = deduce_filesize(model)
    op = deduce_operation_from_file(f"{filename}.tflite",
                                    beginning="quant_",
                                    ending="_edgetpu.tflite")
    out_dir = "results/usb/"
    cnt = int(count)

    for i in range(cnt):
        print(f"\nOperation: {op}")
        print("Begun capture.")

        p_1 = Process(target=shark_capture_cont,
                        args=(op, i, edge_tpu_id, filesize))

        p_2 = Process(target=docker_exec, 
                        args=("shark_single_edge_deploy", model,))

        p_1.start()
        p_2.start()

        p_2.join()
        p_1.join()
    
        print("Ended capture.")

    print("\n")


if __name__ == '__main__':
    import argparse
    from docker import docker_start

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-c', '--count', required=False,
                        default=1000,
                        help='Count of the number of times of edge deployment.')

    parser.add_argument('-m', '--mode', required=False,
                        default="Both",
                        help='Mode in which the script will run: All, Single, or Debug.')

    parser.add_argument('-f', '--folder', required=False,
                        default="models/compiled/",
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
        edge_tpu_id = lsusb_identify()
        docker_start()
        shark_single_manager(args.target, args.count, edge_tpu_id)
        reset_ulimit()

    elif (args.mode == "Debug"):
        from utils import deduce_filename, deduce_filesize, deduce_operations_from_folder
        DEBUG = 1

        shark_usbmon_init()
        edge_tpu_id = lsusb_identify()

        models_info = deduce_operations_from_folder(args.folder,
                                                    beginning="quant_",
                                                    ending="_edgetpu.tflite")

        for m_i in models_info:
            filename = m_i[1]
            filepath = m_i[0]
            filesize = deduce_filesize(filepath)

            inp = input(f"Operation {m_i[1]}, Continue to Next? ")
            if inp == "":
                continue
            elif inp == "c":
                print("End.")
                break
            else:
                shark_capture_cont(filename, 0, edge_tpu_id, filesize)


    else:
        print("Invalid arguments.")
