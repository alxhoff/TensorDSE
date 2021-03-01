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

        self.interrupt_begin_src = ""
        self.interrupt_begin_dst = ""
        self.interrupt_end_src = ""
        self.interrupt_end_dst = ""

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
        self.valid_data = False
        self.stamped = False
        self.foreign = False

    def timestamp(self, packet):
        self.ts = float(packet.frame_info.time_relative)

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

    def find_data_validity(self, packet, edge_tpu_id):
        """Finds if the overloaded packet contains valid DATA being sent."""
        self.urb_size = float(packet.usb.urb_len)
        self.data_size = float(packet.usb.data_len)

        if self.src != "host":
            if self.data_size > 0 and self.data_size >= 16:
                self.valid_data = True
                return True
            else:
                self.valid_data = False
                return False
        else:
            if self.data_size > 0:
                self.valid_data = True
                return True
            else:
                self.valid_data = False
                return False

    def verify_src(self, edge_tpu_id, string):
        valid=f"{edge_tpu_id}.3"
        if string == "begin":
            return True if ("host" == self.src and valid == self.dest) else False

        elif string == "end":
            return True if (valid == self.src and "host" == self.dest) else False

        else: # both
            return True if (((edge_tpu_id in self.src) and ("host" == self.dest))
                            or (("host" == self.src) and (edge_tpu_id in self.dest))) else False

    def mark_stamped(self):
        self.stamped = True

    def mark_foreign(self):
        self.foreign = True


def debug_color_text(text, color, back):
    from termcolor import colored
    return colored(text, color, back)


def debug_stamps(usb_timer_arr, sessions):
    """Function used to debug timing values, prints all important stamps."""
    from termcolor import colored
    from tabulate import tabulate

    first = True
    cnt = 0
    s_cnt = 0
    length = len(usb_timer_arr)
    for usb_timer in usb_timer_arr:
        s_cnt += 1
        cnt += 1
        host_comms = float(usb_timer.ts_begin_submission - usb_timer.ts_begin_host_send_request)
        host_data = float(usb_timer.ts_end_submission - usb_timer.ts_begin_submission)
        tpu_comms = float(usb_timer.ts_begin_return - usb_timer.ts_end_submission)
        tpu_data = float(usb_timer.ts_end_return - usb_timer.ts_begin_return)
        inference = float(usb_timer.ts_begin_return - usb_timer.ts_end_submission)
        total_time = float(usb_timer.ts_end_return - usb_timer.ts_begin_host_send_request)

        table = [
                    [   
                    f"INTERRUPT BEGIN" if (first) else "      ", 
                    f"{float(usb_timer.ts_absolute_begin)}" if (first) else "     ", 
                    "       "
                    ],

                    [
                    f"BEGIN OF REQUESTS (HOST)", 
                    f"{float(usb_timer.ts_begin_host_send_request)}",
                    debug_color_text(
                        f"{float(usb_timer.ts_begin_submission)} - {float(usb_timer.ts_begin_host_send_request)}", 
                        "white", None),
                    debug_color_text(
                        f"HOST COMMS: {host_comms * 10**6} us", 
                        "white", None),
                    debug_color_text(
                        f"{(host_comms / total_time) * 100} %", 
                        "white", None)
                    ],

                    [
                    f"END OF REQUESTS (HOST)", 
                    f"{float(usb_timer.ts_end_host_send_request)}"
                    ],

                    [
                    f"BEGIN OF HOST SENT DATA", 
                    f"{float(usb_timer.ts_begin_submission)}",
                    debug_color_text(
                        f"{float(usb_timer.ts_end_submission)} - {float(usb_timer.ts_begin_submission)}", 
                        "white", "on_red"),
                    debug_color_text(
                        f"HOST DATA: {host_data * 10**6} us", 
                        "white", "on_red"),
                    debug_color_text(
                        f"{(host_data / total_time) * 100} %", 
                        "white", "on_red")
                    ],

                    [
                    f"END OF HOST SENT DATA", 
                    f"{float(usb_timer.ts_end_submission)}"
                    ],

                    [
                    f"BEGIN OF REQUESTS (TPU)", 
                    f"{float(usb_timer.ts_begin_tpu_send_request)}",
                    debug_color_text(
                        f"{float(usb_timer.ts_begin_return)} - {float(usb_timer.ts_end_submission)}", 
                        "blue", None),
                    debug_color_text(
                        f"TPU COMMS: {tpu_comms * 10**6} us", 
                        "blue", None),
                    debug_color_text(
                        f"{(tpu_comms / total_time) * 100} %", 
                        "blue", None)
                    ],

                    [
                    f"END OF REQUESTS (TPU)", 
                    f"{float(usb_timer.ts_end_tpu_send_request)}"
                    ],

                    [
                    f"BEGIN OF SUBMISSION (TPU)", 
                    f"{float(usb_timer.ts_begin_return)}",
                    debug_color_text(
                        f"{float(usb_timer.ts_end_return)} - {float(usb_timer.ts_begin_return)}", 
                        "white", "on_blue"),
                    debug_color_text(
                        f"TPU DATA: {tpu_data * 10**6} us", 
                        "white", "on_blue"),
                    debug_color_text(
                        f"{(tpu_data / total_time) * 100} %", 
                        "white", "on_blue")
                    ],

                    [
                    f"END OF SUBMISSION (TPU)", 
                    f"{float(usb_timer.ts_end_return)}"
                    ],

                    [
                    f"INTERRUPT END" if (length == cnt) else "      ", 
                    f"{float(usb_timer.ts_absolute_end)}" if (length == cnt) else "     ", 
                    "       "
                    ],

                    [
                    f"INFERENCE TIME", 
                    "              ", 
                    debug_color_text(
                        f"{float(usb_timer.ts_begin_return)} - {float(usb_timer.ts_end_submission)}", 
                        "red", "on_white"),
                    debug_color_text(
                        f"INF: {inference * 10**6} us", 
                        "red", "on_white"),
                    debug_color_text(
                        f"{(inference / total_time) * 100} %", 
                        "red", "on_white")
                    ],

                    [
                    f"TOTAL TIME", 
                    "              ", 
                    debug_color_text(
                        f"{float(usb_timer.ts_end_return)} - {float(usb_timer.ts_begin_host_send_request)}", 
                        "blue", "on_white"),
                    debug_color_text(
                        f"TOT: {total_time * 10**6} us", 
                        "blue", "on_white"),
                    debug_color_text(
                        f"100 %", 
                        "blue", "on_white")
                    ]
               ]

        print(f"\nTIMESTAMPS[{cnt}/{int(length/sessions)} within SESSION {s_cnt}/{sessions}]\n{tabulate(table)}")

        if cnt == length/sessions:
            cnt = 0

        if s_cnt == sessions:
            s_cnt = 0

        first = False


def debug_usb(usb_array):
    from termcolor import colored
    from tabulate import tabulate

    i = 0
    print()
    color = ""
    back = None
    header = []
    header.append([
                    f"USB Nr",
                    f"Direction",
                    f"Types",
                    f"Timestamp",
                    f"Data Size",
                    f"URB Size",
                    f"Data Presence",
                    f"Data Validity",
                    f"STAMPED"
                    ])

    table = []
    for u,i in zip(usb_array, range(len(usb_array))):
        if not u.foreign:
            if u.transfer_type == "INTERRUPT":
                color = "magenta"
                back = None

            elif ("BULK" in u.transfer_type):
                if u.src == "host" and not u.data_presence:
                    color = "white"
                    back = None

                elif (u.src == "host" and not u.valid_data):
                    color = "red"
                    back = "on_blue"

                elif (u.src == "host" and u.data_presence and u.valid_data):
                    color = "white"
                    back = "on_red"

                elif (u.src != "host" and not u.data_presence):
                    color = "blue"
                    back = None

                elif (u.src != "host" 
                        and u.data_presence
                        and u.valid_data):
                    color = "white"
                    back = "on_blue"

                elif (u.src != "host" 
                        and u.data_presence
                        and not u.valid_data):
                    color = "red"
                    back = None

                else:
                    color = "white"
                    back = "on_magenta"
            else:
                color = "white"
                back = "on_magenta"
        else:
            color = "white"
            back = "on_magenta"

        tmp = [
               debug_color_text(f"USB ({i + 1})", color, back), 
               debug_color_text(f"{u.src} --> {u.dest}", color, back), 
               debug_color_text(f"{u.transfer_type} - {u.urb_type}", color, back),
               debug_color_text(f"{float(u.ts)}", color, back),
               debug_color_text(f"{u.data_size}", color, back),
               debug_color_text(f"{u.urb_size}", color, back),
               debug_color_text(f"{u.data_presence}", color, back),
               debug_color_text(f"{u.valid_data}", color, back),
               debug_color_text(f"{u.stamped}", color, back)
               ]

        table.append(tmp)

    print("USB PACKETS")
    print(f"{tabulate(header)}")
    print(f"{tabulate(table)}")


def export_analysis(op, filesize, sessions, usb_timer_arr):
    """Creates CSV file with the relevant usb transfer timestamps.

    The csv file will contain a header exposing the names of the variables
    stored in each row and then corresponding timestamps regarding the current
    usb traffic of the current instance of edge_tpu deployment.

    Parameters
    ---------
    op : String
    Current operation name.

    filesize : String
    Size in Bytes of current op file.

    sessions : int
    Number of sessions per model.

    usb_timer_arr : array
    Arrray containing instances of the UsbTimer() class.
    """
    import csv
    import logging
    from utils import extend_directory

    results_dir = "results/usb/"
    results_folder = f"{op}_{filesize}"
    results_file = f"results/usb/{op}_{filesize}/Results.csv"

    extend_directory(results_dir, results_folder)

    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    log.info("Exporting usb results...")

    with open(results_file, 'w') as csvfile:
        fw = csv.writer(csvfile, delimiter=',', quotechar='|',
                        quoting=csv.QUOTE_MINIMAL)

        fw.writerow(["start",
                     "end_host_comms",
                     "end_host_submission",
                     "start_tpu_comms",
                     "start_tpu_return",
                     "end_tpu_return"
                     ])

        for usb_timer in usb_timer_arr:
            row = []
            for _ in range(sessions):
                row.append(usb_timer.ts_begin_host_send_request)    # Begin
                row.append(usb_timer.ts_begin_submission)           # Begin Host Sub
                row.append(usb_timer.ts_end_submission)             # End Host Sub
                row.append(usb_timer.ts_end_submission)             # Begin TPU Request
                row.append(usb_timer.ts_begin_return)               # Begin TPU Return/ End TPU Request
                row.append(usb_timer.ts_end_return)                 # End TPU Sub/Return

            fw.writerow(row)


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


def usbmon_init():
    """Initializes the usbmon driver module."""
    import os
    usbmon_cmd = "sudo modprobe usbmon"
    os.system(usbmon_cmd)


def docker_deploy_sessions(model, op, count):
    import os
    from docker import HOME, FROM_DOCKER, TO_DOCKER
    from docker import docker_exec, docker_copy
    from utils import retrieve_folder_path, extend_directory
    import time

    cnt = int(count)

    path_to_tensorDSE = retrieve_folder_path(os.getcwd(), "TensorDSE")
    path_to_results = f"results/edge/{op}/"
    path_to_docker_results = HOME + \
        "TensorDSE/benchmarking/reading_tflite_model/results/"

    extend_directory("results/edge/", f"{op}")
    docker_copy(path_to_tensorDSE, TO_DOCKER)
    docker_exec("edge_single_deploy", model, count)
    docker_copy(f"{path_to_docker_results}edge/{op}/Results.csv",
                FROM_DOCKER, path_to_results)
    docker_exec("remove", "TensorDSE")


def shark_analyze_stream(edge_tpu_id):
    import pyshark

    usb_array = []

    BEGIN = False
    END = False

    capture_filter = ""
    capture = pyshark.LiveCapture(
        interface='usbmon0', display_filter=capture_filter)

    for raw_packet in capture.sniff_continuously():
        custom_packet = UsbPacket()
        custom_packet.timestamp(raw_packet)
        custom_packet.find_direction(raw_packet)
        custom_packet.find_transfer_type(raw_packet)
        custom_packet.find_scr_dest(raw_packet)
        custom_packet.find_urb_type(raw_packet.usb.urb_type)

        comms_is_valid = custom_packet.verify_src(edge_tpu_id, "all")
        data_is_present = custom_packet.find_data_presence(raw_packet)
        data_is_valid = custom_packet.find_data_validity(raw_packet)

        if (custom_packet.transfer_type == "INTERRUPT"):
            if (BEGIN == False and custom_packet.verify_src(edge_tpu_id, "begin")):
                BEGIN = True
                usb_array.append(custom_packet)

            elif (BEGIN == True and custom_packet.verify_src(edge_tpu_id, "end")):
                END = True
                usb_array.append(custom_packet)

        elif comms_is_valid and BEGIN:
            usb_array.append(custom_packet)

        if END:
            debug_usb(usb_array)
            break


def shark_analyze_sessions(op, op_filesize, sessions, count, edge_tpu_id):
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
    import copy
    import pyshark

    usb_timer_array = []
    usb_array = []

    BEGUN = False
    ENDED = False

    beginning_of_submission = False
    begin_host_send_request = False
    begin_tpu_send_request = False
    beginning_of_return = False

    beginning_invalid_return = False

    capture_filter = ""
    # c_parameters = {"-l":""}
    capture = pyshark.LiveCapture(
        interface='usbmon0', display_filter=capture_filter,
        only_summaries=True)

    rounds_nr = 0
    usb_timer = UsbTimer()
    for raw_packet in capture.sniff_continuously():
        custom_packet = UsbPacket()

        custom_packet.timestamp(raw_packet)
        custom_packet.find_direction(raw_packet)
        custom_packet.find_transfer_type(raw_packet)
        custom_packet.find_scr_dest(raw_packet)
        custom_packet.find_urb_type(raw_packet.usb.urb_type)

        comms_is_valid = custom_packet.verify_src(edge_tpu_id, "all")
        data_is_present = custom_packet.find_data_presence(raw_packet)
        data_is_valid = custom_packet.find_data_validity(raw_packet, edge_tpu_id)

        if (custom_packet.transfer_type == "INTERRUPT"): # Interrupts mark beginning and end
            if (BEGUN == False and custom_packet.verify_src(edge_tpu_id, "begin")):
                BEGUN = True
                custom_packet.mark_stamped()
                usb_timer.stamp_beginning(raw_packet)
                usb_array.append(custom_packet)

            elif (BEGUN == True and custom_packet.verify_src(edge_tpu_id, "end")):
                ENDED = True
                custom_packet.mark_stamped()
                usb_timer.stamp_ending(raw_packet)
                usb_array.append(custom_packet)

        elif comms_is_valid and BEGUN:
            if (custom_packet.transfer_type == "BULK OUT"):
                # Token packets from edge (non-data)
                if (not data_is_present
                        and edge_tpu_id in custom_packet.src
                        and custom_packet.urb_type == "COMPLETE"):

                    if (begin_tpu_send_request == False
                            and not beginning_of_return):
                        custom_packet.mark_stamped()
                        usb_timer.stamp_begin_tpu_send_request(raw_packet)
                        begin_tpu_send_request = True

                    if not beginning_of_return:
                        custom_packet.mark_stamped()
                        usb_timer.stamp_end_tpu_send_request(raw_packet)

                    usb_array.append(custom_packet)

                # Data packets from host
                if (data_is_present
                        and custom_packet.src == "host"
                        and custom_packet.urb_type == "SUBMIT"):

                    if begin_tpu_send_request == True:
                        begin_tpu_send_request = False

                    if beginning_of_submission and beginning_of_return: # New Session
                        custom_packet.mark_stamped()
                        last_usb_timer = copy.deepcopy(usb_timer)
                        usb_timer = UsbTimer()
                        usb_timer_array.append(last_usb_timer)

                        usb_timer.ts_begin_host_send_request = last_usb_timer.ts_end_return
                        usb_timer.ts_end_host_send_request = usb_array[-1].ts

                        beginning_of_submission = False
                        begin_tpu_send_request = False
                        beginning_of_return = False

                    if beginning_of_submission == False:
                        custom_packet.mark_stamped()
                        usb_timer.stamp_beginning_submission(raw_packet)
                        beginning_of_submission = True

                    if (data_is_valid
                            and beginning_of_submission):
                        custom_packet.mark_stamped()
                        usb_timer.stamp_src_host(raw_packet)

                    usb_array.append(custom_packet)

            # Data from edge and tokens/requests from host
            if (custom_packet.transfer_type == "BULK IN"):
                # Token packets from host (non-data)
                if (not data_is_present 
                        and custom_packet.src == "host"
                        and custom_packet.urb_type == "SUBMIT"):

                    if (begin_host_send_request == False
                            and not beginning_of_submission): # Stamp initial packets 
                        custom_packet.mark_stamped()
                        usb_timer.stamp_begin_host_send_request(raw_packet)
                        begin_host_send_request = True

                    if not beginning_of_submission:
                        custom_packet.mark_stamped()
                        usb_timer.stamp_end_host_send_request(raw_packet)

                    usb_array.append(custom_packet)

                # Data packets from edge
                if (data_is_present and 
                        edge_tpu_id in custom_packet.src
                        and custom_packet.urb_type == "COMPLETE"):

                    if (beginning_of_return == False # Stamp initial packets 
                            and data_is_valid
                            and custom_packet.src != f"{edge_tpu_id}.2"):
                        custom_packet.mark_stamped()
                        usb_timer.stamp_beginning_return(raw_packet)
                        beginning_of_return = True

                    if (data_is_valid
                        and beginning_of_return):
                        custom_packet.mark_stamped()
                        usb_timer.stamp_src_device(raw_packet)

                    usb_array.append(custom_packet)

            else:
                custom_packet.mark_foreign
                usb_array.append(custom_packet)

        if ENDED:
            if DEBUG:
                usb_timer_array.append(usb_timer)
                debug_usb(usb_array)
                debug_stamps(usb_timer_array, sessions)
            else:
                usb_timer_array.append(usb_timer)
                export_analysis(op, op_filesize, sessions, usb_timer_array)
            break


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
    import logging
    import subprocess
    from multiprocessing import Process
    from utils import retrieve_folder_path, extend_directory, deduce_operation_from_file
    from utils import create_csv_file, deduce_filename, deduce_filesize
    from utils import deduce_sessions_nr
    from plot import plot_manager

    filename = deduce_filename(model)
    filesize = deduce_filesize(model)
    op = deduce_operation_from_file(
            f"{filename}.tflite", beginning="quant_", ending="_edgetpu.tflite")

    sessions = deduce_sessions_nr(op)

    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.INFO)
    log.info(f"Operation {op}: {count}x")
    # assert count % 5 == 0, "Count must divisable by 5."

    p_1 = Process(target=shark_analyze_sessions,
                    args=(op, filesize, sessions, count, edge_tpu_id))

    p_2 = Process(target=docker_deploy_sessions, 
                    args=(model, op, count))

    p_1.start()
    p_2.start()

    p_2.join()
    p_1.join()

    plot_manager(f"{op}_{filesize}", filesize, sessions)


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
    from utils import retrieve_folder_path, deduce_operations_from_folder
    from plot import plot_manager

    models_info = deduce_operations_from_folder(folder,
                                                beginning="quant_",
                                                ending="_edgetpu.tflite")
    out_dir = "results/usb/"
    cnt = int(count)

    for m_i in models_info:
        filepath = m_i[0]
        op = m_i[1]
        shark_single_manager(filepath, cnt, edge_tpu_id)


if __name__ == '__main__':
    import argparse
    from utils import prepare_ulimit, reset_ulimit, deduce_sessions_nr
    from utils import deduce_filename, deduce_filesize, deduce_operations_from_folder
    from docker import docker_start

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-m', '--mode', required=False,
                        default="Both",
                        help='Mode in which the script will run: Group, Single, Capture or Debug.')

    parser.add_argument('-f', '--folder', required=False,
                        default="models/compiled/",
                        help='Folder.')

    parser.add_argument('-t', '--target', required=False,
                        default="",
                        help='Model.')

    parser.add_argument('-c', '--count', required=False,
                        default=1,
                        help='Count of the number of times of edge deployment.')


    args = parser.parse_args()

    if (args.mode == "Group" and args.folder != ""):
        usbmon_init()
        prepare_ulimit(10096)
        edge_tpu_id = lsusb_identify()
        docker_start()
        shark_manager(args.folder, args.count, edge_tpu_id)
        reset_ulimit()

    elif (args.mode == "Single"):
        usbmon_init()
        prepare_ulimit()
        edge_tpu_id = lsusb_identify()
        docker_start()
        shark_single_manager(args.target, args.count, edge_tpu_id)
        reset_ulimit()

    elif (args.mode == "Capture"):
        usbmon_init()
        edge_tpu_id = lsusb_identify()
        shark_analyze_stream(edge_tpu_id)

    elif (args.mode == "Debug"):
        DEBUG = 1
        usbmon_init()
        edge_tpu_id = lsusb_identify()
        models_info = deduce_operations_from_folder(
                args.folder, beginning="quant_", ending="_edgetpu.tflite")

        for m_i in models_info:
            filename = m_i[1]
            filepath = m_i[0]
            filesize = deduce_filesize(filepath)
            sessions = deduce_sessions_nr(filename)

            inp = input(f"Operation {m_i[1]}, Continue to Next? ")
            if inp == "": continue
            elif inp == "c": break
            else:
                shark_analyze_sessions(
                        filename, filesize, sessions, int(args.count), edge_tpu_id)

    else:
        print("Invalid arguments.")
