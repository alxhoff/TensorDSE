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
        self.ts_begin_host_send_request = float(
            packet.frame_info.time_relative)

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
        """Verifies if the packet's source is one that is valid for our analysis."""
        valid = f"{edge_tpu_id}.3"
        if string == "begin":
            return True if ("host" == self.src and valid == self.dest) else False

        elif string == "end":
            return True if (valid == self.src and "host" == self.dest) else False

        else:  # both
            return True if (((edge_tpu_id in self.src) and ("host" == self.dest))
                            or (("host" == self.src) and (edge_tpu_id in self.dest))) else False

    def mark_stamped(self):
        """Sets the stamped flag, useful for debugging to see if packets were
        stamped properly.
        """
        self.stamped = True

    def mark_foreign(self):
        """Sets the foreign flag, useful for debugging to see if foreign packets
        or unwanted packets are appearing during our URB transfers.
        """
        self.foreign = True


def debug_color_text(text, color, back):
        """Applies color to overloaded text using 'color' as foreground and
        'back' as the background.
        """
    from termcolor import colored
    return colored(text, color, back)


def debug_stamps(usb_timer_arr, sessions):
    """Function used to debug timing values, prints all important stamps
    regarding the saved packets in usb_timer_arr and applies colors to them
    according to their characteristics (URB Transfer Type, Size of Data, etc).
    """
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

def integrate_results(op, count):
    import os
    import csv
    from utils import parse_csv

    first_file = f"results/edge/{op}/Firsts.csv"
    res_file = f"results/edge/{op}/Results.csv"
    temp_file = f"results/edge/{op}/Temp.csv"

    results = parse_csv(temp_file)
    first_results = results[0]
    if count:
        flag = 'a+'
        i = len(parse_csv(res_file))
        j = len(parse_csv(first_file))
    else:
        flag = 'w'
        i = 0
        j = 0

    with open(res_file, flag) as csvfile:
        fw = csv.writer(csvfile, delimiter=',', quotechar='|',
                        quoting=csv.QUOTE_MINIMAL)

        if len(results) > 1:
            for res in results[1:]:
                fw.writerow([i, res])
                i += 1
        else:
            fw.writerow([i, results[0]])

    with open(first_file, flag) as csvfile:
        fw = csv.writer(csvfile, delimiter=',', quotechar='|',
                        quoting=csv.QUOTE_MINIMAL)

        fw.writerow([j, first_results])
    os.system(f"[ -f {temp_file} ] && rm {temp_file}")

def export_analysis(op, filesize, sessions, first_results, results, comms):
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

    firsts_file = f"results/usb/{op}_{filesize}/Firsts.csv"
    results_file = f"results/usb/{op}_{filesize}/Results.csv"
    comms_file = f"results/usb/{op}_{filesize}/Comms.csv"

    extend_directory(results_dir, results_folder)

    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    log.info("Exporting usb results...")

    with open(results_file, 'w') as csvfile:
        fw = csv.writer(csvfile, delimiter=',', quotechar='|',
                        quoting=csv.QUOTE_MINIMAL)

        fw.writerow(["Host_Submission",
                     "Tpu_Comms",
                     "Tpu_Return",
                     "Inference",
                     "Total"
                     ])

        for res in results:
            row = []
            for j in range(sessions):
                row.append(res[0 + j*5])  # Host Submission
                row.append(res[1 + j*5])  # Tpu Comms/Comms
                row.append(res[2 + j*5])  # Tpu Return
                row.append(res[3 + j*5])  # Inference
                row.append(res[4 + j*5])  # Total

            fw.writerow(row)

    with open(comms_file, 'w') as csvfile:
        fw = csv.writer(csvfile, delimiter=',', quotechar='|',
                        quoting=csv.QUOTE_MINIMAL)
        i = 0
        for comm in comms:
            fw.writerow([i, comm])
            i += 1

    with open(firsts_file, 'w') as csvfile:
        fw = csv.writer(csvfile, delimiter=',', quotechar='|',
                        quoting=csv.QUOTE_MINIMAL)

        fw.writerow(["Host_Submission",
                     "Tpu_Comms",
                     "Tpu_Return",
                     "Inference",
                     "Total"
                     ])
        for res in first_results:
            row = []
            for j in range(sessions):
                row.append(res[0 + j*5])  # Host Submission
                row.append(res[1 + j*5])  # Tpu Comms/Comms
                row.append(res[2 + j*5])  # Tpu Return
                row.append(res[3 + j*5])  # Inference
                row.append(res[4 + j*5])  # Total
            fw.writerow(row)

def process_usb_results(results, sessions):
    """Processes the array of UsbTimer objects created during the analysis
    of the USB traffic that occurs during deployment.

    Parameters
    ----------
    results : array of UsbTimer Objects.
    
    sessions : Integer
    Number of sessions taking place within the model being analyzed. Only really
    important for mapped models.
    """
    import logging
    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.INFO)

    p_results = []
    p_comms = float(
            results[0].ts_begin_submission - results[0].ts_begin_host_send_request)
    i = 0
    while(i < len(results)):
        tmp_arr = []
        for s in range(sessions):
            res = results[i]
            host_submission = float(
                    res.ts_end_submission - res.ts_begin_submission)
            tpu_comms = float(
                    res.ts_begin_return - res.ts_end_submission)
            tpu_return = float(
                    res.ts_end_return - res.ts_begin_return)
            inference = float(
                    res.ts_begin_return - res.ts_end_submission)
            total_time = float(
                    res.ts_end_return - res.ts_begin_submission)

            # In case there is a single packet being returned
            if tpu_return == 0:
                tpu_return = res.ts_end_return - res.ts_end_tpu_send_request

            tmp = [
                host_submission, tpu_comms, tpu_return, inference, total_time
                ]

            tmp_arr += tmp
            i += 1
        p_results.append(tmp_arr)


    if len(p_results) == 1:
        return p_results, p_results, p_comms
    else:
        return p_results[0], p_results[1:], p_comms



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
            os.system("[ -f temp ] && rm temp")
            raise ValueError(
                "Unable to identify Google Coral Bus and Device address.")

    os.system("[ -f temp ] && rm temp")
    return edge_tpu_id


def usbmon_init():
    """Initializes the usbmon driver module."""
    import os
    usbmon_cmd = "sudo modprobe usbmon"
    os.system(usbmon_cmd)


def docker_deploy_sessions(model, op, count, itr):
    import os
    from docker import HOME, FROM_DOCKER, TO_DOCKER
    from docker import docker_exec, docker_copy
    from utils import retrieve_folder_path, extend_directory
    import time

    cnt = int(count)
    path_to_results = f"results/edge/{op}/Temp.csv"
    path_to_docker_results = HOME + \
        "TensorDSE/benchmarking/usb_analysis/results/"

    if itr == 0:
        extend_directory("results/edge/", f"{op}")

    docker_exec("edge_single_deploy", model, count)
    docker_copy(f"{path_to_docker_results}edge/{op}/Results.csv",
                FROM_DOCKER, path_to_results)


def shark_analyze_stream(edge_tpu_id):
    """Continuouly reads usb packet traffic and retreives the necessary
    timestamps within that cycle.

    This is used in Capture mode, will just read/store a number of packets
    from begin to end of capture. Will not filter anything at all, will then
    print to the terminal a large summary of the packets coming in. Used for
    debugging mostly.

    Parameters
    ---------
    edge_tpu_id : String
    Characterizes the device and bus address where the tpu is mounted on.
    """
    import pyshark
    import time

    usb_array = []

    BEGIN = False
    END = False

    addr = edge_tpu_id.split(".")[1]
    capture_filter = f"usb.transfer_type==URB_BULK || usb.transfer_type==URB_INTERRUPT && usb.device_address=={addr}"
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
        data_is_valid = custom_packet.find_data_validity(raw_packet, edge_tpu_id)

        if (custom_packet.transfer_type == "INTERRUPT"):
            if (BEGIN == False and custom_packet.verify_src(edge_tpu_id, "begin")):
                print("BEGIN")
                BEGIN = True

            elif (BEGIN == True and custom_packet.verify_src(edge_tpu_id, "end")):
                print("END")
                END = True

            usb_array.append(custom_packet)

        elif BEGIN and comms_is_valid:
            usb_array.append(custom_packet)

        if END:
            debug_usb(usb_array)
            break


def shark_analyze_sessions(op, op_filesize, sessions, edge_tpu_id, q):
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

    op_filesize : String
    String that contains the data size of the model that is being deployed in parallel.

    sessions : Integer
    Number of sessions that take place within this model (for mapped models).

    edge_tpu_id : String
    Characterizes the device and bus address where the tpu is mounted on.

    q : Multiprocessing.Queue() Class
    Needed to pass down data containing the raw usb results from this thread to the parent process.
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

    addr = edge_tpu_id.split(".")[1]
    capture_filter = f"usb.transfer_type==URB_BULK || usb.transfer_type==URB_INTERRUPT && usb.device_address=={addr}"

    capture = pyshark.LiveCapture(
        interface='usbmon0', display_filter=capture_filter)

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
                if DEBUG:
                    usb_array.append(custom_packet)

            elif (BEGUN == True and custom_packet.verify_src(edge_tpu_id, "end")):
                ENDED = True
                custom_packet.mark_stamped()
                usb_timer.stamp_ending(raw_packet)
                if DEBUG:
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

                    if DEBUG:
                        usb_array.append(custom_packet)

                # Data packets from host
                if (data_is_present
                        and custom_packet.src == "host"
                        and custom_packet.urb_type == "SUBMIT"):

                    if begin_tpu_send_request == True:
                        begin_tpu_send_request = False

                    # New Session
                    if beginning_of_submission and beginning_of_return:
                        custom_packet.mark_stamped()
                        last_usb_timer = copy.deepcopy(usb_timer)
                        usb_timer = UsbTimer()
                        usb_timer_array.append(last_usb_timer)

                        usb_timer.ts_begin_host_send_request = last_usb_timer.ts_end_return
                        # usb_timer.ts_end_host_send_request = usb_array[-1].ts

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

                    if DEBUG:
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

                    if DEBUG:
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

                    if DEBUG:
                        usb_array.append(custom_packet)

            else:
                custom_packet.mark_foreign
                if DEBUG:
                    usb_array.append(custom_packet)

        if ENDED:
            if DEBUG:
                usb_timer_array.append(usb_timer)
                debug_usb(usb_array)
                debug_stamps(usb_timer_array, sessions)
            else:
                usb_timer_array.append(usb_timer)
                q.put(usb_timer_array)
            break


def shark_single_manager(model, count, edge_tpu_id):
    """Manages the two threads that take care of deploying and listening to edge_tpu.

    This Function is called to manage two simple threads, one will deploy
    a single edge_tpu tflite model a number of times (stream_nr) and the other calls on
    the 'shark_capture_cont' function which will listen on usb traffic and
    retrieve the necessary timestamps.

    Parameters
    ---------
    model : String
    Path to the input tflite model to be analyzed.

    count : Integer
    Number of times the threads will execute. Remember, the actual number of times of
    deployment of a model will be (count * stream_nr).

    edge_tpu_id : String
    The Edge Coral Device identification, is composed by <Bus Address>.<Device Address>.
    """
    import os
    import time
    import logging
    import subprocess
    from multiprocessing import Process, Queue
    from utils import retrieve_folder_path, extend_directory, deduce_operation_from_file
    from utils import create_csv_file, deduce_filename, deduce_filesize
    from utils import deduce_sessions_nr
    from plot import plot_manager
    from docker import copy_project, remove_project
    from docker import copy_project, docker_exec, docker_copy, HOME, TO_DOCKER

    copy_project()

    q = Queue()
    filename = deduce_filename(model)
    filesize = deduce_filesize(model)
    op = deduce_operation_from_file(
            f"{filename}.tflite", beginning="quant_", ending="_edgetpu.tflite")

    sessions = deduce_sessions_nr(op)

    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.INFO)
    log.info(f"Operation {op}: {count}x")

    stream_nr = 2   # Number of times we deploy a model
                    # 2 Should be used for all Mobilenet Models and their custom mapped alternatives.
                    # The rest may use 5 or even 10.

    first_results = []
    results = []
    comms = []
    for i in range(int(count)):
        p_1 = Process(target=shark_analyze_sessions,
                        args=(op, filesize, sessions, edge_tpu_id, q))

        p_2 = Process(target=docker_deploy_sessions,
                        args=(model, op, stream_nr, i))

        p_1.start()
        p_2.start()

        tmp_results = q.get()

        p_2.join()
        p_1.join()

        tmp_first, tmp_results, tmp_comms = process_usb_results(tmp_results, sessions)

        comms.append(tmp_comms)
        results += tmp_results
        if stream_nr == 1: first_results += tmp_results
        else: first_results.append(tmp_first)

        integrate_results(op, i)

    export_analysis(op, filesize, sessions, first_results, results, comms)
    plot_manager(f"{op}_{filesize}", filesize, sessions)
    remove_project()


def shark_manager(folder, count, edge_tpu_id):
    """Takes in a folder path, will then deploy the shark single manager for each
    and one of the models that should be found within this folder.

    Parameters
    ---------
    folder : String
    Path to folder where the to-be-deployed models should be.

    count : Integer
    Number of times the single manager function will execute its child-threads, this
    is just passed down to the shark_single_manager function.

    edge_tpu_id : String
    Characterizes the device and bus address where the tpu is mounted on, is passed
    down onto the shark_single_manager function.
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
    """Entry point to execute this script.

    Flags
    ---------
    -m or --mode
        Mode in which the script should run. Group, Single, Capture or Debug.
        Group is supposed to be used to deploy a group of models in sequence (lazy).
        Single deploys a single model a number of times passed by the '-c' flag.
        Capture will begin capture directly without any filters or manager function,
        useful for debugging, but needs that a separate script executes the deployment
        of a model.
        Debug will turn on the Debug flag and go directly to the usb analysis function,
        to be useful should be done before a separate script gets runned by the user to
        deploy a model onto the edgetpu. Use the deploy.py script.

    -f or --folder
        Should be used in conjunction with the 'Group' mode, where -f must be followed
        by the path to a folder containing all models to be deployed.

    -t or --target
        Should be used in conjunction with the 'Single' mode, where -t must be followed
        by the path to the model (target) that will be deployed.

    -c or --count
        Should be followed by the number of times one wishes to deploy the group of models
        or the single target (Depends on mode).
    """
    import argparse
    from utils import prepare_ulimit, reset_ulimit, deduce_sessions_nr
    from utils import deduce_filename, deduce_filesize, deduce_operations_from_folder
    from docker import docker_start

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-m', '--mode', required=False,
                        default="Group",
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
        prepare_ulimit() # Increases the maximum size limit of files created within execution.
        edge_tpu_id = lsusb_identify()
        docker_start()
        shark_single_manager(args.target, args.count, edge_tpu_id)
        reset_ulimit()

    elif (args.mode == "Capture"):
        usbmon_init()
        edge_tpu_id = lsusb_identify()
        shark_analyze_stream(edge_tpu_id)

    elif (args.mode == "Debug"):
        from multiprocessing import Queue
        q = Queue()
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
                        filename, filesize, sessions, edge_tpu_id, q)

    else:
        print("Invalid arguments.")
