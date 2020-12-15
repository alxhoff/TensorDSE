EDGE_TPU_ID = ""


class UsbTimer:
    def __init__(self):
        pass

    ts_absolute_begin = 0
    ts_begin_edge2host = 0
    ts_absolute_end = 0


def prep_capture_file():
    import os
    cap_file = "/home/duclos/Documents/work/TensorDSE/shark/capture.pcap"
    os.system("[ -f " + cap_file + " ] || touch " + cap_file)

def shark_deploy_edge(count, objct):
    import os
    import utils
    import docker
    from docker import TO_DOCKER, FROM_DOCKER, home
    
    path_to_tensorDSE = utils.retrieve_folder_path(os.getcwd(), "TensorDSE")
    docker.docker_copy(path_to_tensorDSE, TO_DOCKER)

    docker.set_globals(count)
    docker.docker_exec("shark_edge_python_deploy")


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

    global EDGE_TPU_ID
    
    lsusb_cmd = "lsusb | grep Google > temp.txt"
    os.system(lsusb_cmd)
    with open("temp.txt", "r") as f:
        for line in f.readlines():
            if "Google" in line:
                bus = int(line.split()[1])
                device = int((line.split()[3]).split(":")[0])
                EDGE_TPU_ID = str(bus) + "." + str(device)

    os.system("[ -f temp.txt ] && rm temp.txt")


def shark_usbmon_init():
    import os
    usbmon_cmd = "sudo modprobe usbmon"
    os.system(usbmon_cmd)


def shark_capture_init(timeout, op_name):
    import os
    import asyncio
    import pyshark

    prep_capture_file()
    out_file = "shark/" + op_name + "capture.pcap"

    param_dict = {
            "--param" : "value",
            "--param" : "value"
    }

    capture = pyshark.LiveCapture(interface='usbmon0', output_file=out_file, include_raw=True)
    
    try:
        capture.sniff(timeout=timeout)

    except asyncio.exceptions.TimeoutError:  # Pyshark is a bit broken.
        pass

    capture.close()


def shark_read_captures():
    import os
    import pyshark
    from os import listdir
    from deploy import deduce_operation_from_file

    capture_dir = "shark/"
    read_filter = "usb.transfer_type==URB_BULK"

    for capture_file in listdir(capture_dir):

        usb_timer = UsbTimer() 
        cap_file_path = capture_dir + capture_file
        op = deduce_operation_from_file(capture_file, ending="_capture.cap")

        capture = pyshark.FileCapture(input_file=cap_file_path, 
                                      display_filter=read_filter)

        host_first = 0
        edge_first = 0

        try:
            for pkt in capture:
                if ("host" in pkt.usb.src or EDGE_TPU_ID in pkt.usb.src):
                    if ("host" in pkt.usb.src):
                        if (host_first == 0):
                            usb_timer.ts_absolute_begin = pkt.frame_info.time_relative
                            host_first = 1
                        
                        if (edge_first == 0):
                            usb_timer.begin_inference = pkt.frame_info.time_relative

                    if (EDGE_TPU_ID in pkt.usb.src):
                        if (edge_first == 0):
                            usb_timer.ts_begin_edge2host = pkt.frame_info.time_relative
                            edge_first = 1

                    usb_timer.ts_absolute_end = pkt.frame_info.time_relative

        except RuntimeError:
            pass
        except pyshark.capture.capture.TSharkCrashException:
            pass

        try:
            capture.close()

        except RuntimeError:
            pass
        except pyshark.capture.capture.TSharkCrashException:
            pass

        export_analysis(usb_timer, op)


def shark_manager(folder):
    import os
    import threading
    import pyshark
    from docker import TO_DOCKER, FROM_DOCKER, home, docker_exec, docker_copy
    from deploy import deduce_operations_from_folder
    from  utils import retrieve_folder_path

    path_to_tensorDSE = retrieve_folder_path(os.getcwd(), "TensorDSE")
    docker_copy(path_to_tensorDSE, TO_DOCKER)

    models_info = deduce_operations_from_folder(folder, 
                                                beginning="quant_", 
                                                ending = "_edgetpu.tflite")

    for m_i in models_info:
        import time

        t1 = threading.Thread(target=shark_capture_init, args=(15, m_i[1] + "_"))
        t2 = threading.Thread(target=docker_exec, args=("shark_single_edge_deploy", m_i[0],))

        t1.start()
        t2.start()

        while (t1.is_alive() or t2.is_alive()):
           pass 
            

        t1.join()
        t2.join()

        time.sleep(2)

    # shark_read_captures()


def export_analysis(usb_timer, op):
    import csv
    from utils import extend_directory 

    out_dir = "results/shark/" 
    extend_directory(out_dir, op)
    extended_dir = out_dir + op
    results_file = extended_dir + "/Results.csv"

    with open(results_file, 'w') as csvfile:
        fw = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        fw.writerow([usb_timer.ts_absolute_begin, usb_timer.ts_absolute_end,
                    usb_timer.ts_absolute_end, usb_timer.ts_absolute_end])


if __name__ == '__main__':
    from docker import docker_start
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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
        shark_deploy_edge(args.count)

    elif (args.mode == "Read"):
        lsusb_identify()
        shark_read_captures()

    elif (args.mode == "All" and args.folder != ""):
        shark_usbmon_init()
        lsusb_identify()
        docker_start()
        shark_manager(args.folder)

    else:
        print("Invaild arguments.")

