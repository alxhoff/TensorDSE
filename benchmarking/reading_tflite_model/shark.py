class UsbTimer:
    def __init__(self):
        pass

    time_elapsed = 0
    initial_time = 0
    final_time = 0

    ts_host_first = None
    ts_host_last = None

    ts_edge_first = None
    ts_edge_last = None

    def obtain_first_host_pkt(self, cap):
        for c in cap:
            if (c['USB'].src == "host"):
                self.ts_host_first = c['USB'].urb_ts_sec
                break

    def obtain_last_host_pkt(self, cap):
        for c in reversed(cap):
            if (c['USB'].src == "host"):
                self.ts_host_first = c['USB'].urb_ts_sec
                break

    def obtain_first_edge_pkt(self, cap):
        for c in cap:
            if (not c['USB'].src == "host"):
                self.ts_edge_first = c['USB'].urb_ts_sec
                break

    def obtain_last_edge_pkt(self, cap):
        for c in reversed(cap):
            if (not c['USB'].src == "host"):
                self.ts_edge_last = c['USB'].urb_ts_sec
                break

    def obtain_total_time_elapsed(self, cap):
        lgth = len(cap)
        self.initial_time = cap[0]['USB'].urb_ts_sec
        self.final_time = cap[lgth - 1]['USB'].urb_ts_sec
        self.time_elapsed = self.final_time - self.initial_time

def prep_capture_file():
    import os
    cap_file = "/home/duclos/Documents/work/TensorDSE/shark/capture.cap"
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


def shark_usbmon_init():
    import os
    usbmon_cmd = "sudo modprobe usbmon"
    os.system(usbmon_cmd)


def shark_capture_init(timeout, op_name):
    import asyncio
    import pyshark

    prep_capture_file()
    out_file = "shark/" + op_name + "capture.cap"

    capture = pyshark.LiveCapture(interface='usbmon0', output_file=out_file,
                                  include_raw=True, use_json=True)

    #capture.set_debug()  # Toggle comment to toggle Debug mode of TShark.

    #try:
    capture.sniff(timeout=timeout)

    #except asyncio.exceptions.TimeoutError:  # Pyshark is a bit broken.
    #    pass

    capture.close()


def shark_read_captures():
    import os
    import pyshark
    from os import listdir
    from deploy import deduce_operation_from_file

    capture_dir = "shark/"

    for capture_file in listdir(capture_dir):

        op = deduce_operation_from_file(capture_file, ending="_capture.cap")

        cap = pyshark.FileCapture(capture_dir + capture_file)
        raw_cap = cap[0].get_raw_packet()

        usb_timer = UsbTimer() 
        usb_timer.obtain_total_time_elapsed(cap)

        usb_timer.obtain_first_host_pkt(cap)
        usb_timer.obtain_last_host_pkt(cap)

        usb_timer.obtain_first_edge_pkt(cap)
        usb_timer.obtain_last_edge_pkt(cap)

        cap.close()

        export_analysis(usb_timer, op)


def shark_manager(folder):
    import os
    import threading
    import pyshark
    import time
    from docker import TO_DOCKER, FROM_DOCKER, home, docker_exec, docker_copy
    from deploy import deduce_operations_from_folder
    from  utils import retrieve_folder_path

    path_to_tensorDSE = retrieve_folder_path(os.getcwd(), "TensorDSE")
    docker_copy(path_to_tensorDSE, TO_DOCKER)

    models_info = deduce_operations_from_folder(folder, 
                                                beginning="quant_", 
                                                ending = "_edgetpu.tflite")

    for m_i in models_info:
        t1 = threading.Thread(target=shark_capture_init, args=(10, m_i[1] + "_"))
        t2 = threading.Thread(target=docker_exec, args=("shark_single_edge_deploy", m_i[0],))

        t1.start()
        t2.start()

        start = time.perf_counter()
        while (t1.is_alive() or t2.is_alive()):
            pass
            
        end = time.perf_counter()
        time = end - start

        t1.join()
        t2.join()

    shark_read_captures()


def export_analysis(usb_timer, op):
    from utils import extend_directory 

    out_dir = "results/shark/" 
    extend_directory(out_dir, op)
    extended_dir = out_dir + op
    
    results_file = extended_dir + "Results.txt"

    with open(results_file, 'w') as f:
        f.write("Operation Name: " + str(op) + "\n")
        f.write("Operation Name: " + str(op) + "\n")
        f.write("Operation Name: " + str(op) + "\n")
        f.write("Operation Name: " + str(op) + "\n")
        f.write("Operation Name: " + str(op) + "\n")


if __name__ == '__main__':
    from docker import docker_start
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-c', '--count', required=False, 
                        default=1000, 
                        help='Count of the number of times of edge deployment.')

    parser.add_argument('-m', '--mode', required=False, 
                        default="Both", 
                        help='Mode in which the script will run: All, Read, Capture or Deploy.')

    parser.add_argument('-f', '--folder', required=False, 
                        default="", 
                        help='Folder.')

    args = parser.parse_args()

    if (args.mode == "Capture"):
        shark_usbmon_init()
        shark_capture_init()

    elif(args.mode == "Deploy"):
        docker_start()
        shark_deploy_edge(args.count)

    elif (args.mode == "Read"):
        shark_read_captures()

    elif (args.mode == "All" and args.folder != ""):
        shark_usbmon_init()
        docker_start()
        shark_manager(args.folder)

    else:
        print("Invaild arguments.")

