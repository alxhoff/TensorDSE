class UsbTimer:
    def __init__(self):
        pass

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

def prep_capture_file():
    import os
    cap_file = "/home/duclos/Documents/work/TensorDSE/shark/capture.cap"
    os.system("[ -f " + cap_file + " ] || touch " + cap_file)

def shark_deploy_edge(count):
    import os
    import utils
    import docker
    from docker import TO_DOCKER, FROM_DOCKER, home
    
    path_to_tensorDSE = utils.retrieve_folder_path(os.getcwd(), "TensorDSE")
    path_to_docker_results =  home + "TensorDSE/benchmarking/reading_tflite_model/results/"

    docker.set_globals(count)
    docker.docker_copy(path_to_tensorDSE, TO_DOCKER)
    docker.docker_exec("shark_edge_python_deploy")


def shark_usbmon_init():
    import os
    usbmon_cmd = "sudo modprobe usbmon"
    os.system(usbmon_cmd)


def shark_capture_init():
    import asyncio
    import pyshark

    prep_capture_file()
    out_file = "/home/duclos/Documents/work/TensorDSE/shark/capture.cap"
    capture = pyshark.LiveCapture(interface='usbmon0', output_file=out_file)
    capture.set_debug()  # Comment this to turn off Debug mode of TShark.

    try:
        capture.sniff(timeout=100)

    except asyncio.exceptions.TimeoutError:  # Pyshark is a bit broken.
        pass

    capture.close()


def shark_read_capture():
    import pyshark
    from pprint import pprint

    in_file = "/home/duclos/Documents/work/TensorDSE/shark/capture.cap"
    out_file = "/home/duclos/Documents/work/TensorDSE/shark/attributes.txt"

    cap = pyshark.FileCapture(in_file)

    usb_timer = UsbTimer() 

    usb_timer.obtain_total_time_elapsed(cap)

    usb_timer.obtain_first_host_pkt(cap)
    usb_timer.obtain_last_host_pkt(cap)

    usb_timer.obtain_first_edge_pkt(cap)
    usb_timer.obtain_last_edge_pkt(cap)

    cap.close()

    export_analysis(usb_timer)


def shark_manager(count):
    import threading
    import pyshark

    t1 = threading.Thread(target=shark_capture_init, args=())
    t2 = threading.Thread(target=shark_deploy_edge, args=(count,))

    t1.start()
    t2.start()

    while (t1.is_alive() or t2.is_alive()):
        pass

    t1.join()
    t2.join()

    shark_read_capture()


def export_analysis():
    pass


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-c', '--count', required=False, 
                        default=1000, 
                        help='Count of the number of times of edge deployment.')

    parser.add_argument('-m', '--mode', required=False, 
                        default="Both", 
                        help='Mode in which the script will run: All, Read, Capture or Deploy.')

    args = parser.parse_args()

    if (args.mode == "Capture"):
        shark_usbmon_init()
        shark_capture_init()

    elif(args.mode == "Deploy"):
        shark_deploy_edge(args.count)

    elif (args.mode == "Read"):
        shark_read_capture()

    else:
        shark_manager(args.count)
