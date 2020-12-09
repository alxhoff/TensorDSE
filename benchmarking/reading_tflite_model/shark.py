
def shark_deploy_edge(count):
    import os
    import utils
    import compile
    from compile import TO_DOCKER, FROM_DOCKER, home
    
    path_to_tensorDSE = utils.retrieve_folder_path(os.getcwd(), "TensorDSE")
    path_to_docker_results =  home + "TensorDSE/benchmarking/reading_tflite_model/results/"

    compile.set_globals(count)
    compile.docker_copy(path_to_tensorDSE, TO_DOCKER)
    compile.docker_exec("shark_edge_python_deploy")


def shark_usbmon_init():
    import os
    usbmon_cmd = "sudo modprobe usbmon"
    os.system(usbmon_cmd)


def shark_capture_init():
    import asyncio
    import pyshark

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

    print(cap[0]['USB'])
    #print(cap[0]['USB'].src)
    #print(dir(cap[0]['USB'])

    with open(out_file, 'wt') as out:
        pprint(dir(cap[0]['USB']), stream=out)

    cap.close()


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
