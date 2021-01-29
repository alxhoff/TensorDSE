import argparse
import os
from shark import shark_capture_cont, lsusb_identify, shark_usbmon_init
from utils import deduce_operations_from_folder
from docker import TO_DOCKER, FROM_DOCKER, HOME, docker_exec, docker_copy, docker_start
from utils import retrieve_folder_path

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-m', '--mode', required=True,
                    default="",
                    help='Mode/Functionality')

parser.add_argument('-f', '--folder', required=False,
                    default="",
                    help='Folder')

args = parser.parse_args()

if args.mode != "":
    if (args.mode == "Deploy"):
        # shark_usbmon_init()
        edge_tpu_id = lsusb_identify()
        docker_start()

        inp = input("Wait: ")

        models_info = deduce_operations_from_folder(args.folder,
                                                    beginning="quant_",
                                                    ending="_edgetpu.tflite")

        path_to_tensorDSE = retrieve_folder_path(os.getcwd(), "TensorDSE")
        docker_copy(path_to_tensorDSE, TO_DOCKER)

        for m_i in models_info:
            docker_exec("shark_single_edge_deploy", m_i[0])
            inp = input("Wait: ")
        
    elif (args.mode == "Capture"):
        edge_tpu_id = lsusb_identify()
        models_info = deduce_operations_from_folder("models/compiled/",
                                                    beginning="quant_",
                                                    ending="_edgetpu.tflite")
        for i in range(3):
            for m_i in models_info:
                print(f"Operation {m_i[1]}")
                shark_capture_cont(m_i[1], i, edge_tpu_id)
                inp = input("Wait: ")

    elif (args.mode == "CSV"):
        from shark import export_analysis, UsbTimer

        example = UsbTimer()
        example.ts_absolute_begin = 4
        example.ts_absolute_end = 4

        cnt = 5
        for i in range(cnt):
            export_analysis(example, "CONV_2D", i!=0)
else:
    print("Incorrect passed arguments.")
