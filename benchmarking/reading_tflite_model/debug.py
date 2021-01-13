import argparse
from shark import shark_capture_cont
from deploy import deduce_operations_from_folder
from docker import docker_exec

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

        inp = input("Continue [y/n]")

        models_info = deduce_operations_from_folder(args.folder,
                                                    beginning="quant_",
                                                    ending="_edgetpu.tflite")

        for m_i in models_info:
            docker_exec("shark_single_edge_deploy", m_i[0])
            inp = input("Continue [y/n]: ")

            if (inp != "y"):
                break
        
    elif (args.mode == "Capture"):
        shark_capture_cont()
else:
    print("Incorrect passed arguments.")
