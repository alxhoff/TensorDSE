def retrieve_converted_operations():
    import os
    from docker import CONVERTED_MODELS_DIR
    from os import listdir
    from os.path import isfile, isdir, join, exists

    ops = []
    path_ops = []

    for d in listdir(CONVERTED_MODELS_DIR): 
        if (isdir(join(CONVERTED_MODELS_DIR, d))):
            path_to_dir = join(CONVERTED_MODELS_DIR, d)
            path_ops.append(path_to_dir)
            ops.append(d)

    return ops, path_ops


def retrieve_quantized_tflites(ops, path_ops):
    import os
    from os import listdir
    from os.path import isfile, join, exists
    from docker import HOME, LOCATION

    quant_sources = []
    quant_targets = []

    subfolder = "quant/"
    beginning = "quant"
    ending = ".tflite"

    for op, op_path in zip(ops, path_ops):
        quantized_models_dir = op_path + "/quant/"

        if(os.path.exists(quantized_models_dir)):
            for q in listdir(quantized_models_dir):
                if (isfile(join(quantized_models_dir, q)) and q.startswith(beginning) and q.endswith(ending)):
                    path_to_quant = join(quantized_models_dir, q)
                    quant_sources.append(path_to_quant)
                    quant_targets.append(f"{HOME}{LOCATION}/{q}")

    return quant_sources, quant_targets
                
def compile_quantized_files_on_dckr(quant_targets):
    from docker import docker_exec

    for q in quant_targets:
        docker_exec("edgetpu_compiler", q)

def create_folders_dckr():
    from docker import docker_exec

    docker_exec("mkdir", "quant")
    docker_exec("mkdir", "comp")

def single_tflite_compilation(target):
    from docker import docker_exec

    # Compiles target on Docker
    docker_exec("edgetpu_compiler", target)

def tflite_compilation():
    from docker import docker_start
    from docker import copy_quantized_files_to_dckr, copy_quantized_files_from_dckr

    docker_start()

    ops, path_ops = retrieve_converted_operations()
    quant_sources, quant_targets = retrieve_quantized_tflites(ops, path_ops)
    
    create_folders_dckr()
    copy_quantized_files_to_dckr(quant_sources)
    compile_quantized_files_on_dckr(quant_targets)
    copy_quantized_files_from_dckr()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-m', '--mode', required=False, 
                        default="Group", 
                        help='Compilation mode.')

    parser.add_argument('-t', '--target', required=False, 
                        default="", 
                        help='Target file to be compiled in case of single mode \
                        compilation.')

    args = parser.parse_args()

    if (args.mode == "Group"):
        tflite_compilation()
    elif (args.mode == "Single" and args.target != ""):
        single_tflite_compilation(args.target)
    else:
        print("Invalid passed argument combination.")

