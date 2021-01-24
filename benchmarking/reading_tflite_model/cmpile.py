def retrieve_converted_operations():
    import os
    from docker import CONVERTED_MODELS_DIR
    from os import listdir
    from os.path import isfile, isdir, join, exists
    """From the folder containing all converted-to-tflite models retreives the paths
    to these files and also the corresponding operation names.

    Uses the 'CONVERTED_MODELS_DIR' variable found in the docker.py script to
    know the path to the compiled models first level directoty.

    Returns
    -------
    ops : array of strings
    Containing the names of the operations of these converted tflite models.

    path_ops : array of strings
    Containing the paths to these converted tflite models directories.
    """

    ops = []
    path_ops = []

    for d in listdir(CONVERTED_MODELS_DIR): 
        if (isdir(join(CONVERTED_MODELS_DIR, d))):
            path_to_dir = join(CONVERTED_MODELS_DIR, d)
            path_ops.append(path_to_dir)
            ops.append(d)

    return ops, path_ops


def retrieve_quantized_tflites(ops, path_ops):
    """Retreives the path to the quantized tflite models on host and their
    future paths on the docker to be used for copying and compilation.

    As the quantized tflite models

    Parameters
    ----------
    ops : array
    Array of strings containing the operations names of the to-be-compiled
    models.

    path_ops : array
    Array of strings containing the paths to the folders of the 
    to-be-compiled models.

    Returns
    -------
    quant_sources : array of strings
    Containing the path to the quantized tflite models not yet compiled.

    quant_targets : array of strings
    Containing the 'future' paths of the already edge compiled tflite models.
    """
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
                if isfile(join(quantized_models_dir, q) 
                    and q.startswith(beginning) 
                    and q.endswith(ending)):

                    path_to_quant = join(quantized_models_dir, q)
                    quant_sources.append(path_to_quant)
                    quant_targets.append(f"{HOME}{LOCATION}/{q}")

    return quant_sources, quant_targets
                
def compile_quantized_files_on_dckr(quant_targets):
    """Compiles the quantized tflite models onto the docker.

    Loops through the array of paths to the existent quantized tflite files and
    compiles them individually.

    Parameters
    ----------
    quant_targets : array
    Array of strings containing the paths to each quantized tflite model located
    on the Docker.
    """
    from docker import docker_exec

    for q in quant_targets:
        docker_exec("edgetpu_compiler", q)

def create_folders_dckr():
    """Creates folders necessary for the compilation of the quantized tflite
    models.

    Creates a 'quant' and a 'comp' folder onto the $HOME path of the used
    docker.
    """
    from docker import docker_exec

    docker_exec("mkdir", "quant")
    docker_exec("mkdir", "comp")

def single_tflite_compilation(target):
    """TODO : Not yet implemented -> copy, compile and copy back a single model."""
    # from docker import docker_exec
    # Compiles target on Docker
    # docker_exec("edgetpu_compiler", target)

    raise NotImplementedError

def tflite_compilation():
    """Manager function responsible for preping and executing the compilation
    of the quantized tflite models.

    Starts the docker, reteives the paths and operation names of the quantized
    tflite models, creates necessary folders on the docker, copies quantized
    tflite models from host to docker, compiled them and copies them back.
    """
    from docker import docker_start
    from docker import copy_quantized_files_to_dckr, copy_compiled_files_from_dckr

    docker_start()

    ops, path_ops = retrieve_converted_operations()
    quant_sources, quant_targets = retrieve_quantized_tflites(ops, path_ops)
    
    create_folders_dckr()
    copy_quantized_files_to_dckr(quant_sources)
    compile_quantized_files_on_dckr(quant_targets)
    copy_compiled_files_from_dckr()

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

    if args.mode == "Group":
        tflite_compilation()

    elif args.mode == "Single" and args.target != "":
        single_tflite_compilation(args.target)

    else:
        print("Invalid passed argument combination.")

