def GetOperationsAndPaths():

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
    import os
    from docker import CONVERTED_MODELS_DIR
    from os import listdir
    from os.path import isdir, join

    operations = []
    operation_paths = []

    for d in listdir(CONVERTED_MODELS_DIR):
        if isdir(join(CONVERTED_MODELS_DIR, d)):
            path_to_dir = join(CONVERTED_MODELS_DIR, d)
            operation_paths.append(path_to_dir)
            operations.append(d)

    return operations, operation_paths


def GetModelCopyPaths(ops, path_ops):

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

        if os.path.exists(quantized_models_dir):
            for q in listdir(quantized_models_dir):
                if (
                    isfile(join(quantized_models_dir, q))
                    and q.startswith(beginning)
                    and q.endswith(ending)
                ):

                    path_to_quant = join(quantized_models_dir, q)
                    quant_sources.append(path_to_quant)
                    quant_targets.append(f"{HOME}{LOCATION}/{q}")

    return quant_sources, quant_targets


def DockersCompileQuantizedModels(quant_targets):

    """Compiles the quantized tflite models onto the docker.

    Loops through the array of paths to the existent quantized tflite files and
    compiles them individually.

    Parameters
    ----------
    quant_targets : array
    Array of strings containing the paths to each quantized tflite model located
    on the Docker.
    """
    from docker import DockerExec

    for q in quant_targets:
        DockerExec("edge_compile", q)


def DockerCreateCompilationDirs():

    """Creates folders necessary for the compilation of the quantized tflite
    models.

    Creates a 'quant' and a 'comp' folder onto the $HOME path of the used
    docker.
    """
    from docker import DockerExec

    DockerExec("mkdir", "quant")
    DockerExec("mkdir", "comp")


def single_tflite_compilation(target, target_filename):
    """"""
    from docker import HOME
    from docker import TO_DOCKER, FROM_DOCKER
    from docker import DockerExec, DockerCopyFileToDocker

    docker_compiled_file = f"{HOME}comp/{target_filename}"
    docker_copied_file = f"{HOME}{target_filename}"

    DockerCopyFileToDocker(target, TO_DOCKER)
    DockerExec("edgetpu_compiler", docker_copied_file)
    DockerCopyFileToDocker(docker_compiled_file, FROM_DOCKER, "models/compiled/")


def CompileTFLiteModelsForCoral():

    """Manager function responsible for preping and executing the compilation
    of the quantized tflite models.

    Starts the docker, reteives the paths and operation names of the quantized
    tflite models, creates necessary folders on the docker, copies quantized
    tflite models from host to docker, compiles them and copies them back.
    """
    from docker import DockerStart
    from docker import DockerCopyQuanModelsToDocker, DockerCopyCompiledModelsFromDocker

    DockerStart()

    operations, operation_paths = GetOperationsAndPaths()
    quant_sources, quant_targets = GetModelCopyPaths(operations, operation_paths)

    DockerCreateCompilationDirs()
    DockerCopyQuanModelsToDocker(quant_sources)
    DockersCompileQuantizedModels(quant_targets)
    DockerCopyCompiledModelsFromDocker()


if __name__ == "__main__":

    """Entry point to execute this script.

    Flags
    ---------
    -m or --mode
    Mode in which the script should run. Group or Single.
        Group is supposed to be used to compile a group of models in sequence(lazy).
        Single compiles a single model.

    -t or --target
        Should be used in conjunction with the 'Single' mode, where -t must be followed
        by the path to the model (target) that will be deployed.
    """
    from utils import deduce_filename
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-m", "--mode", required=False, default="Group", help="Compilation mode."
    )

    parser.add_argument(
        "-t",
        "--target",
        required=False,
        default="",
        help="Target file to be compiled in case of single mode \
                        compilation.",
    )

    args = parser.parse_args()

    if args.mode == "Group":
        CompileTFLiteModelsForCoral()

    elif args.mode == "Single" and args.target != "":
        model_name = deduce_filename(args.target)
        single_tflite_compilation(args.target, f"{model_name}.tflite")

    else:
        print("Invalid passed argument combination.")
