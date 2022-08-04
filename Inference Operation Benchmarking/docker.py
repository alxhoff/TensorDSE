CONVERTED_MODELS_DIR = "models/layers"
TO_DOCKER = 1
FROM_DOCKER = 0

HOME = "/home/deb/"
LOCATION = "quant"


def DockerStart(docker="debian-docker"):
    
    """Starts docker."""
    import logging
    import os

    log = logging.getLogger(__name__)
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

    log.info("Starting docker...")
    docker_start_cmd = f"docker start {docker} > /dev/null"
    os.system(docker_start_cmd)


def DockerExec(cmd_type, objct="", count=1, docker="debian-docker"):
    """Executes a command onto the docker.

    Is somewhat the center of docker.py, since compiling and deploying tflite
    models onto the coral edge tpu are of interest. The software necessary is
    mostly readily available on Debian-native systems, where then the docker
    comes into place.

    This function receives a simple string, which is nothing other than a
    pre-defined 'word' that matches onto a dictionary who returns the
    corresponding command, also in a string form. This command is then simply
    run on the underlying shell using the 'os.system' method.

    Parameters
    ----------
    cmd_type : String
    Value used match onto the dictionary of commands below and retreive the
    necessary command.

    objct : String, default=''
    Some commands may need an object or target, such that this functions remains
    mosty generic.


    Examples
    --------
    >>> docker_exec('mkdir', 'folder')
    # Executes the following into the shell:
    docker exec -ti {DOCKER} sh -c [ -d {HOME}{objct} ] || mkdir {HOME}{objct}

    # docker exec -ti {DOCKER} sh -c   +    [ -d {HOME}{objct} ] || mkdir {HOME}{objct}
    #
    # docker prefix command necessary       Actual command run onto the docker,
    # to run shell commands onto the        which in this case is a simple make
    # docker.                               directory command preceded by a
    #                                       sh/bash conditional check to first
    #                                       if the directory already exists.
    #                                       If it doesnt, only then will it be
    #                                       created.
    """
    import os
    import logging
    from utils import place_within_quotes, concat_args

    log = logging.getLogger(__name__)
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    log.info(f"Executing command: '{cmd_type}' on Docker...")

    # Listing relevant command strings.
    mkdir_prefix = f"[ -d {HOME}{objct} ] || "
    rm_prefix = f"[ -d {HOME}{objct} ] && "
    docker_exec_prefix = f"docker exec -it {docker} sh -c "
    edge_compiler_suffix = f"{objct} -o {HOME}/comp"

    cd_deploy_dir = f"cd {HOME}TensorDSE/benchmarking/usb_analysis/"
    edge_deploy = f"sudo python3 deploy.py -m Group -f {objct} -d edge_tpu -c {count}"
    cpu_deploy = f"sudo python3 deploy.py -m Group -f {objct} -d cpu -c {count}"
    edge_single_deploy = (
        f"sudo python3 deploy.py -m Single -d edge_tpu -c {count} -t {objct}"
    )
    cpu_single_deploy = f"sudo python3 deploy.py -m Single -d cpu -c {count} -t {objct}"

    docker_exec_dict = {
        "mkdir": [
            docker_exec_prefix,
            place_within_quotes(f"{mkdir_prefix} mkdir {HOME}{objct}"),
        ],
        "remove": [
            docker_exec_prefix,
            place_within_quotes(f"{rm_prefix} sudo rm -rf {HOME}{objct}"),
        ],
        "edge_compile": [
            docker_exec_prefix,
            place_within_quotes(f"edgetpu_compiler -s {edge_compiler_suffix}"),
        ],
        "edge_deploy": [
            docker_exec_prefix,
            place_within_quotes(f"{cd_deploy_dir} && {edge_deploy}"),
        ],
        "cpu_deploy": [
            docker_exec_prefix,
            place_within_quotes(f"{cd_deploy_dir} && {cpu_deploy}"),
        ],
        "edge_single_deploy": [
            docker_exec_prefix,
            place_within_quotes(f"{cd_deploy_dir} && {edge_single_deploy}"),
        ],
        "cpu_single_deploy": [
            docker_exec_prefix,
            place_within_quotes(f"{cd_deploy_dir} && {cpu_single_deploy}"),
        ],
        "": None,
    }

    default = None
    args = docker_exec_dict.get(cmd_type, default)

    assert args != None, "Incorrect docker exec command given."
    if args:
        # Concatenates arguments.
        docker_exec_cmd = concat_args(args)
        log.info(f"Running '{docker_exec_cmd}' on Docker...")
        os.system(docker_exec_cmd)


def DockerCopyFileToDocker(file, direction_flag, Location="", docker="debian-docker"):
    """Copys files/folders to and from the docker.

    Is necessary since the docker script's functions may be called from
    different scripts that may need to set this variable to a wished value.

    Parameters
    ---------
    file : String
    File or Folder path.

    direction_flag : Integer (0 or 1)
    Flag denoting direction of copy.
    1 - From host to Docker.
    0 - From Docker to host.

    Location : String, default='' (Empty String)
    Path to where the to-be-copied files must land.
    """
    import os
    import logging

    log = logging.getLogger(__name__)
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

    log.info(f"Docker-Copying file {file} onto Location: {Location}...")

    if direction_flag:
        docker_copy_cmd = f"docker cp {file} {docker}:{HOME}{Location}"
    else:
        docker_copy_cmd = f"docker cp {docker}:{file} {os.getcwd()}/{Location}"

    os.system(docker_copy_cmd)


def DockerCopyQuanModelsToDocker(quant_sources):
    """Copys quantized tflite models/files to the docker.

    This is done as a pre-step to compiling these quantized files onto the
    docker.

    Parameters
    ----------
    quant_sources : array
    Contains strings describing the paths to the quantized tflite models.
    [ path_to_quantized_tflite_file_1, path_to_quantized_tflite_file_2, .. ]
    """
    for q in quant_sources:
        DockerCopyFileToDocker(q, TO_DOCKER, Location=LOCATION + "/")


def DockerCopyCompiledModelsFromDocker():
    """Copys edge compiled tflite models/files from docker to the host."""
    import os

    DockerCopyFileToDocker(HOME + "/comp", FROM_DOCKER, Location="models/compiled/")
    os.system("cp models/compiled/comp/*edgetpu.tflite models/compiled/")
    os.system("rm -r models/compiled/comp/")


def copy_project():
    """Copys project folder to docker."""
    import os
    from utils import retrieve_folder_path

    path_to_tensorDSE = retrieve_folder_path(os.getcwd(), "TensorDSE")
    DockerCopyFileToDocker(path_to_tensorDSE, TO_DOCKER)


def CleanUpProject():
    """Removes project from docker."""
    DockerExec("remove", "TensorDSE")
