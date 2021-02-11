CONVERTED_MODELS_DIR = "models/layers"
TO_DOCKER = 1
FROM_DOCKER = 0

HOME = "/home/deb/"
DOCKER = "exp-docker"
LOCATION = "quant"

count = 1000

def set_docker_globals(cnt):
    """Sets the value of the global variable count.
    
    Is necessary since the docker script's functions may be called from
    different scripts that may need to set this variable to a wished value.

    Parameters
    ---------
    cnt : Integer
    Value used to set a count to a specific iteration, be that the number of
    edge deployments or the number pyshark runs.
    """
    global count
    count = cnt


def docker_start():
    """Starts docker."""
    import logging
    import os

    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.INFO)

    log.info("Starting docker...")
    docker_start_cmd = f"docker start {DOCKER} > /dev/null"
    os.system(docker_start_cmd)


def docker_exec(cmd_type, objct=""):
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
    logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.INFO)

    log.info(f"Executing command: '{cmd_type}' on Docker...")
    
    # Listing relevant command strings.
    mkdir_prefix = f"[ -d {HOME}{objct} ] || "
    docker_exec_prefix = f"docker exec -ti {DOCKER} sh -c "
    edge_compiler_suffix = f"{objct} -o {HOME}/comp"

    cd_deploy_dir = f"cd {HOME}TensorDSE/benchmarking/reading_tflite_model/"
    edge_deploy = f"sudo python3 deploy.py -g True -f models/compiled/ -d edge_tpu -c {count}"
    shark_edge_deploy = f"sudo python3 deploy.py -g True -l False -f models/compiled/ -d edge_tpu -c {count}"
    shark_single_edge_deploy = f"sudo python3 deploy.py -l False -d edge_tpu -c 1 -m {objct}"
    shark_single_edge_deploy_log = f"sudo python3 deploy.py -d edge_tpu -c 1 -m {objct}"
    cpu_deploy = f"sudo python3 deploy.py -g True -f models/layers/ -d cpu -c {count}"
    cpu_single_deploy = f"sudo python3 deploy.py -l False -d cpu -c 1 -m {objct}"

    docker_exec_dict = {
        "mkdir"                         : [docker_exec_prefix, place_within_quotes(f"{mkdir_prefix} mkdir {HOME}{objct}")],
        "edgetpu_compiler"              : [docker_exec_prefix, place_within_quotes(f"edgetpu_compiler -s {edge_compiler_suffix}")],
        "edge_python_deploy"            : [docker_exec_prefix, place_within_quotes(f"{cd_deploy_dir} && {edge_deploy}")],
        "shark_edge_python_deploy"      : [docker_exec_prefix, place_within_quotes(f"{cd_deploy_dir} && {shark_edge_deploy}")],
        "shark_single_edge_deploy"      : [docker_exec_prefix, place_within_quotes(f"{cd_deploy_dir} && {shark_single_edge_deploy}")],
        "shark_single_edge_deploy_log"  : [docker_exec_prefix, place_within_quotes(f"{cd_deploy_dir} && {shark_single_edge_deploy_log}")],
        "cpu_single_deploy"             : [docker_exec_prefix, place_within_quotes(f"{cd_deploy_dir} && {cpu_single_deploy}")],
        "cpu_python_deploy"             : [docker_exec_prefix, place_within_quotes(f"{cd_deploy_dir} && {cpu_deploy}")],
        ""                              : None
    }

    default = None

    # Retreives the command arguments from dictionary.
    args = docker_exec_dict.get(cmd_type, default)

    if(args):
        # Concatenates arguments.
        docker_exec_cmd = concat_args(args)
        os.system(docker_exec_cmd)

def docker_copy(File, direction_flag, Location=""):
    """Copys files/folders to and from the docker.
    
    Is necessary since the docker script's functions may be called from
    different scripts that may need to set this variable to a wished value.

    Parameters
    ---------
    File : String
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
    logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.INFO)

    log.info(f"Docker-Copying file {File} onto Location: {Location}...")

    if(direction_flag):
        docker_copy_cmd = f"docker cp {File} {DOCKER}:{HOME}{Location}"
    else:
        docker_copy_cmd = f"docker cp {DOCKER}:{File} {os.getcwd()}/{Location}"

    os.system(docker_copy_cmd)


def copy_quantized_files_to_dckr(quant_sources):
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
        docker_copy(q, TO_DOCKER, Location=LOCATION + "/")


def copy_compiled_files_from_dckr():
    """Copys edge compiled tflite models/files from docker to the host."""
    import os
    docker_copy(HOME + "/comp", FROM_DOCKER, Location="models/compiled/")
    os.system("cp models/compiled/comp/*edgetpu.tflite models/compiled/")
    os.system("rm -r models/compiled/comp/")
