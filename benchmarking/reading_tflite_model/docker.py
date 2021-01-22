CONVERTED_MODELS_DIR = "models/single_layer_models"
TO_DOCKER = 1
FROM_DOCKER = 0

HOME = "/home/deb/"
DOCKER = "exp-docker"
LOCATION = "quant"

count = 1000

def set_docker_globals(cnt):
    global count
    count = cnt


def docker_start():
    import os

    print("Starting docker...")
    docker_start_cmd = f"docker start {DOCKER}"
    os.system(docker_start_cmd)


def docker_exec(cmd_type, objct=""):
    import os
    from utils import place_within_quotes, concat_args

    print(f"Executing command: '{cmd_type}' on Docker...")

    mkdir_prefix = f"[ -d {HOME}{objct} ] || "
    docker_exec_prefix = f"docker exec -ti {DOCKER} sh -c "
    edge_compiler_suffix = f"{objct} -o {HOME}/comp"

    cd_deploy_dir = f"cd {HOME}TensorDSE/benchmarking/reading_tflite_model/"
    edge_deploy = f"sudo python3 deploy.py -g True -f models/tpu_compiled_models/ -d edge_tpu -c {count}"
    shark_edge_deploy = f"sudo python3 deploy.py -g True -l False -f models/tpu_compiled_models/ -d edge_tpu -c {count}"
    shark_single_edge_deploy = f"sudo python3 deploy.py -l False -d edge_tpu -c 1 -m {objct}"
    cpu_deploy = f"sudo python3 deploy.py -g True -f models/single_layer_models/ -d cpu -c {count}"

    docker_exec_dict = {
        "mkdir"                     : [docker_exec_prefix, place_within_quotes(f"{mkdir_prefix} mkdir {HOME}{objct}")],
        "edgetpu_compiler"          : [docker_exec_prefix, place_within_quotes(f"edgetpu_compiler -s {edge_compiler_suffix}")],
        "edge_python_deploy"        : [docker_exec_prefix, place_within_quotes(f"{cd_deploy_dir} && {edge_deploy}")],
        "shark_edge_python_deploy"  : [docker_exec_prefix, place_within_quotes(f"{cd_deploy_dir} && {shark_edge_deploy}")],
        "shark_single_edge_deploy"  : [docker_exec_prefix, place_within_quotes(f"{cd_deploy_dir} && {shark_single_edge_deploy}")],
        "cpu_python_deploy"         : [docker_exec_prefix, place_within_quotes(f"{cd_deploy_dir} && {cpu_deploy}")],
        ""                          : None
    }

    default = None
    args = docker_exec_dict.get(cmd_type, default)

    if(args):
        docker_exec_cmd = concat_args(args)
        os.system(docker_exec_cmd)

def docker_copy(File, DIRECTION_FLAG, Location = ""):
    import os
   
    print(f"Docker-Copying file {File} onto Location: {Location}...")
    if(DIRECTION_FLAG):
        docker_copy_cmd = f"docker cp {File} {DOCKER}:{HOME}{Location}"
    else:
        docker_copy_cmd = f"docker cp {DOCKER}:{File} {os.getcwd()}/{Location}"

    os.system(docker_copy_cmd)


def copy_quantized_files_to_dckr(quant_sources):
    for q in quant_sources:
        docker_copy(q, TO_DOCKER, Location=LOCATION + "/")


def copy_quantized_files_from_dckr():
    import os
    docker_copy(HOME + "/comp", FROM_DOCKER, Location="models/tpu_compiled_models/")
    os.system("cp models/tpu_compiled_models/comp/*edgetpu.tflite models/tpu_compiled_models/")
    os.system("rm -r models/tpu_compiled_models/comp/")
