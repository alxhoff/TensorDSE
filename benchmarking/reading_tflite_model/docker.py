CONVERTED_MODELS_DIR = "models/single_layer_models"
TO_DOCKER = 1
FROM_DOCKER = 0

home = "/home/deb/"
count = 1000
docker = "exp-docker"
location = "quant"

ops = []
path_ops = []
quant_targets = []
quant_sources = []
compiled_sources = []

def set_globals(cnt):
    global count
    count = cnt


def place_within_quotes(string):
    from shlex import quote
    return "".join(quote(string))


def concat_args(args):
    summed_args = ""
    for arg in range(len(args)):
        summed_args += args[arg]
    return summed_args


def docker_start():
    import os

    print("Starting docker...")
    docker_start_cmd = f"docker start {docker}"
    os.system(docker_start_cmd)


def docker_exec(cmd_type, objct=""):
    import os

    print(f"Executing command: '{cmd_type}' on Docker...")

    mkdir_prefix = f"[ -d {home}{objct} ] || "
    docker_exec_prefix = f"docker exec -ti {docker} sh -c "
    edge_compiler_suffix = f"{objct} -o {home}/comp"

    cd_deploy_dir = f"cd {home}TensorDSE/benchmarking/reading_tflite_model/"
    edge_deploy = f"sudo python3 deploy.py -g True -f models/tpu_compiled_models/ -d edge_tpu -c {count}"
    shark_edge_deploy = f"sudo python3 deploy.py -g True -l False -f models/tpu_compiled_models/ -d edge_tpu -c {count}"
    shark_single_edge_deploy = f"sudo python3 deploy.py -l False -d edge_tpu -c 1 -m {objct}"
    cpu_deploy = f"sudo python3 deploy.py -g True -f models/single_layer_models/ -d cpu -c {count}"

    docker_exec_dict = {
        "mkdir"                     : [docker_exec_prefix, place_within_quotes(f"{mkdir_prefix} mkdir {home}{objct}")],
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
        docker_copy_cmd = f"docker cp {File} {docker}:{home}{Location}"
    else:
        docker_copy_cmd = f"docker cp {docker}:{File} {os.getcwd()}/{Location}"

    os.system(docker_copy_cmd)

def compile_quantized_files_on_dckr():
    for q in quant_targets:
        docker_exec("edgetpu_compiler", q)

def copy_quantized_files_to_dckr():
    for q in quant_sources:
        docker_copy(q, TO_DOCKER, Location=location + "/")

def copy_quantized_files_from_dckr():
    import os
    docker_copy(home + "/comp", FROM_DOCKER, Location="models/tpu_compiled_models/")
    os.system("cp models/tpu_compiled_models/comp/*edgetpu.tflite models/tpu_compiled_models/")
    os.system("rm -r models/tpu_compiled_models/comp/")
        

def init_compile_folders_dckr():
    docker_exec("mkdir", location)
    docker_exec("mkdir", "comp")

def init_dckr():
    docker_start()

def retrieve_quantized_tflites():
    import os
    from os import listdir
    from os.path import isfile, join, exists

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
                    quant_targets.append(home + location + "/" + q)
                

def retrieve_converted_operations():
    import os
    from os import listdir
    from os.path import isfile, isdir, join, exists

    for d in listdir(CONVERTED_MODELS_DIR): 
        if (isdir(join(CONVERTED_MODELS_DIR, d))):
            path_to_dir = join(CONVERTED_MODELS_DIR, d)
            path_ops.append(path_to_dir)
            ops.append(d)

def edge_tflite_compilation():

    retrieve_converted_operations()
    retrieve_quantized_tflites()
    
    init_dckr()
    init_compile_folders_dckr()
    copy_quantized_files_to_dckr()
    compile_quantized_files_on_dckr()
    copy_quantized_files_from_dckr()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d', '--docker', required=False, 
                        default="exp-docker", 
                        help='Docker Name.')

    parser.add_argument('-l', '--location', required=False, 
                        default="quant", 
                        help='Folder where quantized tflites are placed in docker.')

    parser.add_argument('-u', '--user', required=False, 
                        default="deb", 
                        help='Username on created docker.')

    args = parser.parse_args()

    home = "home/" + args.user + "/"
    docker = args.docker
    location = args.location

    retrieve_converted_operations()
    retrieve_quantized_tflites()
    
    init_dckr()
    init_compile_folders_dckr()
    copy_quantized_files_to_dckr()
    compile_quantized_files_on_dckr()
    copy_quantized_files_from_dckr()

