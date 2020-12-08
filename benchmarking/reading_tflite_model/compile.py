TO_DOCKER = 1
FROM_DOCKER = 0

docker = "exp-docker"
location = "quant"
home = "/home/deb/"

ops = []
path_ops = []
quant_targets = []
quant_sources = []
compiled_sources = []

count = 1000

converted_models_dir = "models/single_layer_models"

cd_deploy_dir = "cd " + home + "TensorDSE/benchmarking/reading_tflite_model/"
edge_deploy = "sudo python3 deploy.py -g True -f models/tpu_compiled_models/ -d edge_tpu -c " + str(count)
cpu_deploy = "sudo python3 deploy.py -g True -f models/single_layer_models/ -d cpu -c " + str(count)


def set_globals(cnt):
    global cpu_deploy
    global edge_deploy
    global count

    count = cnt
    edge_deploy = "sudo python3 deploy.py -g True -f models/tpu_compiled_models/ -d edge_tpu -c " + str(count)
    cpu_deploy = "sudo python3 deploy.py -g True -f models/single_layer_models/ -d cpu -c " + str(count)

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

    docker_start_cmd = "docker start " + docker
    os.system(docker_start_cmd)

def docker_exec(cmd_type, objct=""):
    import os

    docker_exec_dict = {
        "mkdir"                 : ["-ti ", docker + " ", "sh -c ", place_within_quotes("[ -d " + home + objct + " ] || " + cmd_type + " " + home + objct)],
        "edgetpu_compiler"      : ["-ti ", docker + " ", "sh -c ", place_within_quotes(cmd_type + " -s " + objct + " -o " + home + "comp/")],
        "edge_python_deploy"    : ["-ti ", docker + " ", "sh -c ", place_within_quotes(cd_deploy_dir + " && " + edge_deploy)],
        "cpu_python_deploy"    : ["-ti ", docker + " ", "sh -c ", place_within_quotes(cd_deploy_dir + " && " + cpu_deploy)],
        ""                      : None
    }

    default = None
    args = docker_exec_dict.get(cmd_type, default)

    if(args):
        docker_exec_cmd = "docker exec " + concat_args(args)
        os.system(docker_exec_cmd)

def docker_copy(File, DIRECTION_FLAG, Location = ""):
    import os
   
    if(DIRECTION_FLAG):
        docker_copy_cmd = "docker cp " + File + " " + docker + ":" + home + Location 
    else:
        docker_copy_cmd = "docker cp " + docker + ":" + File + " " + os.getcwd() + "/" + Location

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

    for d in listdir(converted_models_dir): 
        if (isdir(join(converted_models_dir, d))):
            path_to_dir = join(converted_models_dir, d)
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

    docker = args.docker
    location = args.location
    home = "home/" + args.user + "/"

    retrieve_converted_operations()
    retrieve_quantized_tflites()
    
    init_dckr()
    init_compile_folders_dckr()
    copy_quantized_files_to_dckr()
    compile_quantized_files_on_dckr()
    copy_quantized_files_from_dckr()

