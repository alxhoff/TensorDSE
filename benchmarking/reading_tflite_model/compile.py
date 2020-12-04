TO_DOCKER = 1
FROM_DOCKER = 0

DOCKER = "exp-docker"
LOCATION = "quant"
HOME = "/home/deb/"

OPS = []
PATH_OPS = []
QUANTIZED_TARGETS = []
QUANTIZED_SOURCES = []
COMPILED_SOURCES = []

COUNT = 1000

converted_models_dir = "models/single_layer_models"

cd_deploy_dir = "cd " + HOME + "TensorDSE/benchmarking/reading_tflite_model/"
edge_deploy = "sudo python3 deploy.py -g True -f models/tpu_compiled_models/ -d edge_tpu -c " + str(COUNT)
cpu_deploy = "sudo python3 deploy.py -g True -f models/single_layer_models/ -d cpu -c " + str(COUNT)


def set_count(count):
    global COUNT

    COUNT = count

def place_within_quotes(string):
    from shlex import quote

    return "".join(quote(string))

def concat_args(ARGS):
    SUMMED_ARGS = ""
    for ARG in range(len(ARGS)):
        SUMMED_ARGS += ARGS[ARG]
    return SUMMED_ARGS

def docker_start():
    import os

    docker_start_cmd = "docker start " + DOCKER
    os.system(docker_start_cmd)

def docker_exec(CMD_TYPE, OBJECT=""):
    import os

    DOCKER_EXEC_DICT = {
        "mkdir"                 : ["-ti ", DOCKER + " ", "sh -c ", place_within_quotes("[ -d " + HOME + OBJECT + " ] || " + CMD_TYPE + " " + HOME + OBJECT)],
        "edgetpu_compiler"      : ["-ti ", DOCKER + " ", "sh -c ", place_within_quotes(CMD_TYPE + " -s " + OBJECT + " -o " + HOME + "comp/")],
        "edge_python_deploy"    : ["-ti ", DOCKER + " ", "sh -c ", place_within_quotes(cd_deploy_dir + " && " + edge_deploy)],
        "cpu_python_deploy"    : ["-ti ", DOCKER + " ", "sh -c ", place_within_quotes(cd_deploy_dir + " && " + cpu_deploy)],
        ""                      : None
    }

    default = None
    ARGS = DOCKER_EXEC_DICT.get(CMD_TYPE, default)

    if(ARGS):
        docker_exec_cmd = "docker exec " + concat_args(ARGS)
        os.system(docker_exec_cmd)

def docker_copy(File, DIRECTION_FLAG, Location = ""):
    import os
   
    if(DIRECTION_FLAG):
        docker_copy_cmd = "docker cp " + File + " " + DOCKER + ":" + HOME + Location 
    else:
        docker_copy_cmd = "docker cp " + DOCKER + ":" + File + " " + os.getcwd() + "/" + Location

    os.system(docker_copy_cmd)

def compile_quantized_files_on_dckr():
    for q in QUANTIZED_TARGETS:
        docker_exec("edgetpu_compiler", q)

def copy_quantized_files_to_dckr():
    for q in QUANTIZED_SOURCES:
        docker_copy(q, TO_DOCKER, Location=LOCATION + "/")

def copy_quantized_files_from_dckr():
    import os
    docker_copy(HOME + "/comp", FROM_DOCKER, Location="models/tpu_compiled_models/")
    os.system("cp models/tpu_compiled_models/comp/*edgetpu.tflite models/tpu_compiled_models/")
    os.system("rm -r models/tpu_compiled_models/comp/")
        

def init_compile_folders_dckr():
    docker_exec("mkdir", LOCATION)
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

    for OP, OP_PATH in zip(OPS, PATH_OPS):
        quantized_models_dir = OP_PATH + "/quant/"

        if(os.path.exists(quantized_models_dir)):
            for q in listdir(quantized_models_dir):
                if (isfile(join(quantized_models_dir, q)) and q.startswith(beginning) and q.endswith(ending)):
                    path_to_quant = join(quantized_models_dir, q)
                    QUANTIZED_SOURCES.append(path_to_quant)
                    QUANTIZED_TARGETS.append(HOME + LOCATION + "/" + q)
                

def retrieve_converted_operations():
    import os
    from os import listdir
    from os.path import isfile, isdir, join, exists

    for d in listdir(converted_models_dir): 
        if (isdir(join(converted_models_dir, d))):
            path_to_dir = join(converted_models_dir, d)
            PATH_OPS.append(path_to_dir)
            OPS.append(d)

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

    parser.add_argument('-d', '--docker', required=False, default="exp-docker", help='Docker Name.')
    parser.add_argument('-l', '--location', required=False, default="quant", help='Location/Folder in which quantized tflites are placed in docker.')
    parser.add_argument('-u', '--user', required=False, default="deb", help='Username on created docker.')

    args = parser.parse_args()

    DOCKER = args.docker
    LOCATION = args.location
    HOME = "home/" + args.user + "/"

    retrieve_converted_operations()
    retrieve_quantized_tflites()
    
    init_dckr()
    init_compile_folders_dckr()
    copy_quantized_files_to_dckr()
    compile_quantized_files_on_dckr()
    copy_quantized_files_from_dckr()

