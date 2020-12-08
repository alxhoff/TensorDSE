
def shark_deploy_edge(count):
    import os
    import utils
    import compile
    from compile import TO_DOCKER, FROM_DOCKER, home
    
    path_to_tensorDSE = utils.retrieve_folder_path(os.getcwd(), "TensorDSE")
    path_to_docker_results =  home + "TensorDSE/benchmarking/reading_tflite_model/results/"

    compile.set_globals(count)
    compile.docker_copy(path_to_tensorDSE, TO_DOCKER)
    compile.docker_exec("shark_edge_python_deploy")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-c', '--count', required=False, 
                        default=1000, 
                        help='Count of the number of times of edge deployment.')

    args = parser.parse_args()

    shark_deploy_edge(args.count)
