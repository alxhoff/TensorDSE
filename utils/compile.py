def CompileTFLiteModelsForCoral(layers:list):
    """Function responsible for executing the compilation
    of the quantized tflite models.

    """
    from main import log
    from main import LAYERS_FOLDER, COMPILED_MODELS_FOLDER

    from os import listdir, system
    from os.path import isfile, isdir, join, exists

    op_names = []
    op_paths = []

    for i in listdir(LAYERS_FOLDER):
        if isdir(join(LAYERS_FOLDER, i)) and i in layers:
            path = join(LAYERS_FOLDER, i, "quant")
            if exists(path):
                for j in listdir(path):
                    if (isfile(join(path, j))
                        and j.startswith("quant")
                        and j.endswith(".tflite")):
                        op_names.append(i)
                        op_paths.append(join(path, j))

    for op, path in zip(op_names, op_paths):
        ret = system(f"edgetpu_compiler -s {path} -o {COMPILED_MODELS_FOLDER}")
        if ret == 0:
            log.info(f"Operation: {op} compiled for the edgetpu successfully")
        else:
            log.info(f"Operation: {op} failed to compiler for the edgetpu successfully")
            log.info(f"edgetpu_compiler -s {path} -o {COMPILED_MODELS_FOLDER}")
            raise Exception(f"Error during compilation of {op}")
