def CompileTFLiteModelsForCoral():
    """Function responsible for executing the compilation
    of the quantized tflite models.

    """
    from main import log
    from main import MODELS_FOLDER, COMPILED_MODELS_FOLDER

    from os import listdir, system
    from os.path import isfile, isdir, join, exists

    op_names = []
    op_paths = []

    for i in listdir(MODELS_FOLDER):
        if isdir(join(MODELS_FOLDER, i)):
            path = join(MODELS_FOLDER, i, "quant")
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
            log.error(f"Operation: {op} faulty compilation with return value: {ret}")

