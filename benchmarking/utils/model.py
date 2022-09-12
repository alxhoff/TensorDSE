class Model:
    def __init__(self, file:str, delegate:str, parent:str=""):
        self.delegate       = delegate
        self.parent         = parent
        self.model_path     = file
        self.model_name     = self._get_model_name(file)
        self.results        = []
        self.timers         = []

    def  _get_model_name (self, file_path:str) -> str:
        file = file_path.split("/")[file_path.count("/")]
        if (not file.startswith("quant_") or
            not file.endswith(".tflite")):
               raise Exception(
                       f"File: {file_path} not a tflite file")
        return (
         file.split("quant_")[1]
        ).split("_edgetpu.tflite"
           if self.delegate == "tpu"
           else ".tflite" )[0]

    def set_input(self, shape, datatype):
        import numpy as np
        types = {
                np.uint8    : "uint8",
                np.uint16   : "uint16",
                np.uint32   : "uint32",
                np.uint64   : "uint64",
                np.int8     : "int8",
                np.int16    : "int16",
                np.int32    : "int32",
                np.int64    : "int64",
                np.float16  : "float16",
                np.float32  : "float32",
                np.float64  : "float64",
                np.float128 : "float128",
        }

        t = types.get(datatype, ValueError("Input datatype is unknown!"))
        self.input_shape    = shape.tolist()
        self.input_datatype = t

