import numpy as np

class Model:
    def __init__(self, layer, delegate:str, parent:str=""):
        self.details        = layer
        self.delegate       = delegate
        self.parent         = parent
        self.model_path     = ""
        self.model_name     = layer["type"] if isinstance(layer, dict) else None
        self.index          = layer["index"] if isinstance(layer, dict) else None
        self.results        = []
        self.timers         = []
        self.input_vector   = None
        self.output_vector   = None
        self.set_input_details()
        self.set_output_details()

    def set_input_details(self):
        if isinstance(self.details, dict):
            input_tensor = self.details["inputs"][0]
        elif isinstance(self.details, list):
            input_tensor = self.details[0]["inputs"][0]
        self.input_shape = input_tensor["shape"]
        self.input_datatype = input_tensor["type"]
        

    def set_output_details(self):
        if isinstance(self.details, dict):
            output_tensor = self.details["outputs"][0]
        elif isinstance(self.details, list):
            output_tensor = self.details[-1]["outputs"][0]
        self.output_shape = output_tensor["shape"]
        self.output_datatype = output_tensor["type"]
    

    def get_np_dtype(self, datatype: str):
        types = {
            "uint8"     :  np.uint8,
            "uint16"    :  np.uint16,
            "uint32"    :  np.uint32,
            "uint64"    :  np.uint64,
            "int8"      :  np.int8,
            "int16"     :  np.int16,
            "int32"     :  np.int32,
            "int64"     :  np.int64,
            "float16"   :  np.float16,
            "float32"   :  np.float32,
            "float64"   :  np.float64,
        }

        t = types.get(datatype, ValueError("Input datatype is unknown!"))
        return t
