class Model:
    def __init__(self, file:str, delegate:str, parent:str=""):
        self.model_path     = file
        self.delegate       = delegate
        self.model_name     = self._get_model_name(file)
        self.results        = []
        self.timers         = []
        self.parent         = parent

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

