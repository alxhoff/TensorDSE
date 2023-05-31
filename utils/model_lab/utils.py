import csv
import json
import time


class LayerDetails:
    def __init__(self, summary_path:str) -> None:
        summary = ReadJSON(summary_path)
        self.layers = summary["layers"]
    
    def ReadLayerDetails(self, layer:dict, layer_index=-1):
        if (layer_index > 0) and (layer == None):
            layer = self.layers[layer_index]

        input_tensor = layer["inputs"][0]
        self.input_tensor_shape = input_tensor["shape"]
        self.input_tensor_dtype = input_tensor["type"]
        
        output_tensor = layer["outputs"][0]
        self.output_tensor_shape = output_tensor["shape"]
        self.output_tensor_dtype = output_tensor["type"]

        self.hardware_target = layer["mapping"]
        self.name = layer["type"]
        self.index = layer["index"]

    def GetTensorSize(self, entry:str):
        if (entry == "Input"):
            shape = self.input_tensor_shape
        elif (entry == "Output"):
            shape = self.output_tensor_shape
        size = 1
        for dim in shape:
            size *= int(dim)
        return size
            
def RunTerminalCommand(*cmd, save_output=False, wait_time=0.5):
    """ Execute an arbitrary command and echo its output."""
    import subprocess
    p = subprocess.run(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    time.sleep(wait_time)
    output = p.stdout.decode()
    p.check_returncode()
    if save_output:
        return output
    
def ReadCSV(csv_file_path: str):
    with open(csv_file_path, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        data = [list(map(int,rec)) for rec in reader]
        return data

def ReadJSON(file_path: str):
    with open(file_path) as fin:
        return json.load(fin)

def CopyFile(source: str, destination: str):
    RunTerminalCommand("cp", source, destination)

def MoveFile(source: str, destination: str):
    RunTerminalCommand("mv", source, destination)

def CompareKeys(dictA: dict, dictB: dict):
    excluded = []
    for ka in list(dictA.keys()):
        if (ka not in list(dictB.keys())):
            excluded.append(ka)
    if len(excluded) !=0:
        return False, excluded
    else:
        return True, None