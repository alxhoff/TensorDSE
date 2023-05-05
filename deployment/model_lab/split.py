import os
import json
import urllib
from .model import Model, Submodel
from .utils import LoggerInit, ReadCSV, CopyFile

log = LoggerInit("splitter.log")

WORK_DIR      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR    = os.path.join(WORK_DIR, "models")
MAPPING_DIR   = os.path.join(WORK_DIR, "mapping")
RESOURCES_DIR = os.path.join(WORK_DIR, "resources")
SOURCE_DIR    = os.path.join(MODELS_DIR, "source")
SUB_DIR       = os.path.join(MODELS_DIR, "sub")
FINAL_DIR     = os.path.join(MODELS_DIR, "final")

HW_ID = ["CPU", "GPU", "TPU", "XPU"]
MODES = ["benchmarking", "deployment"]

class Splitter:
    def __init__(self, mode_of_operation, source_model_path, mapping_path) -> None:
        log.info("Initializing Environment ...")
        self.InitializeEnv(source_model_path, mapping_path)
        log.info("Initializing Source Model ...")
        self.source_model = Model(self.source_model_path, self.schema_path)
        log.info("Source Model saved under: {}".format(os.path.join(MODELS_DIR, "source", "tflite")))
        if not(mapping_path == None):
            log.info("Reading Mapping ...")
            self.mapping = ReadCSV(self.mapping_path)
        self.mode = mode_of_operation
        
    def CheckSchema(self):
        log.info("Checking schema ...")
        self.schema_path = os.path.join(RESOURCES_DIR,"schema","schema.fbs")
        if not os.path.exists(self.schema_path):    
            log.info("    File schema.fbs was not found, downloading...")
            urllib.request.urlretrieve("https://github.com/tensorflow/tensorflow/raw/master/tensorflow/lite/schema/schema.fbs",
                                   "schema.fbs")
            log.info("    Downloaded schema.fbs")
        else:
            log.info("    File schema.fbs found.")

    def InitializeEnv(self, source_model_path, mapping_path):
        self.CheckSchema()
        os.mkdir(MODELS_DIR)
        for directory in ["source", "sub", "final"]:
            sub_dir = os.path.join(MODELS_DIR, directory)
            os.mkdir(sub_dir)
            for ext in ["tflite", "json"]:
                ext_dir = os.path.join(sub_dir, ext)
                os.mkdir(ext_dir)

        #model_filename = source_model_path.split("/")[len(source_model_path.split("/"))-1]
        self.source_model_path = os.path.join(SOURCE_DIR, "tflite/")
        CopyFile(source_model_path, self.source_model_path)
        
        os.mkdir(MAPPING_DIR)

        mapping_filename = mapping_path.split("/")[len(mapping_path.split("/"))-1]
        self.mapping_path = os.path.join(MAPPING_DIR, mapping_filename)
        CopyFile(mapping_path, MAPPING_DIR)
        log.info("Mapping saved under: {}".format(os.path.join(MAPPING_DIR)))
    
    def AnalyseMapping(self):
        log.info("Analysing Mapping ...")
        self.final_mapping = {}

        if (self.mode == MODES[1]):
            sequence = {
                "NEW": True,
                "IN" : False,
                "END": False
            }
            sequence_tracker = dict.fromkeys(HW_ID, 0)
            for i, op in enumerate(self.mapping):
                current_hw = HW_ID[op[1]]
                if sequence["NEW"]:
                    sequence_tracker[current_hw]+=1
                    self.final_mapping[current_hw+str(sequence_tracker[current_hw])] = [i]
                    sequence["NEW"] = False
                    sequence["IN"] = True
                elif sequence["IN"]:
                    self.final_mapping[current_hw+str(sequence_tracker[current_hw])].append(i)

                if (i+1 == len(self.mapping)):
                    sequence["NEW"] = False
                    sequence["IN"]  = False
                    sequence["END"] = True
                else:
                    sequence["END"] = (op[1] != self.mapping[i+1][1])
                    if sequence["END"]:
                        sequence["NEW"] = True
                        sequence["IN"]  = False
            
        elif (self.mode == MODES[0]):
            for i, op in enumerate(self.mapping):
                current_hw = HW_ID[op[1]]
                self.final_mapping[current_hw+str(i+1)] = [i]


        for e in self.final_mapping:
            ops = self.final_mapping[e]
            log.info("        Operations : " + ", ".join(str(op) for op in ops))
            log.info("        Target HW  : " + e[:3])
            
        final_mapping_path = os.path.join(RESOURCES_DIR, "final_mapping.json")
        with open(final_mapping_path, "w") as fout:
            json.dump(self.final_mapping, fout, indent=2)

        log.info("Final Mapping Saved!\n")

    def ReadSourceModel(self):
        log.info("Converting Source Model from TFLite to JSON ...")
        self.source_model.Convert("tflite", "json")
        log.info("OK\n")

    def CreateSubmodels(self):
        self.submodel_list = []
        for i, block_key in enumerate(self.final_mapping.keys()):
            log.info("Initializing Shell Model for Submodel {0} ...".format(str(i)))
            submodel = Submodel(self.source_model.json)
            log.info("OK")
            log.info("Adding Operations (" + ", ".join(str(op) for op in self.final_mapping[block_key]) + ") to Shell Model ...")
            submodel.AddOps(self.final_mapping[block_key])
            log.info("OK")
            log.info("Saving Submodel {0} | Operations: {1} | Target HW: {2} ...".format(str(i), ", ".join(str(op) for op in self.final_mapping[block_key]), block_key))
            submodel.Save(block_key, i)
            log.info("OK")
            log.info("Converting Submodel {0} from JSON to TFLite ...".format(str(i)))
            submodel.Convert("json", "tflite")
            self.submodel_list.append(submodel)
            log.info("OK\n")
    
    def CompileForEdgeTPU(self):
        for i, submodel in enumerate(self.submodel_list):
            submodel_path = submodel.paths["tflite"]
            if(self.mode == MODES[1]):
                if ((submodel_path.startswith("submodel{0}_tpu".format(i))) and (not(submodel_path.endswith("_edgetpu.tflite")))):
                    submodel.Compile()
                else:
                    continue
            elif(self.mode == MODES[0]):
                submodel.Compile()

    def Run(self):
        self.AnalyseMapping()
        self.ReadSourceModel()
        self.CreateSubmodels()
        self.CompileForEdgeTPU()