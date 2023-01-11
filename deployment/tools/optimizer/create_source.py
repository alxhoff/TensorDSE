import sys
import os
import jinja2
import argparse
import utils

OPTIMIZER_DIR  = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR      = os.path.dirname(OPTIMIZER_DIR)
DEPLOYMENT_DIR = os.path.dirname(TOOLS_DIR)
log = utils.LoggerInit()

def ParseArgs():
    # Initialize parser
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument("-md", "--ModelsDirectory", help = "Path to Directory Containing TFLite models", required=False)
    parser.add_argument("-fm", "--FinalMapping", help = "Path to JSON file containing Final Mapping", required=False)
    # Read arguments from command line
    try:
        args = parser.parse_args()
        if (args.ModelsDirectory == None):
            args.ModelsDirectory = os.path.join(OPTIMIZER_DIR, "models", "sub", "tflite")
        if (args.FinalMapping == None):
            args.FinalMapping = os.path.join(OPTIMIZER_DIR, "resources", "final_mapping.json")
        return args
    except Exception as e:
        log.error('The provided argument could not be parsed! Potential Cause: {}'.format(str(e)))
        log.info('Example Usage: create_source.py -md <path/to/models/dir> -fm <path/to/json/file/containing/final/maping>')
        sys.exit(1)

def PreparePaths(directory: str):
    file_list = os.listdir(directory)
    ordered_file_list = sorted(file_list)
    for p in ordered_file_list:
        if "tpu" in p:
            if not(p.endswith("edgetpu.tflite")):
                ordered_file_list.remove(p)
    for i, file in enumerate(ordered_file_list):
        ordered_file_list[i] = os.path.join(directory, file)
    return ordered_file_list
            
def CreateLoadModelsSection(filepaths: list):
    load_models_section = ""
    for i, file in enumerate(filepaths):
        model_for = file.split("submodel")[1]
        submodel_name = "submodel{0}".format(model_for)
        load_models_instance = "\
  std::unique_ptr<tflite::FlatBufferModel> model_{0};\n\
  model_{0} = LoadSubmodel(\"{1}\");\n".format(str(i), file)
        check_load_model = "  if (!model_{0}) {{\n    std::cerr << \"Cannot load model from \" << \"{1}\" << std::endl;\n    return 1;\n  }}\n  std::cout << \"{2} successfully loaded! \" << std::endl;\n".format(str(i), file, submodel_name)
        load_models_section = load_models_section + load_models_instance + check_load_model
    return load_models_section

def CreateBuildTpuInterpreter(index: int, key: str):
    return "\
  std::cout << \"############################ Build {1} Interpreter ############################\" << std::endl;\n\
  std::cout << \"\\n\" << std::endl;\n\
  std::cout << \"Initializing Edge TPU Context ... \" << \"\\n\";\n\
  const std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context_{0} = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(available_tpus[0].type, available_tpus[0].path);\n\
  tflite::ops::builtin::BuiltinOpResolver resolver_{0};\n\
  resolver_{0}.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());\n\
  std::cout << \"Building Edge TPU Interpreter ... \" << \"\\n\";\n\
  tflite::InterpreterBuilder builder_{0}(*model_{0}, resolver_{0});\n\
  std::unique_ptr<tflite::Interpreter> interpreter_{0};\n\
  builder_{0}(&interpreter_{0});\n\
  interpreter_{0}->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context_{0}.get());\n\
  interpreter_{0}->SetNumThreads(1);\n\
  // Allocate tensors \n\
  if (interpreter_{0}->AllocateTensors() != kTfLiteOk) {{\n\
    std::cerr << \"Failed to allocate tensors.\" << std::endl;\n\
    return 1;\n\
  }}\n\
  std::cout << \"Tensors successfully allocated.\" << \"\\n\";\n\
  std::cout << \"\\n\" << std::endl;\n\n".format(str(index), key)

def CreateBuildGpuInterpreter(index: int, key: str):
    return "\
  std::cout << \"############################ Build {1} Interpreter ############################\" << std::endl;\n\
  std::cout << \"\\n\" << std::endl;\n\
  tflite::ops::builtin::BuiltinOpResolver resolver_{0};\n\
  tflite::InterpreterBuilder builder_{0}(*model_{0}, resolver_{0});\n\
  std::unique_ptr<tflite::Interpreter> interpreter_{0};\n\
  builder_{0}(&interpreter_{0});\n\
  // Allocate tensors \n\
  if (interpreter_{0}->AllocateTensors() != kTfLiteOk) {{\n\
    std::cerr << \"Failed to allocate tensors.\" << std::endl;\n\
    return 1;\n\
  }}\n\
  std::cout << \"Tensors successfully allocated.\" << \"\\n\";\
  // NEW: Prepare GPU delegate.\n\
  const TfLiteGpuDelegateOptionsV2 options_{0} = TfLiteGpuDelegateOptionsV2Default();\n\
  auto* delegate_{0} = TfLiteGpuDelegateV2Create(&options_{0});\n\
  if (interpreter_{0}->ModifyGraphWithDelegate(delegate_{0}) != kTfLiteOk) {{ // Experimental: tflite::InterpreterBuilder::AddDelegate();\n\
    fprintf(stderr, \"Error at %s:%d\\n\", __FILE__, __LINE__);\n\
    return 1;\n\
  }}\n\
  std::cout << \"\\n\" << std::endl;\n\n".format(str(index), key)

def CreateBuildCpuInterpreter(index: int, key: str):
    return "\
  std::cout << \"############################ Build {1} Interpreter ############################\" << std::endl;\n\
  std::cout << \"\\n\" << std::endl;\n\
  tflite::ops::builtin::BuiltinOpResolver resolver_{0};\n\
  tflite::InterpreterBuilder builder_{0}(*model_{0}, resolver_{0});\n\
  std::unique_ptr<tflite::Interpreter> interpreter_{0};\n\
  builder_{0}(&interpreter_{0});\n\
  // Allocate tensors\n\
  if (interpreter_{0}->AllocateTensors() != kTfLiteOk) {{\n\
    std::cerr << \"Failed to allocate tensors.\" << std::endl;\n\
    return 1;\n\
  }}\n\
  std::cout << \"Tensors successfully allocated.\" << \"\\n\";\n\
  std::cout << \"\\n\" << std::endl;\n\n".format(str(index), key)

def CreateBuildInterpretersSection(mapping: dict):
    build_interpreters_section = ""
    for i, key in enumerate(mapping):
        section_to_add = ""
        if "CPU" in key:
            section_to_add =  CreateBuildCpuInterpreter(i, key)
        elif "GPU" in key:
            section_to_add =  CreateBuildGpuInterpreter(i, key)
        elif "TPU" in key:
            section_to_add =  CreateBuildTpuInterpreter(i, key)
        build_interpreters_section = build_interpreters_section + section_to_add
    return build_interpreters_section

def CreateInvokeSection(mapping: dict):
    invoke_section = ""
    input_invoke = "\
  if (interpreter_{0}->Invoke() != kTfLiteOk) {{\n\
    std::cerr << \"Cannot invoke interpreter\" << std::endl;\n\
    return 1;\n\
  }}\n\
  const auto* intermediate_tensor_{0} = interpreter_{0}->output_tensor(0);\n\n".format(str(0))
    output_invoke = "\
  *interpreter_{0}->input_tensor(0) = *intermediate_tensor_{1};\n\
  if (interpreter_{0}->Invoke() != kTfLiteOk) {{\n\
    std::cerr << \"Cannot invoke interpreter\" << std::endl;\n\
    return 1;\n\
  }}\n\n".format(str(len(mapping)-1), str(len(mapping)-2))

    invoke_section = invoke_section + input_invoke

    for index in range(1, len(mapping)-1):
        section_to_add = "\
  *interpreter_{0}->input_tensor(0) = *intermediate_tensor_{1};\n\
  if (interpreter_{0}->Invoke() != kTfLiteOk) {{\n\
    std::cerr << \"Cannot invoke interpreter\" << std::endl;\n\
    return 1;\n\
  }}\n\
  const auto* intermediate_tensor_{0} = interpreter_{0}->output_tensor(0);\n\n".format(str(index), str(index-1))
        invoke_section = invoke_section + section_to_add

    return invoke_section + output_invoke

def CreateSource(ModelsDirectory: str, FinalMapping: dict):
  # Prepare Jinja Environment
  environment = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.join("resources", "templates")))
  output_filepath = os.path.join(DEPLOYMENT_DIR, "tflite-cpp", "src", "classification", "distributed", "main.cpp")
  template = environment.get_template("main.cpp")

  # Read Mapping
  mapping = utils.ReadJSON(FinalMapping)

  # Prepare Full Model Paths
  model_paths = PreparePaths(ModelsDirectory)

  # Prepare Section for Loading Models (Submodels)
  load_models_section = CreateLoadModelsSection(model_paths)

  # Prepare Section for Building Interpreter Objects for each Model
  build_interpreters_section = CreateBuildInterpretersSection(mapping)

  # Prepare Section for Invoking Inference and Passing Intermediate Tensors
  invoke_interpreters_section = CreateInvokeSection(mapping)

  # Populate Context
  context = {
      "input_interpreter": "interpreter_{0}".format("0"),
      "output_interpreter": "interpreter_{0}".format(str(len(mapping)-1)),
      "load_models_section": load_models_section,
      "build_interpreters_section": build_interpreters_section,
      "invoke_interpreters_section": invoke_interpreters_section
  }

  with open(output_filepath, mode="w", encoding="utf-8") as output:
      output.write(template.render(context))

def main():
  try:
    # Parse Arguments
    args = ParseArgs()
    CreateSource(args.ModelsDirectory, args.FinalMapping)
  except Exception as e:
    log.error("Failed to Create Target Source File! Potential Cause: {}".format(str(e)))
    
if __name__ == '__main__':
    main()
    pass