import os

SOURCE_GEN_DIR = os.path.dirname(os.path.abspath(__file__))
DEPLOYMENT_DIR = os.path.dirname(SOURCE_GEN_DIR)
MODEL_LAB_DIR  = os.path.join(DEPLOYMENT_DIR, "ModelLab")


load_models_instance = "\
  std::unique_ptr<tflite::FlatBufferModel> model_{0};\n\
  model_{0} = LoadModelFile(\"{1}\");\n"

check_load_model = "\
  if (!model_{0}) {{\n\
    std::cerr << \"Cannot load model from \" << \"{1}\" << std::endl;\n\
    return 1;\n\
  }}\n\
  std::cout << \"{2} successfully loaded! \" << std::endl;\n"

tpu_interpreter_template = "\
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
  std::cout << \"\\n\" << std::endl;\n\n" 

gpu_interpreter_template = "\
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
  std::cout << \"\\n\" << std::endl;\n\n"

cpu_interpreter_template = "\
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
  std::cout << \"\\n\" << std::endl;\n\n"

invoke_section = "\
  if (interpreter_{0}->Invoke() != kTfLiteOk) {{\n\
    std::cerr << \"Cannot invoke interpreter\" << std::endl;\n\
    return 1;\n\
  }}\n"

input_invoke = "{1}\n\
  auto inference_{0}_end = std::chrono::high_resolution_clock::now();\n\
  const auto* intermediate_tensor_{0} = interpreter_{0}->output_tensor(0);\n\n"

output_invoke = "\
  *interpreter_{0}->input_tensor(0) = *intermediate_tensor_{1};\n\
  auto inference_{0}_start = std::chrono::high_resolution_clock::now();\n\
  {2}\n\n"

timing_section = "\
  *interpreter_{0}->input_tensor(0) = *intermediate_tensor_{1};\n\
  auto inference_{0}_start = std::chrono::high_resolution_clock::now();\n\
  {2}\n\
  auto inference_{0}_end = std::chrono::high_resolution_clock::now();\n\
  const auto* intermediate_tensor_{0} = interpreter_{0}->output_tensor(0);\n\n"

function_str_ms = "auto exact_inference_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>"
function_str_ns = "auto exact_inference_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>"