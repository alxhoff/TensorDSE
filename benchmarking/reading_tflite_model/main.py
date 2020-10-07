from tflite.Model import Model
from tflite.BuiltinOperator import BuiltinOperator

#tflite folder generated using tflite schema and the flattbuffer compiler
#See  https://google.github.io/flatbuffers/flatbuffers_guide_tutorial.html

model_filename = "MNIST_model.tflite"

def result_code_to_name(code):
  for name, value in BuiltinOperator.__dict__.items():
    if value == code:
       return name
  return None

def main():
    with open(model_filename, "rb") as f:
        model = Model.GetRootAsModel(f.read(), 0)

        ops = []
        op_codes = []
        graph = model.Subgraphs(0)

        for i in range(graph.OperatorsLength()):
            opc = model.OperatorCodes(graph.Operators(i).OpcodeIndex()).BuiltinCode()
            ops.append(result_code_to_name(opc))
            op_codes.append(opc)

        for op in ops:
            print(op)

if __name__ == '__main__':
    main()