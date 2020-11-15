import platform

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

def make_interpreter(model_file):
    import tflite_runtime.interpreter as tflite
    model_file, *device = model_file.split('@')

    device = {'device': device[0]} if device else {}
    shared_library = EDGETPU_SHARED_LIB
    experimental_delegates = tflite.load_delegate(shared_library, device)       #Returns loaded Delegate object

    return tflite.Interpreter(model_file, experimental_delegates)

def tflite_deployment(model_file):
    import time

    interpreter = make_interpreter(model_file)                                  #Creates Interpreter Object
    interpreter.allocate_tensors()                                              #Allocates its tensors

    """
    This is where model specific inputs/labels are fed to the model in
    order to be run correctly.
    """
    print('----INFERENCE TIME----')
    print('Note: The first inference on Edge TPU is slow because it includes',
            'loading the model into Edge TPU memory.')
    start = time.perf_counter()
    interpreter.invoke()                                                        #Runs the interpreter/inference, be sure
                                                                                #to have set the input sizes and allocate 
                                                                                #tensors.
    """ This is where we should retrieve the ouput"""
    inference_time = time.perf_counter() - start
    print('%.1fms' % (inference_time * 1000))
    pass

if __name__ == '__main__':
    import time
    import argparse
    import tflite_runtime.interpreter as tflite

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', required=True, help='File path of .tflite file.')
    args = parser.parse_args()

    tflite_deployment(args.model)

