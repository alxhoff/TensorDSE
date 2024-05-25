"""
Missing: DOCSTRING
"""

import json
import pandas as pd

def load_json_data(filenames):
    """
    Missing: DOCSTRING
    """
    data = []
    for filename in filenames:
        with open(filename, 'r', encoding="utf-8") as f:
            data.append(json.load(f))
    return data

def extract_data(models_data, model_name):
    """
    Missing: DOCSTRING
    """

    general_info_data = []
    layer_profiles_data = []
    usb_communication_data = []

    for model_data in models_data:
        for model in model_data['models']:
            if model['name'] == model_name:
                runs = model['runs']
                platform = model['platform']
                general_info_data.append({
                    'Model Name': model_name,
                    'Dataset': 'MNIST' if 'MNIST' in model_name else 'ResNet',
                    'Runs': runs,
                    'Platform': platform
                })
                
                for layer in model['layers']:
                    layer_name = layer['name']
                    for delegate in layer['delegates']:
                        device = delegate['device']
                        input_shape = 'x'.join(delegate['input']['shape'])
                        input_type = delegate['input']['type']
                        profile = {
                            'Layer Name': layer_name,
                            'Device': device,
                            'Platform': platform,
                            'Input Shape': input_shape,
                            'Input Type': input_type,
                            'Mean': delegate['mean'],
                            'Median': delegate['median'],
                            'Std Dev': delegate['standard_deviation'],
                            'Avg Abs Dev': delegate['avg_absolute_deviation'],
                            'Distribution': delegate['distribution'],
                            'USB Comm Mean': delegate['usb']['communication']['mean'],
                            'CPP Preprocessing': delegate['cpp_preprocessing'],
                            'CPP Postprocessing': delegate['cpp_postprocessing'],
                            'Backend Inference Time': delegate['backend_inference_time'],
                            'Mean Computation': delegate['mean_computation']
                        }
                        layer_profiles_data.append(profile)

                        if device == 'tpu' and 'send' in delegate['usb'] and 'recv' in delegate['usb']:
                            usb_data = {
                                'Layer Name': layer_name,
                                'Device': device,
                                'Platform': platform,
                                'USB Send Mean': delegate['usb']['send']['mean'],
                                'USB Send Median': delegate['usb']['send']['median'],
                                'USB Send Std Dev': delegate['usb']['send']['standard_deviation'],
                                'USB Send Avg Abs Dev': delegate['usb']['send']['avg_absolute_deviation'],
                                'USB Recv Mean': delegate['usb']['recv']['mean'],
                                'USB Recv Median': delegate['usb']['recv']['median'],
                                'USB Recv Std Dev': delegate['usb']['recv']['standard_deviation'],
                                'USB Recv Avg Abs Dev': delegate['usb']['recv']['avg_absolute_deviation'],
                                'USB Total Mean': delegate['usb']['communication']['mean'],
                                'USB Total Median': delegate['usb']['communication']['median'],
                                'USB Total Std Dev': delegate['usb']['communication']['standard_deviation'],
                                'USB Total Avg Abs Dev': delegate['usb']['communication']['avg_absolute_deviation']
                            }
                            usb_communication_data.append(usb_data)

    return general_info_data, layer_profiles_data, usb_communication_data

# Define filenames for each model
filepaths_resnet = ['/home/alafnayou/Documents/fp/TensorDSE/resources/profiling_results/desktop/model_0_pretrainedResnet_quant.json',
                   '/home/alafnayou/Documents/fp/TensorDSE/resources/profiling_results/coral/model_0_pretrainedResnet_quant.json',
                   '/home/alafnayou/Documents/fp/TensorDSE/resources/profiling_results/rpi/model_0_pretrainedResnet_quant.json']

filepaths_mnist = ['/home/alafnayou/Documents/fp/TensorDSE/resources/profiling_results/desktop/model_0_MNIST_full_quanitization.json',
                   '/home/alafnayou/Documents/fp/TensorDSE/resources/profiling_results/coral/model_0_MNIST_full_quanitization.json',
                   '/home/alafnayou/Documents/fp/TensorDSE/resources/profiling_results/rpi/model_0_MNIST_full_quanitization.json']


filepaths_mnist_extended = ['/home/alafnayou/Documents/fp/TensorDSE/resources/profiling_results/desktop/model_0_MNIST_extended_full_quanitization.json',
                            '/home/alafnayou/Documents/fp/TensorDSE/resources/profiling_results/coral/model_0_MNIST_extended_full_quanitization.json',
                            '/home/alafnayou/Documents/fp/TensorDSE/resources/profiling_results/rpi/model_0_MNIST_extended_full_quanitization.json']

# Load JSON data
data_resnet = load_json_data(filepaths_resnet)
data_mnist = load_json_data(filepaths_mnist)
data_mnist_extended = load_json_data(filepaths_mnist_extended)

# Extract and consolidate data for each model
general_info_resnet, layer_profiles_resnet, usb_communication_resnet = extract_data(data_resnet, 'model_0_pretrainedResnet_quant')
general_info_mnist, layer_profiles_mnist, usb_communication_mnist = extract_data(data_mnist, 'model_0_MNIST_full_quanitization')
general_info_mnist_extended, layer_profiles_mnist_extended, usb_communication_mnist_extended = extract_data(data_mnist_extended, 'model_0_MNIST_extended_full_quanitization')

# Create DataFrames
general_info_resnet_df = pd.DataFrame(general_info_resnet)
layer_profiles_resnet_df = pd.DataFrame(layer_profiles_resnet)
usb_communication_resnet_df = pd.DataFrame(usb_communication_resnet)

general_info_mnist_df = pd.DataFrame(general_info_mnist)
layer_profiles_mnist_df = pd.DataFrame(layer_profiles_mnist)
usb_communication_mnist_df = pd.DataFrame(usb_communication_mnist)

general_info_mnist_extended_df = pd.DataFrame(general_info_mnist_extended)
layer_profiles_mnist_extended_df = pd.DataFrame(layer_profiles_mnist_extended)
usb_communication_mnist_extended_df = pd.DataFrame(usb_communication_mnist_extended)

# Export to Excel
with pd.ExcelWriter('/home/alafnayou/Documents/fp/TensorDSE/resources/profiling_results/profiling_analysis_resnet.xlsx') as writer:
    general_info_resnet_df.to_excel(writer, sheet_name='General_Info', index=False)
    layer_profiles_resnet_df.to_excel(writer, sheet_name='Layer_Profiles', index=False)
    if not usb_communication_resnet_df.empty:
        usb_communication_resnet_df.to_excel(writer, sheet_name='USB_Communication', index=False)

with pd.ExcelWriter('/home/alafnayou/Documents/fp/TensorDSE/resources/profiling_results/profiling_mnist.xlsx') as writer:
    general_info_mnist_df.to_excel(writer, sheet_name='General_Info', index=False)
    layer_profiles_mnist_df.to_excel(writer, sheet_name='Layer_Profiles', index=False)
    if not usb_communication_mnist_df.empty:
        usb_communication_mnist_df.to_excel(writer, sheet_name='USB_Communication', index=False)

with pd.ExcelWriter('/home/alafnayou/Documents/fp/TensorDSE/resources/profiling_results/profiling_mnist_extended.xlsx') as writer:
    general_info_mnist_extended_df.to_excel(writer, sheet_name='General_Info', index=False)
    layer_profiles_mnist_extended_df.to_excel(writer, sheet_name='Layer_Profiles', index=False)
    if not usb_communication_mnist_extended_df.empty:
        usb_communication_mnist_extended_df.to_excel(writer, sheet_name='USB_Communication', index=False)
