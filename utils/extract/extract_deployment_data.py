"""
DOCSTRING: TODO
"""

import json
import pandas as pd

def process_json_files(json_file_paths):
    """
    DOCSTRING: TODO
    """
    submodels_list = []
    total_times_list = []

    for i, file_path in enumerate(json_file_paths):
        with open(file_path, encoding="utf-8") as file:
            data = json.load(file)

        submodels = data['submodels']

        for submodel in submodels:
            row = {
                'iteration': i,
                'name': submodel['name'],
                'type': submodel['layers'][0]['type'],
                'mapping': submodel['layers'][0]['mapping'],
                'inference_time (c++)': submodel['inference_time (c++)'],
                'pre_processing_time (c++)': submodel['pre_processing_time (c++)'],
                'post_processing_time (c++)': submodel['post_processing_time (c++)'],
                'inference_time (python)': submodel['inference_time (python)'],
            }
            submodels_list.append(row)

        total_times = {
            'iteration': i,
            'total_inference_time (c++)': data['total_inference_time (c++)'],
            'total_pre_processing_time (c++)': data['total_pre_processing_time (c++)'],
            'total_post_processing_time (c++)': data['total_post_processing_time (c++)'],
            'total_inference_time (python)': data['total_inference_time (python)']
        }
        total_times_list.append(total_times)

    return submodels_list, total_times_list

def create_excel_file(submodels_list, total_times_list, output_excel_path=''):
    # Create DataFrames
    df_submodels = pd.DataFrame(submodels_list)
    df_totals = pd.DataFrame(total_times_list)

    # Save DataFrames to an Excel file with multiple sheets
    with pd.ExcelWriter(output_excel_path) as writer:
        df_submodels.to_excel(writer, sheet_name='Submodels Times', index=False)
        df_totals.to_excel(writer, sheet_name='Total Times', index=False)

    print(f"Data sheet has been created and saved as '{output_excel_path}'.")


multi3_mnist = [
    '/home/alafnayou/Documents/fp/TensorDSE/resources/deployment_results/rpi/multi_3/model_0_MNIST_full_quanitization.json',
    '/home/alafnayou/Documents/fp/TensorDSE/resources/deployment_results/rpi/multi_3/model_1_MNIST_full_quanitization.json',
    '/home/alafnayou/Documents/fp/TensorDSE/resources/deployment_results/rpi/multi_3/model_2_MNIST_full_quanitization.json'
    ]

multi3_mnist_extended = [
    '/home/alafnayou/Documents/fp/TensorDSE/resources/deployment_results/rpi/multi_3/model_0_MNIST_extended_full_quanitization.json',
    '/home/alafnayou/Documents/fp/TensorDSE/resources/deployment_results/rpi/multi_3/model_1_MNIST_extended_full_quanitization.json',
    '/home/alafnayou/Documents/fp/TensorDSE/resources/deployment_results/rpi/multi_3/model_2_MNIST_extended_full_quanitization.json'
    ]

multi3_resnet = [
    '/home/alafnayou/Documents/fp/TensorDSE/resources/deployment_results/rpi/multi_3/model_0_pretrainedResnet_quant.json',
    '/home/alafnayou/Documents/fp/TensorDSE/resources/deployment_results/rpi/multi_3/model_1_pretrainedResnet_quant.json',
    '/home/alafnayou/Documents/fp/TensorDSE/resources/deployment_results/rpi/multi_3/model_2_pretrainedResnet_quant.json'
    ]

submodels_list, total_times_list = process_json_files(multi3_resnet)
create_excel_file(submodels_list, total_times_list,
                  output_excel_path='/home/alafnayou/Documents/fp/results/deployment_results/rpi/multi_3_pretrainedResnet_quant.xlsx')