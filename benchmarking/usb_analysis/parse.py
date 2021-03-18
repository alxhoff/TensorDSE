schema_path = ""

def load_source_model(path):
    from utils import load_json

    model = load_json(path)
    return model


def validate_schema(log, main_dir):
    import os
    import urllib.request

    global schema_path
    schema_path = os.path.join(main_dir,"schema","schema.fbs")

    if not os.path.exists(schema_path):    
        urllib.request.urlretrieve(schema_path,"schema.fbs")


def convert_to_json(file_path, filename):
    import os
    convert_json_cmd = f"flatc -t --strict-json --defaults-json {schema_path} -- {file_path}"

    os.system(convert_json_cmd)
    os.system(f"mv {filename}.json models/json/")


def json_conversion_manager(tflite_model):
    import os
    import json

    pass

        
if __name__ == '__main__':
    import sys
    import os
    import argparse
    from utils import deduce_filename

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-m', '--model',
                        default="models/source_models/MNIST_model.tflite",
                        help='File path to the SOURCE .tflite file.')

    args = parser.parse_args()

    model = args.model
    model_name = "MNIST_model"
    model_name = deduce_filename(model)
    assert (model_name != None), "Model name was not able to be deduced."

    project_dir = os.path.dirname(os.path.abspath(__file__))
    schema_path = os.path.join(project_dir, "schema", "schema.fbs")
    json_dir = os.path.join(project_dir,"models/json")

    convert_to_json(model, model_name)

