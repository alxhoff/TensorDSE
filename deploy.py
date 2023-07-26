
def DeployLayer():
    pass

def DeployModel(model_path: str, model_summary_path: str) -> None:

    from utils.splitter.utils import ReadJSON
    from utils.splitter.logger import log
    from utils.model import Model

    model_name = (model_path.split("/")[-1]).split(".tflite")[0]

    if model_summary_path is not None:
        try:
            model_summary = ReadJSON(model_summary_path)
        except Exception as e:
            log.error(f"Exception occured while trying to read Model Summary!")

    for layer in model_summary["layers"]:
        m = DeployLayer(Model(layer, layer["mapping"], model_name))

if __name__ == "__main__":
    import os
    from profiler import GetArgs

    args = GetArgs()

    DeployModel(args.model, args.count, args.hardwaresummary, os.path.join(args.summaryoutputdir, "{}.json".format(args.summaryoutputname)))

    print("Model Deployed")
