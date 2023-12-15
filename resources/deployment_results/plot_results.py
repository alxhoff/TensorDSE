import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def compare_distributed_to_native(profiling_results_path: str, deployment_results_path: str):
    profiling_filename = os.path.splitext(os.path.basename(profiling_results_path))[0]
    deployment_filename = os.path.splitext(os.path.basename(deployment_results_path))[0]

    if profiling_filename == deployment_filename:
        profiling_data = pd.read_json(profiling_results_path)["models"][0]
        deployment_data = pd.read_json(deployment_results_path)
        cpu_total = 0
        gpu_total = 0
        tpu_total = 0
        for layer in profiling_data["layers"]:
            cpu_total += layer["delegates"][0]["mean"]
            gpu_total += layer["delegates"][1]["mean"]
            tpu_total += layer["delegates"][2]["mean"]

        # Getting total inference time from deployment results
        total_inference_time = deployment_data["total_inference_time (s)"]

        # Plotting the bar chart
        categories = ["CPU", "GPU", "TPU", "Total Inference Time"]
        values = [cpu_total, gpu_total, tpu_total, total_inference_time]
        
        plt.figure(figsize=(10, 5))
        plt.bar(categories, values, color=['blue', 'green', 'red', 'purple'])
        plt.ylabel('Time (s)')
        plt.title('Comparison of Layer-wise Accumulated Time and Total Inference Time')
        plt.savefig(os.path.join(os.path.dirname(deployment_results_path), f"{deployment_filename}_compare_native.png"))
        plt.close()


def plot_bar_chart(json_path):
    filename = os.path.splitext(os.path.basename(json_path))[0]

    data = pd.read_json(json_path)
    submodels = data['submodels']
    names = [sm['name'] for sm in submodels]
    times = [sm['inference_time (s)'] for sm in submodels]
    hardware = [sm['layers'][0]['mapping'] for sm in submodels]

    plt.bar(names, times, color=['red' if h == 'cpu0' else 'green' if h == 'gpu0' else 'blue' for h in hardware])
    plt.xlabel('Submodel')
    plt.ylabel('Inference Time (s)')
    plt.title('Inference Time by Submodel')
    plt.savefig(os.path.join(os.path.dirname(json_path), f"{filename}_bar.png"))
    plt.close()


def plot_pie_chart(json_path):
    filename = os.path.splitext(os.path.basename(json_path))[0]
    data = pd.read_json(json_path)
    submodels = data['submodels']
    hardware = [sm['layers'][0]['mapping'] for sm in submodels]
    hardware_counts = pd.Series(hardware).value_counts()

    plt.pie(hardware_counts, labels=hardware_counts.index, autopct='%1.1f%%')
    plt.title('Hardware Distribution')
    plt.savefig(os.path.join(os.path.dirname(json_path), f"{filename}_pie.png"))
    plt.close()


def plot_gantt_chart(json_path):
    filename = os.path.splitext(os.path.basename(json_path))[0]
    data = pd.read_json(json_path)
    submodels = data['submodels']
    
    layers = [layer['type'] for sm in submodels for layer in sm['layers']]
    times = [sm['inference_time (s)'] for sm in submodels for _ in sm['layers']]
    
    plt.barh(layers, times)
    plt.xlabel('Inference Time (s)')
    plt.ylabel('Layer')
    plt.title('Layer-wise Execution Timeline')
    plt.savefig(os.path.join(os.path.dirname(json_path), f"{filename}_gantt.png"))
    plt.close()


def getArgs():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )

    parser.add_argument(
            "-dr",
            "--deployresultspath",
            default="resources/deployment_results/",
            help="Path to results file"
            )

    parser.add_argument(
            "-pr",
            "--profileresultspath",
            default="resources/deployment_results/",
            help="Path to results file"
            )

    parser.add_argument(
            "-m",
            "--mode",
            default="compare",
            help="Plot mode"
            )
    
    return parser.parse_args()


if __name__ == "__main__":
    import os

    args = getArgs()

    if os.path.isfile(args.deployresultspath) and os.path.isfile(args.profileresultspath):
        if args.mode == "compare":
            compare_distributed_to_native(args.profileresultspath, args.deployresultspath)
        elif args.mode == "bar":
            plot_bar_chart(args.deployresultspath)
        elif args.mode == "pie":
            plot_pie_chart(args.deployresultspath)
        elif args.mode == "gantt":
            plot_gantt_chart(args.deployresultspath)
        elif args.mode == "all":
            plot_bar_chart(args.deployresultspath)
            plot_pie_chart(args.deployresultspath)
            plot_gantt_chart(args.deployresultspath)
        else:
            print(f"The provided Plot mode is not supported. (Mode: {args.mode})")
    else:
        print("The provided Paths to the results files are not valid. (Deployment Results Path: {args.deployresultspath} | Profiling Results Path: {args.profileresultspath})")


    
