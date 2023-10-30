import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os



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
            "-r",
            "--resultspath",
            default="resources/deployment_results/",
            help="Path to results file"
            )

    parser.add_argument(
            "-m",
            "--mode",
            default="all",
            help="Plot mode"
            )
    
    return parser.parse_args()


if __name__ == "__main__":
    import os

    args = getArgs()

    if os.path.isfile(args.resultspath):
        if args.mode == "bar":
            plot_bar_chart(args.resultspath)
        elif args.mode == "pie":
            plot_pie_chart(args.resultspath)
        elif args.mode == "gantt":
            plot_gantt_chart(args.resultspath)
        elif args.mode == "all":
            plot_bar_chart(args.resultspath)
            plot_pie_chart(args.resultspath)
            plot_gantt_chart(args.resultspath)
        else:
            print(f"The provided Plot mode is not supported. (Mode: {args.mode})")
    else:
        print("The provided Path to the results file is not valid. (Path: {args.resultspath})")


    
