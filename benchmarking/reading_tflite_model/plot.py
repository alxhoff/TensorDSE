class UsbStats():
    def __init__(self):
        pass

    def append_avgs(self, host_comms_avg, host_submission_avg,
                    tpu_comms_avg, tpu_return_avg,
                    inference_avg, total_avg):

        self.host_comms_avg = host_comms_avg
        self.host_submission_avg = host_submission_avg

        self.tpu_comms_avg = tpu_comms_avg
        self.tpu_return_avg = tpu_return_avg

        self.inference_avg = inference_avg
        self.total_avg = total_avg

    def append_stds(self, host_comms_std, host_submission_std,
                    tpu_comms_std, tpu_return_std,
                    inference_std, total_std):

        self.host_comms_std = host_comms_std
        self.host_submission_std = host_submission_std

        self.tpu_comms_std = tpu_comms_std
        self.tpu_return_std = tpu_return_std

        self.inference_std = inference_std
        self.total_std = total_std


class UsbTimes():
    def __init__(self):

        self.host_comms_array = []
        self.host_submission_array = []

        self.tpu_comms_array = []
        self.tpu_return_array = []

        self.inference_array = []
        self.total_array = []

    def append_times(self, host_comms_time, host_submission_time, 
                            tpu_comms_time, tpu_return_time,
                            inference_time, total_time):

        self.host_comms_array.append(host_comms_time)
        self.host_submission_array.append(host_submission_time)

        self.tpu_comms_array.append(tpu_comms_time)
        self.tpu_return_array.append(tpu_return_time)

        self.inference_array.append(inference_time)
        self.total_array.append(total_time)


def parse_plot_csv(filename):
    import csv

    ret = []

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            ret.append(row)
            
    return ret


def deduce_plot_filesize(model_name):
    import os
    from os import listdir
    from os.path import isfile, isdir, join

    num = model_name.count('_')
    filesize = model_name.split("_")[num]

    return filesize


def deduce_plot_ops(folder, filename):
    import os
    from os import listdir
    from os.path import isfile, isdir, join

    plot_info = []

    for dirs in listdir(folder):
        res_path = f"{folder}{dirs}"
        op = dirs
        for results in listdir(res_path):
            if results == "Results.csv":
                cur_path = f"{res_path}/{results}"
                plot_info.append([op, cur_path])

    return plot_info


def find_raw_means(cpu_r, edge_r):
    cnt = 0
    cpu_mean = 0
    edge_mean = 0

    for c,e in zip(cpu_r, edge_r):
        cpu_mean = cpu_mean + c
        edge_mean = edge_mean + e
        cnt += 1

    return ((cpu_mean/cnt) * 10**6), ((edge_mean/cnt) * 10**6)


def integrate_csv(usb_f, cpu_r, edge_r):
    import csv

    with open(usb_f, 'a+') as csvfile:
        fw = csv.writer(csvfile, delimiter=',', quotechar='|',
                        quoting=csv.QUOTE_MINIMAL)

        # MEAN STDDEV MEDIAN

        fw.writerow(["edge"])
        fw.writerow([
                        "mean", "stddev", "median"
                    ])
        fw.writerow([
                        10**6*float(edge_r[0]), 10**6*float(edge_r[0]), 10**6*float(edge_r[0])
                    ])

        fw.writerow([])

        fw.writerow(["cpu"])
        fw.writerow([
                        "mean", "stddev", "median"
                    ])
        fw.writerow([
                        10**6*float(cpu_r[0]), 10**6*float(cpu_r[0]), 10**6*float(cpu_r[0])
                    ])


def integrate_results(usb_results_file, op):
    import os
    from os import listdir
    from os.path import isfile, isdir, join
    from utils import deduce_filename, parse_csv

    cpu_results_dir = "results/cpu"
    edge_results_dir = "results/edge"

    op = deduce_filename(op, ".csv")

    for dirs in listdir(cpu_results_dir):
        if dirs in op:
            cpu_filepath = f"{cpu_results_dir}/{dirs}/Results.csv"
            edge_filepath = f"{edge_results_dir}/{dirs}/Results.csv"

            cpu_results = parse_csv(cpu_filepath)
            edge_results = parse_csv(edge_filepath)

            cpu_results, edge_results = find_python_stats(cpu_results, edge_results)

            integrate_csv(usb_results_file, cpu_results, edge_results)


def read_timestamps(filename):
    import os
    import csv
    assert (os.path.exists(filename)), "File doesnt exist."

    usb_times = UsbTimes()

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        header = True
        for row in csv_reader:
            if (not header):
                host_comms_time = float(row[1]) - float(row[0])
                host_submission_time = float(row[2]) - float(row[1])

                tpu_comms_time = float(row[4]) - float(row[3])
                tpu_return_time = float(row[5]) - float(row[4])

                inference_time = float(row[4]) - float(row[2])
                total_time = float(row[5]) - float(row[0])

                if (host_comms_time > 0 and host_submission_time > 0
                    and tpu_comms_time > 0 and tpu_return_time > 0 
                    and inference_time > 0):

                    usb_times.append_times(host_comms_time, host_submission_time, 
                                            tpu_comms_time, tpu_return_time,
                                            inference_time, total_time)


            header = False

    return usb_times


def find_stats(values):
    import statistics

    usb_stats = UsbStats()

    host_comms_avg = statistics.mean(values.host_comms_array)
    host_comms_std = statistics.stdev(values.host_comms_array)

    host_submission_avg = statistics.mean(values.host_submission_array)
    host_submission_std = statistics.stdev(values.host_submission_array)


    tpu_comms_avg = statistics.mean(values.tpu_comms_array)
    tpu_comms_std = statistics.stdev(values.tpu_comms_array)

    tpu_return_avg = statistics.mean(values.tpu_return_array)
    tpu_return_std = statistics.stdev(values.tpu_return_array)

    inference_avg = statistics.mean(values.inference_array)
    inference_std = statistics.stdev(values.inference_array)

    total_avg = statistics.mean(values.total_array)
    total_std = statistics.stdev(values.total_array)

    usb_stats.append_avgs(host_comms_avg, host_submission_avg, 
                            tpu_comms_avg, tpu_return_avg,
                            inference_avg, total_avg)

    usb_stats.append_stds(host_comms_std, host_submission_std, 
                            tpu_comms_std, tpu_return_std,
                            inference_std, total_std)

    return usb_stats

def find_python_stats(cpu_r, edge_r):
    import statistics

    cpu_r = [statistics.mean(cpu_r), statistics.stdev(cpu_r), statistics.median(cpu_r)]
    edge_r = [statistics.mean(edge_r), statistics.stdev(edge_r), statistics.median(edge_r)]

    return cpu_r, edge_r


def store_stats(filename, stats, filesize):
    import os
    import csv

    from utils import extend_directory

    csv_dir = extend_directory("results/plot/", filename)
    csv_file = f"{csv_dir}/Results.csv"

    with open(csv_file, 'w') as csvfile:
        fw = csv.writer(csvfile, delimiter=',', quotechar='|',
                        quoting=csv.QUOTE_MINIMAL)

        fw.writerow(["host_comms_avg",
                     "host_submission_avg", 
                     "tpu_comms_avg",
                     "tpu_return_avg", 
                     "inference_avg",
                     "total_avg"
                     ])

        fw.writerow([10**6 * float(stats.host_comms_avg), 
                     10**6 * float(stats.host_submission_avg),
                     10**6 * float(stats.tpu_comms_avg),
                     10**6 * float(stats.tpu_return_avg),
                     10**6 * float(stats.inference_avg),
                     10**6 * float(stats.total_avg)
                     ])

        fw.writerow(["host_comms_std",
                     "host_submission_std", 
                     "tpu_comms_std",
                     "tpu_return_std", 
                     "inference_std",
                     "total_std"
                     ])

        fw.writerow([10**6 * float(stats.host_comms_std), 
                     10**6 * float(stats.host_submission_std),
                     10**6 * float(stats.tpu_comms_std),
                     10**6 * float(stats.tpu_return_std),
                     10**6 * float(stats.inference_std),
                     10**6 * float(stats.total_std)
                     ])

        fw.writerow([])

        fw.writerow(["data_size", filesize])

        fw.writerow([])

    return csv_file


def plot_manager(folder):
    import os
    models_info = deduce_plot_ops(folder, "Results.csv")

    for model_info in models_info:
        op = model_info[0]
        filepath = model_info[1]
        plot_single_manager(filepath, op)

def plot_single_manager(filepath, op=""):
    import os
    import logging
    from utils import deduce_filename

    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    log.info(f"Compiling plot results: {op}...")
    filename_fdr = os.path.dirname(
                    os.path.abspath(filepath))

    model_name = deduce_filename(filename_fdr, ending=None)
    filesize = deduce_plot_filesize(model_name)

    times = read_timestamps(filepath)
    stats = find_stats(times)
    usb_results_file = store_stats(model_name, stats, filesize)

    integrate_results(usb_results_file, filename_fdr)


if __name__ == '__main__':
    import os
    from utils import deduce_filename
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-m', '--mode', required=False,
                        default="Single",
                        help='Mode in which the script will run: All or Single.')

    parser.add_argument('-f', '--folder', required=False,
                        default="results/usb/",
                        help='Folder.')

    parser.add_argument('-t', '--target', required=False,
                        default="",
                        help='Model.')

    args = parser.parse_args()
    if (args.mode == "All" and args.folder != ""):
        plot_manager(args.folder)

    elif (args.mode == "Single" and args.target != ""):
        filepath = args.target
        plot_single_manager(filepath)
    else:
        print("Invaild arguments.")
