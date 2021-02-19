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

        self.neg_values = []

    def append_times(self, host_comms_time, host_submission_time, 
                            tpu_comms_time, tpu_return_time,
                            inference_time, total_time,
                            sessions_nr):

        if sessions_nr == 0:
            self.host_comms_array.append(host_comms_time)
            self.host_submission_array.append(host_submission_time)
            self.tpu_comms_array.append(tpu_comms_time)
            self.tpu_return_array.append(tpu_return_time)
            self.inference_array.append(inference_time)
            self.total_array.append(total_time)
        else:
            length = len(self.host_comms_array)

            if length > 0:
                tmp = self.host_comms_array[length - 1] 
                self.host_comms_array[length -1] = [tmp, host_comms_time]

                tmp = self.host_submission_array[length -1] 
                self.host_submission_array[length -1] = [tmp, host_submission_time]

                tmp = self.tpu_comms_array[length -1] 
                self.tpu_comms_array[length -1] = [tmp, tpu_comms_time]

                tmp = self.tpu_return_array[length -1] 
                self.tpu_return_array[length -1] = [tmp, tpu_return_time]

                tmp = self.inference_array[length -1] 
                self.inference_array[length -1] = [tmp, inference_time]

                tmp = self.total_array[length -1] 
                self.total_array[length -1] = [tmp, total_time]

            else:
                tmp = " "
                self.host_comms_array.append([tmp, host_comms_time])
                self.host_submission_array.append([tmp, host_submission_time])
                self.tpu_comms_array.append([tmp, tpu_comms_time])
                self.tpu_return_array.append([tmp, tpu_return_time])
                self.inference_array.append([tmp, inference_time])
                self.total_array.append([tmp, total_time])

    def append_neg_times(self, host_comms_time, host_submission_time, 
                                tpu_comms_time, tpu_return_time,
                                inference_time, total_time,
                                session_nr):

        self.neg_values.append([host_comms_time, host_submission_time, 
                                tpu_comms_time, tpu_return_time,
                                inference_time, total_time,
                                session_nr])

    def print_neg_values(self):
        from tabulate import tabulate
        table = []

        table.append(["HOST_COMMS", "HOST_SUB", "TPU_COMMS", "TPU_RET", "INFER", "TOTAL", "SESSION"])
        for arr in self.neg_values:
            table.append([    
                arr[0] if float(arr[0]) < 0 else "    ",
                arr[1] if float(arr[1]) < 0 else "    ",
                arr[2] if float(arr[2]) < 0 else "    ",
                arr[3] if float(arr[3]) < 0 else "    ",
                arr[4] if float(arr[4]) < 0 else "    ",
                arr[5] if float(arr[5]) < 0 else "    ",
                arr[6]
                ])
            pass

        print(f"NEGATIVE VALUES\n{tabulate(table)}")


def parse_plot_csv(filename):
    import csv

    ret = []

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            ret.append(row)
            
    return ret

def deduce_plot_filename(model_name):
    num = model_name.count('_')
    op = ""

    for i in range(num):
        if i < num - 1:
            op = f"{op}{model_name.split('_')[i]}_"
        else:
            op = f"{op}{model_name.split('_')[i]}"

    return op


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
                        10**6*float(edge_r[0]), 10**6*float(edge_r[1]), 10**6*float(edge_r[2])
                    ])

        fw.writerow([])

        if cpu_r != None:
            fw.writerow(["cpu"])
            fw.writerow([
                            "mean", "stddev", "median"
                        ])
            fw.writerow([
                            10**6*float(cpu_r[0]), 10**6*float(cpu_r[1]), 10**6*float(cpu_r[2])
                        ])


def integrate_results(usb_results_file, op):
    import os
    from os import listdir
    from os.path import isfile
    from utils import parse_csv, deduce_filename

    cpu_results_dir = "results/cpu"
    edge_results_dir = "results/edge"

    op = deduce_filename(op)
    op = deduce_plot_filename(op)

    for dirs in listdir(edge_results_dir):
        if dirs == op:
            cpu_filepath = f"{cpu_results_dir}/{dirs}/Results.csv"
            edge_filepath = f"{edge_results_dir}/{dirs}/Results.csv"

            if isfile(cpu_filepath):
                cpu_results = parse_csv(cpu_filepath)
                edge_results = parse_csv(edge_filepath)
                cpu_results, edge_results = find_python_stats(cpu_results, edge_results)
            else:
                cpu_results = None
                edge_results = parse_csv(edge_filepath)
                edge_results = find_python_stats(cpu_results, edge_results)


            integrate_csv(usb_results_file, cpu_results, edge_results)


def read_timestamps(filename, sessions):
    import os
    import sys
    import csv
    import logging
    assert (os.path.exists(filename)), "File doesnt exist."

    usb_timer_arr = []
    usb_times = UsbTimes()

    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    cnt = 0
    v_cnt = 0
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        header = True
        for row in csv_reader:
            if not header:
                cnt += 1
                for i in range(sessions):
                    expr = 6 * (i)
                    host_comms_time = float(float(row[1 + expr]) - float(row[0 + expr]))
                    host_submission_time = float(float(row[2 + expr]) - float(row[1 + expr]))

                    tpu_comms_time = float(float(row[4 + expr]) - float(row[3 + expr]))
                    tpu_return_time = float(float(row[5 + expr]) - float(row[4 + expr]))

                    print(f"Sess: {i + 1} - Return Time {tpu_return_time}")

                    inference_time = float(float(row[5 + expr]) - float(row[2 + expr]))
                    total_time = float(float(row[5 + expr]) - float(row[0 + expr]))

                    if (host_comms_time > 0 and host_submission_time > 0
                        and tpu_comms_time > 0 and tpu_return_time > 0 
                        and inference_time > 0):

                        v_cnt += 1
                        usb_times.append_times(
                                host_comms_time, host_submission_time, 
                                tpu_comms_time, tpu_return_time,
                                inference_time, total_time,
                                i)

                    else:
                        usb_times.append_neg_times(
                                host_comms_time, host_submission_time, 
                                tpu_comms_time, tpu_return_time,
                                inference_time, total_time, 
                                i)

            header = False


    if len(usb_times.neg_values) > 0:
        print(f"\n{filename}:")
        usb_times.print_neg_values()
        print("\n")

    if v_cnt == 0: 
        sys.exit("NO VALID RESULTS FOUND.")

    log.info(f"Valid: {v_cnt}/{cnt*sessions}")
    return usb_times, f"{v_cnt}/{cnt*sessions}"


def find_stats(values, sessions):
    import copy
    from statistics import mean, stdev

    usb_stats_arr = []
    for i in range(sessions):
        usb_stats = UsbStats()

        host_comms_avg = mean(values.host_comms_array[i])
        host_comms_std = stdev(values.host_comms_array[i])

        host_submission_avg = mean(values.host_submission_array[i])
        host_submission_std = stdev(values.host_submission_array[i])


        tpu_comms_avg = mean(values.tpu_comms_array[i])
        tpu_comms_std = stdev(values.tpu_comms_array[i])

        tpu_return_avg = mean(values.tpu_return_array[i])
        tpu_return_std = stdev(values.tpu_return_array[i])

        inference_avg = mean(values.inference_array[i])
        inference_std = stdev(values.inference_array[i])

        total_avg = mean(values.total_array[i])
        total_std = stdev(values.total_array[i])

        usb_stats.append_avgs(host_comms_avg, host_submission_avg, 
                                tpu_comms_avg, tpu_return_avg,
                                inference_avg, total_avg)

        usb_stats.append_stds(host_comms_std, host_submission_std, 
                                tpu_comms_std, tpu_return_std,
                                inference_std, total_std)

        tmp = copy.deepcopy(usb_stats)
        usb_stats_arr.append(tmp)

    return usb_stats_arr

def find_python_stats(cpu_r, edge_r):
    import statistics

    if cpu_r != None:
        cpu_r = [statistics.mean(cpu_r), statistics.stdev(cpu_r), statistics.median(cpu_r)]
        edge_r = [statistics.mean(edge_r), statistics.stdev(edge_r), statistics.median(edge_r)]
        return cpu_r, edge_r
    else:
        edge_r = [statistics.mean(edge_r), statistics.stdev(edge_r), statistics.median(edge_r)]
        return edge_r


def store_stats(filename, stats, filesize, valid_str, sessions):
    import os
    import csv

    from utils import extend_directory

    csv_dir = extend_directory("results/plot/", filename)
    csv_file = f"{csv_dir}/Results.csv"

    with open(csv_file, 'w') as csvfile:
        fw = csv.writer(csvfile, delimiter=',', quotechar='|',
                        quoting=csv.QUOTE_MINIMAL)


        fw.writerow(["Operation", filename])
        fw.writerow(["Data Size", filesize])
        fw.writerow(["valid", valid_str])
        fw.writerow(["sessions", sessions])

        fw.writerow([])

        for i in range(sessions):
            fw.writerow([f"Session: {i+1}"])
            fw.writerow(["host_comms_avg",
                         "host_submission_avg", 
                         "tpu_comms_avg",
                         "tpu_return_avg", 
                         "inference_avg",
                         "total_avg"
                         ])

            fw.writerow([10**6 * float(stats[i].host_comms_avg), 
                         10**6 * float(stats[i].host_submission_avg),
                         10**6 * float(stats[i].tpu_comms_avg),
                         10**6 * float(stats[i].tpu_return_avg),
                         10**6 * float(stats[i].inference_avg),
                         10**6 * float(stats[i].total_avg)
                         ])

            fw.writerow([])
            fw.writerow(["host_comms_std",
                         "host_submission_std", 
                         "tpu_comms_std",
                         "tpu_return_std", 
                         "inference_std",
                         "total_std"
                         ])

            fw.writerow([10**6 * float(stats[i].host_comms_std), 
                         10**6 * float(stats[i].host_submission_std),
                         10**6 * float(stats[i].tpu_comms_std),
                         10**6 * float(stats[i].tpu_return_std),
                         10**6 * float(stats[i].inference_std),
                         10**6 * float(stats[i].total_std)
                         ])

            fw.writerow([])

    return csv_file

def validity_check(model_name, model_size, sessions):
    filename = f"results/usb/{model_name}_{model_size}/Results.csv"
    read_timestamps(filename, sessions)


def plot_manager(folder):
    from shark import deduce_sessions_nr
    import os
    models_info = deduce_plot_ops(folder, "Results.csv")

    for model_info in models_info:
        op = model_info[0]
        filepath = model_info[1]
        sessions = deduce_sessions_nr(op)
        plot_single_manager(filepath, sessions, op)

def plot_single_manager(filepath, sessions, op=""):
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

    values, valid_str = read_timestamps(filepath, sessions)
    stats = find_stats(values, sessions)
    usb_results_file = store_stats(model_name, stats, filesize, valid_str, sessions)

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
