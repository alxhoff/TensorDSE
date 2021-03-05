class UsbTimes():
    def __init__(self):
        self.positives_arr = []
        self.negatives_arr = []

        self.host_data_arr = []
        self.tpu_comms_arr = []
        self.tpu_data_arr = []
        self.infer_arr = []
        self.total_arr = []

    def get_size_negatives(self):
        return len(self.negatives_arr)

    def get_size_positives(self):
        return len(self.positives_arr)

    def append_negatives(self, host_sub, tpu_comms, tpu_return, 
                            inference, total, sess):
        self.negatives_arr.append(
                    [host_sub, tpu_comms, tpu_return, inference, total, sess]
                )

    def append_positives(self, host_sub, tpu_comms, tpu_return, 
                            inference, total):
        self.positives_arr.append(
                    [host_sub, tpu_comms, tpu_return, inference, total]
                )

    def print_negatives(self):
        from tabulate import tabulate
        table = []

        table.append(
                ["HOST_SUB", "TPU_COMMS", "TPU_RET", "INFER", "TOTAL", "SESSION"])
        for arr in self.negatives_arr:
            table.append([    
                arr[0] if float(arr[0]) <= 0 else "    ",
                arr[1] if float(arr[1]) <= 0 else "    ",
                arr[2] if float(arr[2]) <= 0 else "    ",
                arr[3] if float(arr[3]) <= 0 else "    ",
                arr[4] if float(arr[4]) <= 0 else "    ",
                arr[5]
                ])
            pass

        print(f"NEGATIVE VALUES\n{tabulate(table)}")

    def fetch_column(self, col):
        samples = []

        col_dict = {
        "HOST_DATA" : 0,
        "TPU_COMMS" : 1,
        "TPU_DATA"  : 2,
        "INFERENCE" : 3,
        "TOTAL"     : 4,
        }
        default = None
        col = col_dict.get(col, default)

        if type(col) == int:
            for arr in self.positives_arr:
                samples.append(arr[col])

        return samples
    
    def create_stats(self, multi_arr, sessions):
        import itertools
        from statistics import mean, stdev, median

        tmp_initial_comms = []
        tmp_h_data = []
        tmp_t_comms = []
        tmp_t_data = []
        tmp_infer= []
        tmp_total = []
        length = len(multi_arr[0])
        for s in range(sessions):
            for i in range(length):
                tmp_h_data.append(multi_arr[0][i][s])
                tmp_t_comms.append(multi_arr[1][i][s])
                tmp_t_data.append(multi_arr[2][i][s])
                tmp_infer.append(multi_arr[3][i][s])
                tmp_total.append(multi_arr[4][i][s])

            self.host_data_arr += [mean(tmp_h_data), stdev(tmp_h_data), median(tmp_h_data)]
            self.tpu_comms_arr += [mean(tmp_t_comms), stdev(tmp_t_comms), median(tmp_t_comms)]
            self.tpu_data_arr += [mean(tmp_t_data), stdev(tmp_t_data), median(tmp_t_data)]
            self.infer_arr += [mean(tmp_infer), stdev(tmp_infer), median(tmp_infer)]
            self.total_arr += [mean(tmp_total), stdev(tmp_total), median(tmp_total)]


def integrate_csv(c_file, title, res):
    import csv

    with open(c_file, 'a+') as csvfile:
        fw = csv.writer(csvfile, delimiter=',', quotechar='|',
                        quoting=csv.QUOTE_MINIMAL)
        fw.writerow([title])
        fw.writerow(["mean", "stddev", "median"])
        fw.writerow([10**6*float(res[0]), 10**6*float(res[1]), 10**6*float(res[2])])
        fw.writerow([])


def integrate_results(usb_results_file, op):
    import os
    from os import listdir
    from os.path import isfile
    from utils import parse_csv, deduce_filename

    cpu_results_dir = "results/cpu"
    edge_results_dir = "results/edge"

    # Basically cut the ending of the string that is preceded by a '_'
    # Which means cut the appended filesize, leaving the op name.
    op = (op.rstrip(op.split("_")[op.count("_")])).strip("_")

    for dirs in listdir(edge_results_dir):
        if dirs == op:
            cpu_res_filepath = f"{cpu_results_dir}/{dirs}/Results.csv"
            cpu_first_filepath = f"{cpu_results_dir}/{dirs}/Firsts.csv"
            edge_res_filepath = f"{edge_results_dir}/{dirs}/Results.csv"
            edge_first_filepath = f"{edge_results_dir}/{dirs}/Firsts.csv"

            if isfile(cpu_res_filepath):
                cpu_results = retreive_array_stats(parse_csv(cpu_res_filepath))
                edge_results = retreive_array_stats(parse_csv(edge_res_filepath))
                edge_first_results = retreive_array_stats(parse_csv(edge_first_filepath))

                integrate_csv(usb_results_file, "first_edge", edge_first_results)
                integrate_csv(usb_results_file, "edge", edge_results)
                integrate_csv(usb_results_file, "cpu", cpu_results)
            else:
                edge_results = retreive_array_stats(parse_csv(edge_res_filepath))
                edge_first_results = retreive_array_stats(parse_csv(edge_first_filepath))
                integrate_csv(usb_results_file, "first_edge", edge_first_results)
                integrate_csv(usb_results_file, "edge", edge_results)
            break


def store_usb_stats(usb_stats, first_usb_stats, comms_stats, filename, filesize, valid_str, sessions):
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

        fw.writerow(["initial_comms_mean", "initial_comms_stdev", "inital_comms_median"])
        fw.writerow([10**6 * float(comms_stats[0]), 
                        10**6 * float(comms_stats[1]),
                        10**6 * float(comms_stats[2])
                        ])

        fw.writerow([])


        for s in range(sessions):
            fw.writerow([f"Session: {s+1}"])

            fw.writerow(["first_host_submission_mean", 
                         "first_tpu_comms_mean",
                         "first_tpu_return_mean", 
                         "first_inference_mean",
                         "first_total_mean"
                         ])

            fw.writerow([10**6 * (float(first_usb_stats.host_data_arr[0 + s*3])),
                         10**6 * (float(first_usb_stats.tpu_comms_arr[0 + s*3])),
                         10**6 * (float(first_usb_stats.tpu_data_arr[0 + s*3])),
                         10**6 * (float(first_usb_stats.infer_arr[0 + s*3])),
                         10**6 * (float(first_usb_stats.total_arr[0 + s*3]))
                         ])

            fw.writerow([])

            fw.writerow(["host_submission_mean", 
                         "tpu_comms_mean",
                         "tpu_return_mean", 
                         "inference_mean",
                         "total_mean"
                         ])

            fw.writerow([10**6 * (float(usb_stats.host_data_arr[0 + s*3])),
                         10**6 * (float(usb_stats.tpu_comms_arr[0 + s*3])),
                         10**6 * (float(usb_stats.tpu_data_arr[0 + s*3])),
                         10**6 * (float(usb_stats.infer_arr[0 + s*3])),
                         10**6 * (float(usb_stats.total_arr[0 + s*3]))
                         ])

            fw.writerow([])

            fw.writerow(["host_submission_stdev", 
                         "tpu_comms_stdev",
                         "tpu_return_stdev", 
                         "inference_stdev",
                         "total_stdev"
                         ])

            fw.writerow([10**6 * (float(usb_stats.host_data_arr[1 + s*3])),
                         10**6 * (float(usb_stats.tpu_comms_arr[1 + s*3])),
                         10**6 * (float(usb_stats.tpu_data_arr[1 + s*3])),
                         10**6 * (float(usb_stats.infer_arr[1 + s*3])),
                         10**6 * (float(usb_stats.total_arr[1 + s*3]))
                         ])

            fw.writerow([])

            fw.writerow(["host_submission_med", 
                         "tpu_comms_med",
                         "tpu_return_med", 
                         "inference_med",
                         "total_med"
                         ])

            fw.writerow([10**6 * (float(usb_stats.host_data_arr[2 + s*3])), 
                         10**6 * (float(usb_stats.tpu_comms_arr[2 + s*3])), 
                         10**6 * (float(usb_stats.tpu_data_arr[2 + s*3])), 
                         10**6 * (float(usb_stats.infer_arr[2 + s*3])),
                         10**6 * (float(usb_stats.total_arr[2 + s*3]))
                         ])

            fw.writerow([])

            fw.writerow(["host_data_perc",
                         "tpu_comms_perc",
                         "tpu_data_perc", 
                         "inference_perc",
                         "total_perc"
                         ])

            fw.writerow([10**2 * (float(usb_stats.host_data_arr[0 + s*3])/float(usb_stats.total_arr[0 + s*3])), 
                         10**2 * (float(usb_stats.tpu_comms_arr[0 + s*3])/float(usb_stats.total_arr[0 + s*3])), 
                         10**2 * (float(usb_stats.tpu_data_arr[0 + s*3])/float(usb_stats.total_arr[0 + s*3])), 
                         10**2 * (float(usb_stats.infer_arr[0 + s*3])/float(usb_stats.total_arr[0 + s*3])), 
                         10**2 * (1)
                         ])

            fw.writerow([])
    return csv_file


def retreive_array_stats(arr):
    from statistics import mean, stdev, median
    if len(arr) == 1:
        return [ arr[0], arr[0], arr[0] ]
    else:
        return [ mean(arr), stdev(arr), median(arr) ]


def retreive_usb_stats(values, sessions):
    from statistics import mean, stdev, median
    i = 0

    multi_arr = []
    arr_0 = []
    arr_1 = []
    arr_2 = []
    arr_3 = []
    arr_4 = []

    h_data  = values.fetch_column("HOST_DATA")
    t_comms = values.fetch_column("TPU_COMMS")
    t_data  = values.fetch_column("TPU_DATA")
    infer   = values.fetch_column("INFERENCE")
    total   = values.fetch_column("TOTAL")

    while (i < values.get_size_positives()):
        tmp_h_data = []
        tmp_t_comms = []
        tmp_t_data = []
        tmp_infer= []
        tmp_total = []

        for _ in range(sessions):
            tmp_h_data.append(float(h_data[i]))
            tmp_t_comms.append(float(t_comms[i]))
            tmp_t_data.append(float(t_data[i]))
            tmp_infer.append(float(infer[i]))
            tmp_total.append(float(total[i]))
            i += 1

        arr_0.append(tmp_h_data) 
        arr_1.append(tmp_t_comms) 
        arr_2.append(tmp_t_data) 
        arr_3.append(tmp_infer)
        arr_4.append(tmp_total)
            
            
    multi_arr =  [arr_0, arr_1, arr_2, arr_3, arr_4]
    values.create_stats(multi_arr, sessions)
    return values


def read_usb_results(filename, sessions):
    import os
    import sys
    import csv
    import logging
    assert (os.path.exists(filename)), "File doesnt exist."

    usb_times = UsbTimes()
    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    v = 0
    i = 0
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        header = True
        for row in csv_reader:
            if not header:
                for j in range(sessions):
                    i += 1
                    if (any(float(x) <= 0 for x in row)):
                        usb_times.append_negatives(
                                row[0 + j*5], # host submission
                                row[1 + j*5], # tpu comms
                                row[2 + j*5], # tpu return
                                row[3 + j*5], # inference
                                row[4 + j*5], # total
                                j)
                    else:
                        v += 1
                        usb_times.append_positives(
                                row[0 + j*5], # host submission
                                row[1 + j*5], # tpu comms
                                row[2 + j*5], # tpu return
                                row[3 + j*5], # inference
                                row[4 + j*5]  # total
                                )
            header = False

    log.info(f"Valid: {v}/{i}")

    if v != i:
        usb_times.print_negatives()
        input("Continue:")

    if v == 0: 
        sys.exit("NO VALID RESULTS FOUND.")

    return usb_times, f"{v}/{i}"


def plot_manager(op, filesize, sessions):
    import os
    import logging
    from utils import deduce_filename, parse_csv

    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    log.info(f"Compiling plot results: {op}...")

    results_file=f"results/usb/{op}/Results.csv"
    firsts_file=f"results/usb/{op}/Firsts.csv"
    comms_file=f"results/usb/{op}/Comms.csv"

    values, valid_str = read_usb_results(results_file, sessions)
    first_values, _ = read_usb_results(firsts_file, sessions)

    values_stats = retreive_usb_stats(values, sessions)
    first_values_stats = retreive_usb_stats(first_values, sessions)
    comms_stats = retreive_array_stats(parse_csv(comms_file))

    usb_results_file = store_usb_stats(
            values_stats, first_values_stats, comms_stats, op, filesize, valid_str, sessions)
    integrate_results(usb_results_file, op)


if __name__ == '__main__':
    import os
    from utils import deduce_sessions_nr
    from utils import deduce_filename
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-m', '--mode', required=False,
                        default="Single",
                        help='Mode in which the script will run: Group or Single.')

    parser.add_argument('-f', '--folder', required=False,
                        default="results/usb/",
                        help='Folder.')

    parser.add_argument('-t', '--target', required=False,
                        default="",
                        help='Model.')

    args = parser.parse_args()

    if (args.mode == "Single" and args.target != ""):
        op = os.path.dirname(args.target)
        op = op.split("/")[op.count("/")]
        filesize = op.split("_")[op.count("_")]
        sessions = deduce_sessions_nr(op)
        plot_manager(op, filesize, sessions)

    else:
        print("Invaild arguments.")
