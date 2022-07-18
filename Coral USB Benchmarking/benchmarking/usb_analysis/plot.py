class UsbTimes():
    """Class used to save the data regarding the usb analysis.
    """
    def __init__(self):
        self.negatives_arr = []

        self.host_data_arr = []
        self.tpu_comms_arr = []
        self.tpu_data_arr = []
        self.infer_arr = []
        self.total_arr = []

        self.host_data_stats = []
        self.tpu_comms_stats = []
        self.tpu_data_stats = []
        self.infer_stats = []
        self.total_stats = []

    def get_size_negatives(self):
        return len(self.negatives_arr)

    def append_negatives(self, host_sub, tpu_comms, tpu_return, 
                            inference, total, sess, run):
        self.negatives_arr.append(
                    [host_sub, tpu_comms, tpu_return, inference, total, sess, run]
                )

    def print_negatives(self):
        from tabulate import tabulate
        table = []

        table.append(
                ["HOST_SUB", "TPU_COMMS", "TPU_RET", "INFER", "TOTAL", "SESSION", "TEST_RUN"])
        for arr in self.negatives_arr:
            table.append([    
                arr[0] if float(arr[0]) <= 0 else "    ",
                arr[1] if float(arr[1]) <= 0 else "    ",
                arr[2] if float(arr[2]) <= 0 else "    ",
                arr[3] if float(arr[3]) <= 0 else "    ",
                arr[4] if float(arr[4]) <= 0 else "    ",
                arr[5],
                arr[6]
                ])
            pass

        print(f"NEGATIVE VALUES\n{tabulate(table)}")
    
    def create_stats(self, sessions):
        import itertools
        from statistics import mean, stdev, median

        length = len(self.host_data_arr)
        tmp_h_data = []
        tmp_t_comms = []
        tmp_t_data = []
        tmp_infer= []
        tmp_total = []
        for s in range(sessions):
            for i in range(length):
                tmp_h_data.append(self.host_data_arr[i][s])
                tmp_t_comms.append(self.tpu_comms_arr[i][s])
                tmp_t_data.append(self.tpu_data_arr[i][s])
                tmp_infer.append(self.infer_arr[i][s])
                tmp_total.append(self.total_arr[i][s])

            self.host_data_stats += [mean(tmp_h_data), stdev(tmp_h_data), median(tmp_h_data)]
            self.tpu_comms_stats += [mean(tmp_t_comms), stdev(tmp_t_comms), median(tmp_t_comms)]
            self.tpu_data_stats += [mean(tmp_t_data), stdev(tmp_t_data), median(tmp_t_data)]
            self.infer_stats += [mean(tmp_infer), stdev(tmp_infer), median(tmp_infer)]
            self.total_stats += [mean(tmp_total), stdev(tmp_total), median(tmp_total)]


def integrate_csv(c_file, title, res):
    """Appends CPU and Edge results to an existing CSV file.
    """
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


def store_usb_stats(
        usb_stats, first_usb_stats, comms_stats, 
        filename, filesize, valid_str, first_valid_str, sessions):
    """Stores the final results into a csv file that will be located at
    the results/plot folder.
    """
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
            fw.writerow(["valid", first_valid_str])

            fw.writerow(["first_host_submission_mean", 
                         "first_tpu_comms_mean",
                         "first_tpu_return_mean", 
                         "first_inference_mean",
                         "first_total_mean"
                         ])

            fw.writerow([10**6 * (float(first_usb_stats.host_data_stats[0 + s*3])),
                         10**6 * (float(first_usb_stats.tpu_comms_stats[0 + s*3])),
                         10**6 * (float(first_usb_stats.tpu_data_stats[0 + s*3])),
                         10**6 * (float(first_usb_stats.infer_stats[0 + s*3])),
                         10**6 * (float(first_usb_stats.total_stats[0 + s*3]))
                         ])

            fw.writerow([])

            fw.writerow(["first_host_submission_std", 
                         "first_tpu_comms_std",
                         "first_tpu_return_std", 
                         "first_inference_std",
                         "first_total_std"
                         ])

            fw.writerow([10**6 * (float(first_usb_stats.host_data_stats[1 + s*3])),
                         10**6 * (float(first_usb_stats.tpu_comms_stats[1 + s*3])),
                         10**6 * (float(first_usb_stats.tpu_data_stats[1 + s*3])),
                         10**6 * (float(first_usb_stats.infer_stats[1 + s*3])),
                         10**6 * (float(first_usb_stats.total_stats[1 + s*3]))
                         ])

            fw.writerow([])

            fw.writerow(["first_host_submission_med", 
                         "first_tpu_comms_med",
                         "first_tpu_return_med", 
                         "first_inference_med",
                         "first_total_med"
                         ])

            fw.writerow([10**6 * (float(first_usb_stats.host_data_stats[2 + s*3])),
                         10**6 * (float(first_usb_stats.tpu_comms_stats[2 + s*3])),
                         10**6 * (float(first_usb_stats.tpu_data_stats[2 + s*3])),
                         10**6 * (float(first_usb_stats.infer_stats[2 + s*3])),
                         10**6 * (float(first_usb_stats.total_stats[2 + s*3]))
                         ])

            fw.writerow([])

            fw.writerow(["host_submission_mean", 
                         "tpu_comms_mean",
                         "tpu_return_mean", 
                         "inference_mean",
                         "total_mean"
                         ])

            fw.writerow([10**6 * (float(usb_stats.host_data_stats[0 + s*3])),
                         10**6 * (float(usb_stats.tpu_comms_stats[0 + s*3])),
                         10**6 * (float(usb_stats.tpu_data_stats[0 + s*3])),
                         10**6 * (float(usb_stats.infer_stats[0 + s*3])),
                         10**6 * (float(usb_stats.total_stats[0 + s*3]))
                         ])

            fw.writerow([])

            fw.writerow(["host_submission_stdev", 
                         "tpu_comms_stdev",
                         "tpu_return_stdev", 
                         "inference_stdev",
                         "total_stdev"
                         ])

            fw.writerow([10**6 * (float(usb_stats.host_data_stats[1 + s*3])),
                         10**6 * (float(usb_stats.tpu_comms_stats[1 + s*3])),
                         10**6 * (float(usb_stats.tpu_data_stats[1 + s*3])),
                         10**6 * (float(usb_stats.infer_stats[1 + s*3])),
                         10**6 * (float(usb_stats.total_stats[1 + s*3]))
                         ])

            fw.writerow([])

            fw.writerow(["host_submission_med", 
                         "tpu_comms_med",
                         "tpu_return_med", 
                         "inference_med",
                         "total_med"
                         ])

            fw.writerow([10**6 * (float(usb_stats.host_data_stats[2 + s*3])), 
                         10**6 * (float(usb_stats.tpu_comms_stats[2 + s*3])), 
                         10**6 * (float(usb_stats.tpu_data_stats[2 + s*3])), 
                         10**6 * (float(usb_stats.infer_stats[2 + s*3])),
                         10**6 * (float(usb_stats.total_stats[2 + s*3]))
                         ])

            fw.writerow([])

            fw.writerow(["host_data_perc",
                         "tpu_comms_perc",
                         "tpu_data_perc", 
                         "inference_perc",
                         "total_perc"
                         ])

            fw.writerow([10**2 * (float(usb_stats.host_data_stats[0 + s*3])/float(usb_stats.total_stats[0 + s*3])), 
                         10**2 * (float(usb_stats.tpu_comms_stats[0 + s*3])/float(usb_stats.total_stats[0 + s*3])), 
                         10**2 * (float(usb_stats.tpu_data_stats[0 + s*3])/float(usb_stats.total_stats[0 + s*3])), 
                         10**2 * (float(usb_stats.infer_stats[0 + s*3])/float(usb_stats.total_stats[0 + s*3])), 
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
    values.create_stats(sessions)
    return values


def read_usb_results(filename, sessions):
    """Reads the usb results that were exported to a csv file locaton in
    the results/usb folder.

    Parameters
    ----------
    filename : string
    Name of the file/model being analyzed.

    sessions : Integer
    Name of the file
    """
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
                tmp_h_data = []
                tmp_t_comms = []
                tmp_t_data = []
                tmp_infer= []
                tmp_total = []
                for j in range(sessions):
                    i += 1

                    if (any(float(x) <= 0 for x in row)): # If there is any negative value.
                        usb_times.append_negatives(
                                row[0 + j*5],               # host submission
                                row[1 + j*5],               # tpu comms
                                row[2 + j*5],               # tpu return
                                row[3 + j*5],               # inference
                                row[4 + j*5],               # total
                                j,                          # Session number
                                i)                          # Test Run
                    else:
                        v += 1
                        tmp_h_data.append(float(row[0 + j*5]))  # host submission
                        tmp_t_comms.append(float(row[1 + j*5])) # tpu comms
                        tmp_t_data.append(float(row[2 + j*5]))  # tpu return
                        tmp_infer.append(float(row[3 + j*5]))   # inference
                        tmp_total.append(float(row[4 + j*5]))   # total

                usb_times.host_data_arr.append(tmp_h_data)
                usb_times.tpu_comms_arr.append(tmp_t_comms)
                usb_times.tpu_data_arr.append(tmp_t_data)
                usb_times.infer_arr.append(tmp_infer)
                usb_times.total_arr.append(tmp_total)

            header = False

    log.info(f"Valid: {v}/{i}")

    if v != i:
        usb_times.print_negatives()
        input("Continue:")

    if v == 0: 
        sys.exit("NO VALID RESULTS FOUND.")

    return usb_times, f"{v}/{i}"


def plot_manager(op, filesize, sessions):
    """Manager function responsible for preping and executing the deployment
    of the compiled tflite models.

    Starts the docker, sets the number of times (count) that they will be
    deployed, copies the necessary folders on the docker, deploys the models, 
    copies the results back.

    Parameters
    ----------
    op : string
    Name of the operation/model being analyzed.

    filesize : string
    Data size of the model being analyzed.

    sessions : Integer
    Number of sessions within the being analyzed model.
    """
    import os
    import logging
    from utils import deduce_filename, parse_csv

    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    log.info(f"Compiling plot results: {op}...")

    results_file = f"results/usb/{op}/Results.csv"
    firsts_file = f"results/usb/{op}/Firsts.csv"
    comms_file = f"results/usb/{op}/Comms.csv"

    values, valid_str = read_usb_results(results_file, sessions)
    first_values, first_valid_str = read_usb_results(firsts_file, sessions)

    values_stats = retreive_usb_stats(values, sessions)
    first_values_stats = retreive_usb_stats(first_values, sessions)
    comms_stats = retreive_array_stats(parse_csv(comms_file))

    usb_results_file = store_usb_stats(
            values_stats, first_values_stats, comms_stats, op, filesize, 
            valid_str, first_valid_str, sessions)

    integrate_results(usb_results_file, op)


if __name__ == '__main__':
    """Entry point to execute this script.

    Flags
    ---------
    -m or --mode
    Mode in which the script should run. Group, Single or Debug.
        Group is supposed to be used to plot the results of a group of models in sequence (lazy).
        Single plots the results of a single model.

    -t or --target
        Should be used in conjunction with the 'Single' mode, where -t must be followed
        by the path to the results of a model in results/usb/<MODEL>/Results.csv.

    -f or --folder
        Should be used in conjunction with the 'Group' mode, where -f must be followed
        by the path to a folder containing all results to be deployed.
    """
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
