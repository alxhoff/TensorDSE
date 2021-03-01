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

    def append_meds(self, host_comms_med, host_submission_med,
                    tpu_comms_med, tpu_return_med,
                    inference_med, total_med):

        self.host_comms_med = host_comms_med
        self.host_submission_med = host_submission_med

        self.tpu_comms_med = tpu_comms_med
        self.tpu_return_med = tpu_return_med

        self.inference_med = inference_med
        self.total_med = total_med


class UsbTimes():
    def __init__(self):

        self.host_comms_array = []
        self.host_submission_array = []

        self.tpu_comms_array = []
        self.tpu_return_array = []

        self.inference_array = []
        self.total_array = []

        self.neg_values = []

    def extend_array(self, array, value, cnt):
        tmp = array[len(array) - 1]
        eval_str = ""
        for i in range(cnt):
            if cnt > 1:
                eval_str = f"{eval_str}tmp[{i}], "
            else:
                eval_str = f"tmp, "

        eval_str = f"array[len(array) - 1] = [{eval_str} value]"
        exec(eval_str)


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
                self.extend_array(
                        self.host_comms_array, host_comms_time, sessions_nr)

                self.extend_array(
                        self.host_submission_array, host_submission_time, sessions_nr)

                self.extend_array(
                        self.tpu_comms_array, tpu_comms_time, sessions_nr)

                self.extend_array(
                        self.tpu_return_array, tpu_return_time, sessions_nr)

                self.extend_array(
                        self.inference_array, inference_time, sessions_nr)

                self.extend_array(
                        self.total_array, total_time, sessions_nr)

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
                arr[0] if float(arr[0]) <= 0 else "    ",
                arr[1] if float(arr[1]) <= 0 else "    ",
                arr[2] if float(arr[2]) <= 0 else "    ",
                arr[3] if float(arr[3]) <= 0 else "    ",
                arr[4] if float(arr[4]) <= 0 else "    ",
                arr[5] if float(arr[5]) <= 0 else "    ",
                arr[6]
                ])
            pass

        print(f"NEGATIVE VALUES\n{tabulate(table)}")


def find_python_stats(cpu_r, edge_r):
    import statistics

    if cpu_r != None:
        cpu_r = [statistics.mean(cpu_r), statistics.stdev(cpu_r), statistics.median(cpu_r)]
        edge_r = [statistics.mean(edge_r), statistics.stdev(edge_r), statistics.median(edge_r)]
        return cpu_r, edge_r
    else:
        edge_r = [statistics.mean(edge_r), statistics.stdev(edge_r), statistics.median(edge_r)]
        return edge_r


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

    # Basically cut the ending of the string that is preceded by a '_'
    # Which means cut the appended filesize to the op name.
    op = (op.rstrip(op.split("_")[op.count("_")])).strip("_")

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

            fw.writerow(["host_comms_perc",
                         "host_submission_perc", 
                         "tpu_comms_perc",
                         "tpu_return_perc", 
                         "inference_perc",
                         "total_perc"
                         ])

            fw.writerow([10**2 * (float(stats[i].host_comms_avg)/float(stats[i].total_avg)), 
                         10**2 * (float(stats[i].host_submission_avg)/float(stats[i].total_avg)),
                         10**2 * (float(stats[i].tpu_comms_avg)/float(stats[i].total_avg)),
                         10**2 * (float(stats[i].tpu_return_avg)/float(stats[i].total_avg)),
                         10**2 * (float(stats[i].inference_avg)/float(stats[i].total_avg)),
                         10**2 * (1)
                         ])

            fw.writerow([])

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
            fw.writerow(["host_comms_med",
                         "host_submission_med", 
                         "tpu_comms_med",
                         "tpu_return_med", 
                         "inference_med",
                         "total_med"
                         ])

            fw.writerow([10**6 * float(stats[i].host_comms_med), 
                         10**6 * float(stats[i].host_submission_med),
                         10**6 * float(stats[i].tpu_comms_med),
                         10**6 * float(stats[i].tpu_return_med),
                         10**6 * float(stats[i].inference_med),
                         10**6 * float(stats[i].total_med)
                         ])
            fw.writerow([])



    return csv_file

def find_stats(values, sessions):
    import copy
    from statistics import mean, stdev, median

    usb_stats_arr = []
    for i in range(sessions):
        usb_stats = UsbStats()

        host_comms_avg = mean(
                fetch_column_values(values.host_comms_array, i, sessions))
        host_comms_std = stdev(
                fetch_column_values(values.host_comms_array, i, sessions))
        host_comms_med = median(
                fetch_column_values(values.host_comms_array, i, sessions))

        host_submission_avg = mean(
                fetch_column_values(values.host_submission_array, i, sessions))
        host_submission_std = stdev(
                fetch_column_values(values.host_submission_array, i, sessions))
        host_submission_med = median(
                fetch_column_values(values.host_submission_array, i, sessions))

        tpu_comms_avg = mean(
                fetch_column_values(values.tpu_comms_array, i, sessions))
        tpu_comms_std = stdev(
                fetch_column_values(values.tpu_comms_array, i, sessions))
        tpu_comms_med = median(
                fetch_column_values(values.tpu_comms_array, i, sessions))

        tpu_return_avg = mean(
                fetch_column_values(values.tpu_return_array, i, sessions))
        tpu_return_std = stdev(
                fetch_column_values(values.tpu_return_array, i, sessions))
        tpu_return_med = median(
                fetch_column_values(values.tpu_return_array, i, sessions))

        inference_avg = mean(
                fetch_column_values(values.inference_array, i, sessions))
        inference_std = stdev(
                fetch_column_values(values.inference_array, i, sessions))
        inference_med = median(
                fetch_column_values(values.inference_array, i, sessions))

        total_avg = mean(
                fetch_column_values(values.total_array, i, sessions))
        total_std = stdev(
                fetch_column_values(values.total_array, i, sessions))
        total_med = median(
                fetch_column_values(values.total_array, i, sessions))

        usb_stats.append_avgs(host_comms_avg, host_submission_avg, 
                                tpu_comms_avg, tpu_return_avg,
                                inference_avg, total_avg)

        usb_stats.append_stds(host_comms_std, host_submission_std, 
                                tpu_comms_std, tpu_return_std,
                                inference_std, total_std)

        usb_stats.append_meds(host_comms_med, host_submission_med, 
                                tpu_comms_med, tpu_return_med,
                                inference_med, total_med)

        tmp = copy.deepcopy(usb_stats)
        usb_stats_arr.append(tmp)

    return usb_stats_arr

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

                    inference_time = float(float(row[4 + expr]) - float(row[2 + expr]))
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

    log.info(f"Valid: {v_cnt}/{(cnt)*sessions}")
    return usb_times, f"{v_cnt}/{(cnt)*sessions}"


def fetch_column_values(tuples, i, sessions):
    arr = []
    for val in tuples:
        if sessions > 1:
            arr.append(val[i])
        else:
            arr.append(val)

    return arr


def plot_manager(op, filesize, sessions):
    import os
    import logging
    from utils import deduce_filename

    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    log.info(f"Compiling plot results: {op}...")

    filepath=f"results/usb/{op}/Results.csv"

    values, valid_str = read_timestamps(filepath, sessions)
    stats = find_stats(values, sessions)
    usb_results_file = store_stats(op, stats, filesize, valid_str, sessions)
    integrate_results(usb_results_file, op)


if __name__ == '__main__':
    import os
    from shark import deduce_sessions_nr
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

    if (args.mode == "Group" and args.folder != ""):
        for direc in os.listdir(args.folder):
            op = direc
            sessions = deduce_sessions_nr(op)
            filesize = op.split("_")[op.count("_")]
            plot_manager(op, filesize, sessions)

    elif (args.mode == "Single" and args.target != ""):
        op = os.path.dirname(args.target)
        filesize = op.split("_")[op.count("_")]
        sessions = deduce_sessions_nr(op)
        plot_manager(op, filesize, sessions)

    else:
        print("Invaild arguments.")
