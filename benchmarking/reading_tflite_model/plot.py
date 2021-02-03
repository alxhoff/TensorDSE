class UsbAverages():
    def __init__(self, host_comms_avg, host_submission_avg,
                    tpu_comms_avg, tpu_return_avg,
                    inference_avg, total_avg):

        self.host_comms_avg = host_comms_avg
        self.host_submission_avg = host_submission_avg

        self.tpu_comms_avg = tpu_comms_avg
        self.tpu_return_avg = tpu_return_avg

        self.inference_avg = inference_avg
        self.total_avg = total_avg


class UsbTimes():
    def __init__(self, host_comms_time, host_submission_time, 
                    tpu_comms_time, tpu_return_time,
                    inference_time, total_time):

        self.host_comms_time = host_comms_time
        self.host_submission_time = host_submission_time

        self.tpu_comms_time = tpu_comms_time
        self.tpu_return_time = tpu_return_time

        self.inference_time = inference_time
        self.total_time = total_time


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


def read_timestamps(filename):
    import os
    import csv
    assert (os.path.exists(filename)), "File doesnt exist."

    usb_timers_array = []

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        header = True
        for row in csv_reader:
            if (header == True):
                host_comms_time = float(row[1]) - float(row[0])
                host_submission_time = float(row[2]) - float(row[1])

                tpu_comms_time = float(row[4]) - float(row[3])
                tpu_return_time = float(row[5]) - float(row[4])

                inference_time = float(row[4]) - float(row[2])
                total_time = float(row[5]) - float(row[0])

                usb_timer = UsbTimes(host_comms_time, host_submission_time, 
                                        tpu_comms_time, tpu_return_time,
                                        inference_time, total_time)

                if (host_comms_time > 0
                    and host_submission_time > 0
                    and tpu_comms_time > 0 
                    and tpu_return_time > 0 
                    and inference_time > 0):
                    usb_timers_array.append(usb_timer)

            header = False

    return usb_timers_array


def find_avgs(values):
    i = 0

    host_comms_avg = 0
    host_submission_avg = 0
    tpu_comms_avg = 0
    tpu_return_avg = 0
    inference_avg = 0
    total_avg = 0

    for u_t in values:
        host_comms_avg += u_t.host_comms_time
        host_submission_avg += u_t.host_submission_time

        tpu_comms_avg += u_t.tpu_comms_time
        tpu_return_avg += u_t.tpu_return_time

        inference_avg += u_t.inference_time
        total_avg += u_t.total_time

        i += 1

    host_comms_avg = (host_comms_avg / i) * 10**6
    host_submission_avg = (host_submission_avg / i) * 10**6

    tpu_comms_avg = (tpu_comms_avg / i) * 10**6
    tpu_return_avg = (tpu_return_avg / i) * 10**6

    inference_avg = (inference_avg / i) * 10**6
    total_avg = (total_avg / i) * 10**6

    usb_average = UsbAverages(host_comms_avg, host_submission_avg, 
                                tpu_comms_avg, tpu_return_avg,
                                inference_avg, total_avg)

    return usb_average


def store_avgs(filename, avgs, filesize):
    import os
    import csv

    from utils import extend_directory

    csv_dir = extend_directory("results/usb_end/", filename)
    csv_file = f"{csv_dir}/Results.csv"

    with open(csv_file, 'w') as csvfile:
        fw = csv.writer(csvfile, delimiter=',', quotechar='|',
                        quoting=csv.QUOTE_MINIMAL)

        fw.writerow(["host_comms_avg",
                     "host_submission_avg", 
                     "tpu_comms_avg",
                     "tpu_return_avg", 
                     "inference_avg",
                     "total_avg",
                     "data_size"])

        fw.writerow([avgs.host_comms_avg, 
                     avgs.host_submission_avg,
                     avgs.tpu_comms_avg,
                     avgs.tpu_return_avg,
                     avgs.inference_avg,
                     avgs.total_avg,
                     filesize])

    return csv_file


def plot_manager(folder):
    import os
    models_info = deduce_plot_ops(folder, "Results.csv")

    for model_info in models_info:
        model_name = model_info[0]
        filename = model_info[1]
        filesize = deduce_plot_filesize(filename)

        values = read_timestamps(filename)
        avgs = find_avgs(values)
        results_file = store_avgs(model_name, avgs, filesize)


def plot_single_manager(filepath):
    import os
    import logging
    from utils import deduce_filename

    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    log.info("Obtaining averages...")
    filename_fdr = os.path.dirname(
                    os.path.abspath(filepath))

    model_name = deduce_filename(filename_fdr, ending=None)
    filesize = deduce_plot_filesize(model_name)

    values = read_timestamps(filepath)
    avgs = find_avgs(values)
    results_file = store_avgs(model_name, avgs, filesize)


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
        filename_fdr = os.path.dirname(
                        os.path.abspath(filepath))

        model_name = deduce_filename(filename_fdr, ending=None)
        filesize = deduce_plot_filesize(model_name)

        values = read_timestamps(filepath)
        avgs = find_avgs(values)
        results_file = store_avgs(model_name, avgs, filesize)
    else:
        print("Invaild arguments.")
