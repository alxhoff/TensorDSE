class UsbAverages():
    def __init__(self, host_comms_avg, inference_avg, total_avg):
        self.host_comms_avg = host_comms_avg
        # self.tpu_comms_avg = 0
        self.inference_avg = inference_avg
        self.total_avg = total_avg

class UsbTimes():
    def __init__(self, host_comms_time, inference_time, total_time):
        self.host_comms_time = host_comms_time
        # self.tpu_comms_time = 0
        self.inference_time = inference_time
        self.total_time = total_time

def deduce_ops(folder, filename):
    import os
    from os import listdir
    from os.path import isfile, isdir, join

    plot_info = []

    for dirs in listdir(folder):
        res_path = folder + dirs
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
        row_count = 0
        for row in csv_reader:
            if (row_count != 0):
                host_comms_time = float(row[1]) - float(row[0])
                # tpu_comms_time = float(row[1]) - float(row[0])
                inference_time = float(row[4]) - float(row[2])
                total_time = float(row[5]) - float(row[0])

                usb_timer = UsbTimes(host_comms_time, inference_time,
                                        total_time)
                usb_timers_array.append(usb_timer)

            row_count += 1

    return usb_timers_array

def find_avgs(values):
    i = 0
    host_comms_avg = 0
    inference_avg = 0
    total_avg = 0
    for u_t in values:
        host_comms_avg += u_t.host_comms_time
        inference_avg += u_t.inference_time
        total_avg += u_t.total_time

        i+=1

    host_comms_avg = (host_comms_avg / i) * 10**6
    inference_avg = (inference_avg / i) * 10**6
    total_avg = (total_avg / i) * 10**6

    usb_average = UsbAverages(host_comms_avg, inference_avg, total_avg)
    return usb_average

def store_avgs(filename, avgs):
    import os
    import csv

    from utils import extend_directory

    csv_dir = extend_directory("plots/", filename)
    csv_file = f"{csv_dir}/Results.csv"

    with open(csv_file, 'w') as csvfile:
        fw = csv.writer(csvfile, delimiter=',', quotechar='|',
                        quoting=csv.QUOTE_MINIMAL)
    
        fw.writerow([avgs.host_comms_avg, avgs.inference_avg, avgs.total_avg])

    return csv_file

def plot_manager(folder):
    import os
    models_info = deduce_ops(folder, "Results.csv")

    for model_info in models_info:
        model_name = model_info[0]
        filename = model_info[1]

        values = read_timestamps(filename)
        avgs = find_avgs(values)
        results_file = store_avgs(model_name, avgs)


if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('-m', '--mode', required=False,
                        default="Single",
                        help='Mode in which the script will run: All or Single.')

    parser.add_argument('-f', '--folder', required=False,
                        default="",
                        help='Folder.')

    parser.add_argument('-t', '--target', required=False,
                        default="results/shark/mobilnet/Results.",
                        help='Model.')


    args = parser.parse_args()
    if (args.mode == "All" and args.folder != ""):
        plot_manager(args.folder)

    elif (args.mode == "Single"):
        filename = args.target
        model_name = deduce_ops(os.path.dirname(os.path.abspath(filename)),filename)

        values = read_timestamps(filename)
        avgs = find_avgs(values)
        results_file = store_avgs(model_name, avgs)
    else:
        print("Invaild arguments.")
