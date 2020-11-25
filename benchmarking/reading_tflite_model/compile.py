
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-g', '--group', required=True, help='File path to the .tflite file.')
    parser.add_argument('-m', '--model', required=True, help='File path to the .tflite file.')
    parser.add_argument('-d', '--delegate', required=True, help='cpu, gpu or edge_tpu.')
    parser.add_argument('-n', '--name', required=True, help='Name of Model/Operation, needed to create corresponding folder name.')
    parser.add_argument('-c', '--count', type=int, default=5,help='Number of times to run inference.')

    args = parser.parse_args()

    if ("cpu" in args.delegate):
        cpu_tflite_deployment(args.model, args.name, args.count)
    elif ("gpu" in args.delegate):
        gpu_tflite_deployment(args.model, args.name, args.count)
    elif ("edge_tpu" in args.delegate):
        edge_tflite_deployment(args.model, args.name, args.count)
    else:
        print("INVALID delegate input.")
