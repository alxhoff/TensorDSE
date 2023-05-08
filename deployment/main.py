import sys
import argparse
import model_lab.split as split
from model_lab.logger import log
from source_generator.generate_source import GenerateSource

def ParseArgs():
    # Initialize parser
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument("-mode", "--Mode", help = "Mode of Operation (benchmarking or deployment)", required=True)
    parser.add_argument("-model", "--Model", help = "Path to Source Model file to optimize", required=True)
    parser.add_argument("-map", "--Mapping", help = "Path to CSV file containing mapping", required=True)
    # Read arguments from command line
    try:
        args = parser.parse_args()
        return args
    except:
        print('Wrong or Missing argument!')
        print('Example Usage: main.py -mode <mode of operation> -model <path/to/model/file> -map <path/to/csv/file/containing/maping>')
        sys.exit(1)

def main():
    args = ParseArgs()
    splitter = split.Splitter(args.Mode, args.Model, args.Mapping)
    try:
        log.info("Running Model Splitter ...")
        splitter.Run()
        log.info("Splitting Process Complete!\n")
        log.info("Generating Source File ...")
        #GenerateSource()
        log.info("Source File Generation Complete!\n")
    except Exception as e:
        splitter.Clean(True)
        log.error("Failed to run splitter! {}".format(str(e)))
    finally:
        del splitter

if __name__ == '__main__':
    main()
    pass