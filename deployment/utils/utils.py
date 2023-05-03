import sys
import csv
import json
import time
import logging
import argparse
 
def LoggerInit(filename='optimizer.log'):
    logging.basicConfig(filename=filename,
                        level=logging.DEBUG,
                        format='%(asctime)s  -  %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def ParseArgs():
    # Initialize parser
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument("-model", "--Model", help = "Path to Source Model file to optimize", required=True)
    parser.add_argument("-map", "--Mapping", help = "Path to CSV file containing mapping", required=True)
    # Read arguments from command line
    try:
        args = parser.parse_args()
        return args
    except:
        print('Wrong or Missing argument!')
        print('Example Usage: compile_and_map.py -model <path/to/model/file> -map <path/to/csv/file/containing/maping>')
        sys.exit(1)

def RunTerminalCommand(*cmd, save_output=False, wait_time=0.5):
    """ Execute an arbitrary command and echo its output."""
    import subprocess
    p = subprocess.run(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    time.sleep(wait_time)
    output = p.stdout.decode()
    p.check_returncode()
    if save_output:
        return output
    
def ReadCSV(csv_file_path: str):
    with open(csv_file_path, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        data = [list(map(int,rec)) for rec in reader]
        return data

def ReadJSON(file_path: str):
    with open(file_path) as fin:
        return json.load(fin)

def CopyFile(source: str, destination: str):
    RunTerminalCommand("cp", source, destination)

def MoveFile(source: str, destination: str):
    RunTerminalCommand("mv", source, destination)

def CompareKeys(dictA: dict, dictB: dict):
    excluded = []
    for ka in list(dictA.keys()):
        if (ka not in list(dictB.keys())):
            excluded.append(ka)
    if len(excluded) !=0:
        return False, excluded
    else:
        return True, None
