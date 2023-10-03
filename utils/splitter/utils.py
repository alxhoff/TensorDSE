import csv
import json
import time
            
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
    from ..logging.logger import log
    # List of encodings to try
    encodings_to_try = ['utf-8', 'utf-16', 'latin-1', 'iso-8859-1', 'cp1252']

    data = None
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                data = json.load(file)
            # If the JSON file is successfully loaded, break out of the loop
            break
        except UnicodeDecodeError:
            log.error(f"Failed to read with encoding: {encoding}")
        except json.JSONDecodeError:
            log.error(f"JSONDecodeError with encoding: {encoding}")
        except FileNotFoundError:
            log.error(f"File not found: {file_path}")
        except Exception as e:
            log.error(f"An error occurred: {e}")

    return data


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