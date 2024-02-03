"""
    Missing  Docstring: TODO
"""

import csv
import json
import time

import subprocess

from utils.logging.logger import log

def run_command_and_echo(*cmd, save_output=False, wait_time=0.5):
    """ Execute an arbitrary command and echo its output."""
    p = subprocess.run(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    time.sleep(wait_time)
    output = p.stdout.decode()
    p.check_returncode()
    if save_output:
        return output


def read_csv_file(csv_file_path: str) -> list:
    """ Read the data in a CSV file and return it as a list"""
    with open(csv_file_path, newline='', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=',')
        data = [list(map(int,rec)) for rec in reader]
        return data


def read_json_file(file_path: str) -> dict:
    """ Read the data in a JSON file and return it as a dict"""

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
            log.error("Failed to read with encoding: %s", encoding)
        except json.JSONDecodeError:
            log.error("JSONDecodeError with encoding: %s", encoding)
        except FileNotFoundError:
            log.error("File not found: %s", file_path)
        except IOError:
            log.error("An unexpected error occurred!")

    return data


def copy_file(source: str, destination: str):
    """
    Copies a file from a source location to a target destination. 
    """
    run_command_and_echo("cp", source, destination)


def move_file(source: str, destination: str):
    """
    Moves a file from a source location to a target destination. 
    """
    run_command_and_echo("mv", source, destination)


def compare_dict_keys(dict_a: dict, dict_b: dict):
    """
    Compares the keys of two dictionaries
    """
    excluded = []
    for ka in list(dict_a.keys()):
        if ka not in list(dict_b.keys()):
            excluded.append(ka)
    if len(excluded) !=0:
        return False, excluded
    else:
        return True, None
