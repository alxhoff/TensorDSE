"""
Module Docstring: TODO
"""

from typing import Dict
import subprocess
import json

import os
from os.path import join

def load_json(file) -> Dict:
    """
    Reads a JSON file from a path
    """
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data

def run(command:str) -> str:
    """
    Runs a command and returns its output
    """
    p = subprocess.run(command.split(),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                check=False)
    output = p.stdout.decode()
    return output

def extend_directory(src_dir, dst_dir):
    """Extends directory by creating or appending the 'dst_dir'
       upon the 'src_dir'.
    """

    if os.path.exists(src_dir):
        dst_dir = join(src_dir, dst_dir)
        if os.path.exists(dst_dir):
            os.system(f"rm -r {dst_dir}")
        os.system(f"mkdir -p {dst_dir}")
    else:
        dst_dir = join(src_dir, dst_dir)
        os.system(f"mkdir -p {dst_dir}")

    if dst_dir[-1] == "/":
        return dst_dir
    return dst_dir + "/"


def remove_directory(directory):
    """
    Removes directory overloaded as dir.
    """

    if os.path.exists(directory):
        if os.listdir(path=directory) != 0:
            os.system(f"rm -r {directory}")
    else:
        raise ValueError
