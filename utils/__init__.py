from typing import Dict

def load_json(file) -> Dict:
    import json
    with open(file, "r") as f:
        data = json.load(f)
        return data

def run(command:str) -> str:
    import subprocess
    p = subprocess.run(command.split(),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
    output = p.stdout.decode()
    return output

def extend_directory(src_dir, dst_dir):
    """Extends directory by creating or appending the 'dst_dir'
       upon the 'src_dir'.
    """
    import os
    from os.path import join

    if (os.path.exists(src_dir)):
        dst_dir = join(src_dir, dst_dir)
        if (os.path.exists(dst_dir)):
            os.system(f"rm -r {dst_dir}")
        os.system(f"mkdir -p {dst_dir}")
    else:
        dst_dir = join(src_dir, dst_dir)
        os.system(f"mkdir -p {dst_dir}")

    if dst_dir[-1] == "/":
        return dst_dir
    return dst_dir + "/"


def remove_directory(dir):
    """Removes directory overloaded as dir.
    """
    import os

    if (os.path.exists(dir)):
        if (os.listdir(path=dir) != 0):
            os.system(f"rm -r {dir}")
    else:
        raise ValueError
