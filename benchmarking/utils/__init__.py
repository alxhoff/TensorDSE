def load_json(file_path):
    import json

    with open(file_path) as f:
        model = json.load(f)
        return model


def load_csv(filename):
    import csv

    samples = []

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            samples.append(float(row[1]))

    return samples

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


