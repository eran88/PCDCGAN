import sys
import torch
import pickle

from funcs import *
from models import *
from pathlib import Path


def main():
    if(len(sys.argv) > 1):
        file_name = sys.argv[1]
    else:
        raise Exception("You must enter a file name.")

    file_name = file_name + ".pkl"
    print(file_name)
    my_file = Path(file_name)

    if not my_file.is_file() or len(sys.argv) <= 1:
        raise Exception("The file doesn't exist.")

    gen = None
    with open(my_file, 'rb') as f:
        gen, _, _ = pickle.load(f)

    return create(gen)


if __name__ == '__main__':
    main()