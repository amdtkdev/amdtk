
"""Convert output from LatticeSegmentation."""

import logging
import argparse
import glob
import importlib
import pickle
import os
import numpy as np


# Helper functions.
# ---------------------------------------------------------------------

def load_fea_list(file_list):
    fea_list = []
    with open(file_list, 'r') as fid:
        fea_list = [line.strip() for line in fid]
    return fea_list


def load_map(file_map):
    retval = {}
    with open(file_map, 'r') as fid:
        for line in fid:
            tokens = line.strip().split()
            retval[tokens[0]] = tokens[1]
    return retval


# Logger.
# ---------------------------------------------------------------------

def main():
    # Argument parser.
    # -----------------------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__)

    # Mandatory arguments..
    # -----------------------------------------------------------------
    parser.add_argument('timed_output', help='timed LatticeSegmentation output')

    # Parse the command line.
    # -----------------------------------------------------------------
    args = parser.parse_args()

    timed_trans = {}
    with open(args.timed_output, 'r') as fid:
        for line in fid:
            if '</s>' in line:
                continue
            tokens = line.strip().split()
            key, _ = os.path.splitext(tokens[0])
            try:
                labels = timed_trans[key]
            except KeyError:
                labels = []

            labels.append(tuple(tokens[1:]))
            timed_trans[key] = labels

    for key in timed_trans:
        for label, start, end in timed_trans[key]:
            print(key, start, end, label)


if __name__ == '__main__':
    main()
else:
    print('This script cannot be imported')
    exit(1)

