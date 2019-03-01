import numpy as np
import pandas as pd
import os
import sys
import time
import pickle
import argparse


def parse_args():
    """Parse command line arguments.

    Args:
        None.

    Returns:
        A argparse.Namespace object which contains all parsed argument values.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-file-path', help='test file path')
    parser.add_argument('--label-file-path', help='label file path')
    parser.add_argument('--query-file-path', help='query file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open(args.test_file_path) as f, open(args.label_file_path,
                                              'w') as lf, open(
                                                  args.query_file_path,
                                                  'w') as qf:
        labels = []
        queries = []
        for line in f:
            arr = line.strip().split(',')
            label = arr[0].strip().strip('"') + '\n'
            query = arr[1].strip().strip('"') + '\n'
            labels.append(label)
            queries.append(query)
        lf.writelines(labels)
        qf.writelines(queries)


if __name__ == '__main__':
    sys.exit(main())
