
"""Create a pseudo statistics to initialize HMM/GMM operating in latent space."""

import logging
import argparse
import glob
import importlib
import ast
import pickle
import numpy as np
import amdtk


# Logger.
logger = logging.getLogger('amdtk')


# Possible log-level.
LOG_LEVELS = {
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}


def main():
    # Argument parser.
    parser = argparse.ArgumentParser(description=__doc__)

    # Group of options for the logger.
    group = parser.add_argument_group('Logging')
    group.add_argument('--log_level', choices=['debug', 'info', 'warning'],
                       default='info',  help='file format of the features '
                       '(info)')

    # Mandatory arguments..
    parser.add_argument('fea_dim', type=int,
                        help='dimensionality of the features')
    parser.add_argument('mean', type=float,
                        help='mean of the output statistics')
    parser.add_argument('var', type=float,
                        help='variance of the output statistics')
    parser.add_argument('stats', help='output statistics')

    # Parse the command line.
    args = parser.parse_args()

    # Set the logging level.
    logging.getLogger('amdtk').setLevel(LOG_LEVELS[args.log_level])

    # Create the pseudo statistics.
    data_stats = {
        'mean': args.mean * np.ones(args.fea_dim),
        'var': args.var * np.ones(args.fea_dim),
        'count': 1,
        'mv_norm': false
    }

    # Store the statistics.
    with open(args.data_stats, 'wb') as fid:
        pickle.dump(data_stats, fid)


if __name__ == '__main__':
    main()
else:
    print('This script cannot be imported')
    exit(1)

