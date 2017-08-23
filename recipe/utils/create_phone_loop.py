
"""Train a phone-loop model."""

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

    # Group of options for the model.
    group = parser.add_argument_group('Model')
    group.add_argument('--n_units', type=int , default=1,
                       help='number of acoustic unit in the phone loop (1)')
    group.add_argument('--n_states', type=int, default=1,
                       help='number of state per aoustic unit (1)')
    group.add_argument('--n_comp', type=int, default=1,
                       help='number of Gaussian components per hmm state (1)')
    group.add_argument('--sample_var', type=float, default=None,
                       help='variance for the initialization of the Gaussian '
                            'components')

    # Mandatory arguments.
    parser.add_argument('data_stats', help='statistics of the training data')
    parser.add_argument('out_model', help='output model')

    # Parse the command line.
    args = parser.parse_args()

    # Set the logging level.
    logging.getLogger('amdtk').setLevel(LOG_LEVELS[args.log_level])

    # Load the statistics of the training data.
    with open(args.data_stats, 'rb') as fid:
        data_stats = pickle.load(fid)

    # Features dimensionality.
    fea_dim = len(data_stats['mean'])

    # Mean and variance of the prior.
    if data_stats['mv_norm']:
        mean = np.zeros(fea_dim)
        var = np.ones(fea_dim)
    else:
        mean = data_stats['mean']
        var = data_stats['var']

    # Create a new phone loop model.
    model = amdtk.PhoneLoop.create(
        args.n_units,
        args.n_states,
        args.n_comp,
        mean,
        var,
        sample_var=args.sample_var
    )

    # Store the model and the statistics of the data.
    model.save(args.out_model)


if __name__ == '__main__':
    main()
else:
    print('This script cannot be imported')
    exit(1)
