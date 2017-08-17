
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
    group.add_argument('--encoder', default=None,
                       help='encoder structure front-end (None)')
    group.add_argument('--decoder', default=None,
                       help='decoder structure backend (None)')

    # Group of options for the model.
    group = parser.add_argument_group('Model')

    # Mandatory arguments.
    parser.add_argument('prior', help='statistics of the training data')
    parser.add_argument('out_model', help='output model')

    # Parse the command line.
    args = parser.parse_args()

    # Set the logging level.
    logging.getLogger('amdtk').setLevel(LOG_LEVELS[args.log_level])

    # Load the prior.
    prior = amdtk.load(args.prior)

    # Create SVAE model.
    enc_struct = ast.literal_eval(args.encoder)
    dec_struct = ast.literal_eval(args.decoder)
    model = amdtk.SVAE(enc_struct, dec_struct, prior, n_samples=10)

    # Store the model and the statistics of the data.
    model.save(args.out_model)


if __name__ == '__main__':
    main()
else:
    print('This script cannot be imported')
    exit(1)
