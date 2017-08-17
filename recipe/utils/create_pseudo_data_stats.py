
"""Create a features loader and estimate the statistics associated."""

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


# Helper functions.

def load_fea_list(file_list):
    fea_list = []
    with open(file_list, 'r') as fid:
        for line in fid:
            fea_list = [line.strip() for line in fid]
    return fea_list


def main():
    # Argument parser.
    parser = argparse.ArgumentParser(description=__doc__)

    # Group of options for the logger.
    group = parser.add_argument_group('Logging')
    group.add_argument('--log_level', choices=['debug', 'info', 'warning'],
                       default='info',  help='file format of the features '
                       '(info)')

    # Mandatory arguments.
    parser.add_argument('fea_dim', help='features dimension')
    parser.add_argument('data_stats', help='output statistics of the data')

    # Parse the command line.
    args = parser.parse_args()

    # Set the logging level.
    logging.getLogger('amdtk').setLevel(LOG_LEVELS[args.log_level])

    # Create an empty features loader.
    features_loader = amdtk.FeaturesLoader()

    # Specify the type of file format to be loaded.
    if args.file_format == 'htk':
        features_loader.add_preprocessor(
            amdtk.FeaturesPreprocessorLoadHTK()
        )
    elif args.file_format == 'npy':
        features_loader.add_preprocessor(
            amdtk.FeaturesPreprocessorLoadNumpy()
        )

    # Stack frame. The total number of frame stacked will be equal
    # to 2 * args.context + 1.
    if args.context > 0:
        features_loader.add_preprocessor(
            amdtk.FeaturesPreprocessorStackFrames(args.context)
        )

    # Load the list of features. Only the file names are loaded into
    # memory. The actural features will be loaded on demand by the
    # optimizer.
    fea_list = load_fea_list(args.fea_list)

    # Start the parallel environment..
    with amdtk.parallel(args.profile, args.njobs) as dview:

        # Estimate the statistics of the data. We'll use these
        # statistics to perform mean/variance normalization of the
        # features.
        data_stats = amdtk.collect_stats(dview, fea_list, features_loader)

        # Store whether the data will be normalized.
        data_stats['mv_norm'] = args.mv_norm

        # Store the statistics. They will be used during the training.
        with open(args.data_stats, 'wb') as fid:
            pickle.dump(data_stats, fid)

    # If requested, add a mean/variance normalization step.
    if args.mv_norm:
        features_loader.add_preprocessor(
            amdtk.FeaturesPreprocessorMeanVarNorm(data_stats)
        )

    # Store the features loader.
    with open(args.fea_loader, 'wb') as fid:
        pickle.dump(features_loader, fid)


if __name__ == '__main__':
    main()
else:
    print('This script cannot be imported')
    exit(1)

