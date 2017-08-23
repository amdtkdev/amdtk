
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

    # Group of options for the parallelization.
    group = parser.add_argument_group('Parallel')
    group.add_argument('--profile', default='default',
                       help='profile to use for ipyparallel (default)')
    group.add_argument('--njobs', type=int, default=1,
                       help='number of parallel jobs to use (1)')

    # Group of options for the features loader.
    group = parser.add_argument_group('Features Extractor')
    group.add_argument('--file_format', choices=['htk', 'npy'] ,
                       default='htk',
                       help='file format of the features (htk)')
    group.add_argument('--mv_norm', default=None, action='store_true',
                       help='add mean/variance normalization (None)')
    group.add_argument('--context', type=int, default=0,
                       help='number of frames (in each time direction) to '
                            'stack (0)')

    # Mandatory arguments..
    parser.add_argument('fea_list', help='list of features of the database')
    parser.add_argument('data_stats', help='output statistics of the data')
    parser.add_argument('fea_loader', help='output features loader')

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

