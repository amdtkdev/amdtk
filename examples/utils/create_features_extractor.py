
"""Create a features extractor object."""

import logging
import argparse
import glob
import importlib
import ast
import pickle
import numpy as np
import amdtk



# Logger.
# ---------------------------------------------------------------------
logger = logging.getLogger('amdtk')


# Possible log-level.
# ---------------------------------------------------------------------
LOG_LEVELS = {
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}


def main():
    # Argument parser.
    # -----------------------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__)

    # Group of options for the logger.
    # -----------------------------------------------------------------
    group = parser.add_argument_group('Logging')
    group.add_argument('--log_level', choices=['debug', 'info', 'warning'],
                       default='info',  help='file format of the features '
                       '(info)')

    # Group of options for the features extractor.
    # -----------------------------------------------------------------
    group = parser.add_argument_group('Features Extractor')
    group.add_argument('--file_format', choices=['htk', 'npy'] ,
                       default='htk',
                       help='file format of the features (htk)')
    group.add_argument('--data_stats', default=None,
                       help='data statistics. If specified, add mean/variance'
                            ' normalization')
    group.add_argument('--context', type=int, default=0,
                       help='number of frames (in each time direction) to '
                            'stack (0)')

    # Mandatory arguments..
    # -----------------------------------------------------------------
    parser.add_argument('out_fextractor', help='output statistics of the data')

    # Parse the command line.
    # -----------------------------------------------------------------
    args = parser.parse_args()

    # Set the logging level.
    # -----------------------------------------------------------------
    logging.getLogger('amdtk').setLevel(LOG_LEVELS[args.log_level])

    # Create an empty features loader.
    # -----------------------------------------------------------------
    features_loader = amdtk.FeaturesLoader()

    # Sepcify the type of file format to be loaded.
    # -----------------------------------------------------------------
    if args.file_format == 'htk':
        features_loader.add_preprocessor(
            amdtk.FeaturesPreprocessorLoadHTK()
        )
    elif args.file_format == 'npy':
        features_loader.add_preprocessor(
            amdtk.FeaturesPreprocessorLoadNumpy()
        )

    # Stack frame. The total number of frame stacked will be equal
    # to 2 * args.contxt + 1.
    # -----------------------------------------------------------------
    if args.context > 0:
        features_loader.add_preprocessor(
            amdtk.FeaturesPreprocessorStackFrames(args.context)
        )

    # If the data statistics is specified add a mean/variance
    # normalization step.
    # -----------------------------------------------------------------
    if args.data_stats is not None:
        with open(data_stats, 'rb') as fid:
            data_stats = pickle.load(data_stats)
        features_loader.add_preprocessor(
            amdtk.FeaturesPreprocessorMeanVarNorm(data_stats)
        )

    # Store the features extractor.
    # -----------------------------------------------------------------
    model.save(args.out_model)


if __name__ == '__main__':
    main()
else:
    print('This script cannot be imported')
    exit(1)
