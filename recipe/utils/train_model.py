
"""Train a phone-loop model."""

import logging
import argparse
import os
import glob
import importlib
import ast
import pickle
import numpy as np
import amdtk


# Helper functions.

def load_transcription(transcript_file):
    trans = {}
    with open(transcript_file, 'r') as fid:
        for line in fid:
            tokens = line.strip().split()
            trans[tokens[0]] = [int(token) for token in tokens[1:]]
    return trans


def load_fea_list(file_list):
    fea_list = []
    with open(file_list, 'r') as fid:
        for line in fid:
            fea_list = [line.strip() for line in fid]
    return fea_list


# Logger.
logger = logging.getLogger('amdtk')


# Possible log-level.
LOG_LEVELS = {
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}


# Model's specific optimizer.
OPTIMIZERS = {
    'hmm': amdtk.StochasticVBOptimizer,
    'svae': amdtk.SVAEStochasticVBOptimizer
}


def main():
    # Argument parser.
    parser = argparse.ArgumentParser(description=__doc__)

    # Group of options for the parallelization.
    group = parser.add_argument_group('Parallel')
    group.add_argument('--profile', default='default',
                       help='profile to use for ipyparallel (default)')
    group.add_argument('--njobs', type=int, default=1,
                       help='number of parallel jobs to use (1)')

    # Group of options for the logger.
    group = parser.add_argument_group('Logging')
    group.add_argument('--log_level', choices=['debug', 'info', 'warning'],
                       default='info',  help='file format of the features '
                       '(info)')

    # Group of options for the training.
    group = parser.add_argument_group('Training')
    group.add_argument('--epochs', type=int, default=1,
                       help='number of training epochs (1)')
    group.add_argument('--batch_size', type=int, default=1,
                       help='batch size (1)')
    group.add_argument('--lrate_hmm', type=float, default=1e-1,
                       help='learning rate for the phone loop (1e-1)')
    group.add_argument('--lrate_autoencoder', type=float, default=1e-3,
                       help='learning rate for the autoencoder (1e-3)')
    group.add_argument('--transcription', help='transcription of the training')

    # Mandatory arguments..
    parser.add_argument('tmpdir', help='where to store temporary models')
    parser.add_argument('fea_list', help='list of features to train on')
    parser.add_argument('fea_loader', help='features loader')
    parser.add_argument('data_stats', help='training data statistics')
    parser.add_argument('init_model', help='initial model')
    parser.add_argument('out_model', help='output model')

    # Parse the command line.
    args = parser.parse_args()

    # Set the logging level.
    logging.getLogger('amdtk').setLevel(LOG_LEVELS[args.log_level])

    # Load the list of features. Only the file names are loaded into
    # memory. The actural features will be loaded on demand by the
    # optimizer.
    fea_list = load_fea_list(args.fea_list)

    # Load the features loader.
    with open(args.fea_loader, 'rb') as fid:
        features_loader = pickle.load(fid)

    # Load the data statistics.
    with open(args.data_stats, 'rb') as fid:
        data_stats = pickle.load(fid)

    # load the model to train.
    model = amdtk.load(args.init_model)

    # Connect the features loader and the model.
    features_loader.add_preprocessor(
        amdtk.FeaturesPreprocessorModel(model)
    )

    # If provided, load the transcription.
    if args.transcription is not None:
        logger.info('using transcription given in {fname}'.format(
            fname=args.transcription))
        trans = load_transcription(args.transcription)
        features_loader.add_preprocessor(
            amdtk.FeaturesPreprocessorLoadAlignments(trans)
        )

    # Start the parallel environment..
    with amdtk.parallel(args.profile, args.njobs) as dview:
        # Choose the optimizer according to the model.
        if model.__class__ == amdtk.SVAE :
            optimizer_class = amdtk.SVAEStochasticVBOptimizer
        else:
            optimizer_class = amdtk.StochasticVBOptimizer

        # Callback to monitor the convergence of the training.
        tmpdir = args.tmpdir
        def callback(args):
            logger.info('epoch={epoch} batch={batch}/{n_batch} '
                        'elbo={objective:.4f}'.format(**args))
            if args['batch'] == args['n_batch']:
                path = os.path.join(tmpdir,
                                    'model_' + str(args['epoch']) + '.bin')
                model.save(path)

        # Training.
        train_args = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lrate': args.lrate_hmm,
            'lrate_ae': args.lrate_autoencoder
        }
        optimizer = optimizer_class(
            dview,
            data_stats,
            features_loader,
            train_args,
            model
        )
        optimizer.run(fea_list, callback)

        # Store the model and the statistics of the data.
        model.save(args.out_model)


if __name__ == '__main__':
    main()
else:
    print('This script cannot be imported')
    exit(1)
