
"""Train a phone-loop model."""

import logging
import argparse
import glob
import importlib
import ast
import pickle
import numpy as np
import amdtk


# Helper functions.
# ---------------------------------------------------------------------

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


# Model's specific optimizer.
# ---------------------------------------------------------------------
OPTIMIZERS = {
    'hmm': amdtk.StochasticVBOptimizer,
    'svae': amdtk.SVAEStochasticVBOptimizer
}


def main():
    # Argument parser.
    # -----------------------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__)

    # Group of options for the parallelization.
    # -----------------------------------------------------------------
    group = parser.add_argument_group('Parallel')
    group.add_argument('--profile', default='default',
                       help='profile to use for ipyparallel (default)')
    group.add_argument('--njobs', type=int, default=1,
                       help='number of parallel jobs to use (1)')

    # Group of options for the data.
    # -----------------------------------------------------------------
    group = parser.add_argument_group('Data')
    group.add_argument('--context', type=int, default=0,
                       help='number of frames (in one direction) to stack (0)')
    group.add_argument('--file_format', choices=['htk', 'npy'] ,
                       default='htk',
                       help='file format of the features (htk)')
    group.add_argument('--transcription', default=None,
                       help='file of pairs "utterance_key transcription" '
                            '(None)')

    # Group of options for the logger.
    # -----------------------------------------------------------------
    group = parser.add_argument_group('Logging')
    group.add_argument('--log_level', choices=['debug', 'info', 'warning'],
                       default='info',  help='file format of the features '
                       '(info)')

    # Group of options for the model.
    # -----------------------------------------------------------------
    group = parser.add_argument_group('Model')
    group.add_argument('--n_units', type=int , default=1,
                       help='number of acoustic unit in the phone loop (1)')
    group.add_argument('--n_states', type=int, default=1,
                       help='number of state per aoustic unit (1)')
    group.add_argument('--n_comp', type=int, default=1,
                       help='number of Gaussian components per hmm state (1)')
    group.add_argument('--sample_var', type=float, default=1.,
                       help='variance for the initialization of the Gaussian '
                            'components(1)')
    group.add_argument('--encoder', default=None,
                       help='encoder structure front-end (None)')
    group.add_argument('--decoder', default=None,
                       help='decoder structure backend (None)')

    # Group of options for the training.
    # -----------------------------------------------------------------
    group = parser.add_argument_group('Training')
    group.add_argument('--epochs', type=int, default=1,
                       help='number of training epochs (1)')
    group.add_argument('--batch_size', type=int, default=1,
                       help='batch size (1)')
    group.add_argument('--lrate_hmm', type=float, default=1e-1,
                       help='learning rate for the phone loop (1e-1)')
    group.add_argument('--lrate_autoencoder', type=float, default=1e-3,
                       help='learning rate for the autoencoder (1e-3)')

    # Mandatory arguments..
    # -----------------------------------------------------------------
    parser.add_argument('fea_list', help='list of features to train on')
    parser.add_argument('out_model', help='output model')
    parser.add_argument('out_stats', help='output statistics of the data')

    # Parse the command line.
    # -----------------------------------------------------------------
    args = parser.parse_args()

    # Make sure the arguments are consistent.
    # -----------------------------------------------------------------
    if ((args.encoder is not None and args.decoder is None) or
        (args.encoder is None and args.decoder is not None)):
        parser.error('--encoder and --decoder options should be used '
                     'together')

    # Set the logging level.
    # -----------------------------------------------------------------
    logging.getLogger('amdtk').setLevel(LOG_LEVELS[args.log_level])

    # Load the list of features. Only the file names are loaded into
    # memory. The actural features will be loaded on demand by the
    # optimizer.
    # -----------------------------------------------------------------
    fea_list = load_fea_list(args.fea_list)

    # Initialize the basic features loader. At the beginning the
    # features loader is very basic. We'll add more preprocessors
    # when we get the statistics of the database.
    # -----------------------------------------------------------------
    features_loader = amdtk.FeaturesLoader()
    if args.file_format == 'htk':
        features_loader.add_preprocessor(
            amdtk.FeaturesPreprocessorLoadHTK()
        )
    elif args.file_format == 'npy':
        features_loader.add_preprocessor(
            amdtk.FeaturesPreprocessorLoadNumpy()
        )
    if args.context > 0:
        features_loader.add_preprocessor(
            amdtk.FeaturesPreprocessorStackFrames(args.context)
        )

    # If provided, load the transcription.
    # -----------------------------------------------------------------
    if args.transcription is not None:
        logger.info('using transcription given in {fname}'.format(
            fname=args.transcription))
        trans = load_transcription(args.transcription)
        features_loader.add_preprocessor(
            amdtk.FeaturesPreprocessorLoadAlignments(trans)
        )


    # Start the parallel environment..
    # -----------------------------------------------------------------
    with amdtk.parallel(args.profile, args.njobs) as dview:

        # Before anything, we estimate the statistics of the training
        # data. We'll use these statistics to perform mean/variance
        # normalization of the features.
        # -------------------------------------------------------------
        data_stats = amdtk.collect_stats(dview, fea_list, features_loader)
        with open(args.out_stats, 'wb') as fid:
            pickle.dump(data_stats, fid)

        # Now that we have the database statistics, we'll consider
        # only the mean/variance normalized features.
        # -------------------------------------------------------------
        features_loader.add_preprocessor(
            amdtk.FeaturesPreprocessorMeanVarNorm(data_stats)
        )

        # If the 'encoder/decoder' options were specified so we need
        # to set the proper dimension of the features of the HMM.
        # -------------------------------------------------------------
        if args.encoder is not None:
            enc_struct = ast.literal_eval(args.encoder)
            dec_struct = ast.literal_eval(args.decoder)
            dim = enc_struct[-1][2]
        else:
            dim = data_stats['mean'].shape[0]

        # Create a new phone loop model. We set the mean and the
        # variance to zero as we expect the features to be normalized.
        # -------------------------------------------------------------
        model = amdtk.PhoneLoop.create(
            args.n_units,
            args.n_states,
            args.n_comp,
            np.zeros(dim),
            np.ones(dim),
            sample_var=args.sample_var
        )

        # If the 'encoder/decoder' options were specified we'll create
        # a SVAE model embedding the HMM.
        # -------------------------------------------------------------
        if args.encoder is not None:
            model = amdtk.SVAE(enc_struct, dec_struct, model, n_samples=10)
            optimizer_class = amdtk.SVAEStochasticVBOptimizer
        else:
            optimizer_class = amdtk.StochasticVBOptimizer

        # Last element of the features loader chain: makes the output
        # as expected by the underlying model.
        # -------------------------------------------------------------
        features_loader.add_preprocessor(
            amdtk.FeaturesPreprocessorModel(model)
        )

        # Callback to monitor the convergence of the training.
        # -------------------------------------------------------------
        def callback(args):
            logger.info('epoch={epoch} batch={batch}/{n_batch} '
                        'elbo={objective:.4f}'.format(**args))
            if args['batch'] == args['n_batch']:
                model.save('model_' + str(args['epoch']) + '.bin')

        # Training.
        # -------------------------------------------------------------
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
        # -------------------------------------------------------------
        model.save(args.out_model)


if __name__ == '__main__':
    main()
else:
    print('This script cannot be imported')
    exit(1)
