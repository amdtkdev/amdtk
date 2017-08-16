
"""Use a phone loop model to generate acoustic units posteriors."""

import logging
import argparse
import glob
import importlib
import pickle
import numpy as np
import amdtk


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

    # Group of options for the logger.
    # -----------------------------------------------------------------
    group = parser.add_argument_group('Logging')
    group.add_argument('--log_level', choices=['debug', 'info', 'warning'],
                       default='info',  help='file format of the features '
                       '(info)')

    # Mandatory arguments..
    # -----------------------------------------------------------------
    parser.add_argument('model', help='the model to use')
    parser.add_argument('stats', help='statistics of the training data')
    parser.add_argument('fea_list', help='list of utterance to process')
    parser.add_argument('outdir',
                        help='archive where to store the posteriors')

    # Parse the command line.
    # -----------------------------------------------------------------
    args = parser.parse_args()

    # Set the logging level.
    # -----------------------------------------------------------------
    logging.getLogger('amdtk').setLevel(LOG_LEVELS[args.log_level])

    # Load the model.
    # -----------------------------------------------------------------
    model = amdtk.load(args.model)

    # Load the statistics of the training data.
    # -----------------------------------------------------------------
    with open(args.stats, 'rb') as fid:
        data_stats = pickle.load(fid)

    # Load the list of features. Only the file names are loaded into
    # memory. The actual features will be loaded on demand by the
    # optimizer.
    # -----------------------------------------------------------------
    fea_list = load_fea_list(args.fea_list)
    logger.debug('# utterances: {n_utts}'.format(n_utts=len(fea_list)))

    # Create the features loader.
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
    features_loader.add_preprocessor(
        amdtk.FeaturesPreprocessorMeanVarNorm(data_stats)
    )
    features_loader.add_preprocessor(
        amdtk.FeaturesPreprocessorModel(model)
    )

    # Job that decode the data and write the transcription on disk.
    # -----------------------------------------------------------------
    def posteriors_job(fname):
        import amdtk
        import os
        import numpy as np
        from scipy.misc import logsumexp

        # Extract the key of the utterance.
        # -------------------------------------------------------------
        bname = os.path.basename(fname)
        key, ext = os.path.splitext(bname)

        # Decode the data.
        # -------------------------------------------------------------
        fea_data = fea_loader.load(fname)
        state_llh, _  = model._get_state_llh(fea_data['data'])
        log_norm = logsumexp(state_llh, axis=1)
        state_llh = (state_llh.T - log_norm).T

        # "HTK trick" we transform the posteriors so that when loading
        # it with HVite, it will yield the same results. For more
        # details asks Lukas Burget <burget@fit.vutbr.cz>.
        # -------------------------------------------------------------
        posts = np.exp(state_llh)
        htk_data = np.sqrt(-2 * np.log(posts.clip(min=1e-10)))

        # Write the decoded posteriors on disk.
        # -------------------------------------------------------------
        path = os.path.join(outdir, key + '.posts')
        amdtk.write_htk(path, htk_data)

    # Start the parallel environment.
    # -----------------------------------------------------------------
    with amdtk.parallel(args.profile, args.njobs) as dview:
        # Global variable for the remote jobs.
        # -------------------------------------------------------------
        dview.push({
            'outdir': args.outdir,
            'fea_loader': features_loader,
            'model': model
        })

        # Decode and store the sequence of units.
        # -------------------------------------------------------------
        dview.map_sync(posteriors_job, fea_list)


if __name__ == '__main__':
    main()
else:
    print('This script cannot be imported')
    exit(1)

