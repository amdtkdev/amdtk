
"""Compute the phone error rate."""

import logging
import argparse
import glob
import importlib
import ast
import pickle
import numpy as np
import amdtk


# Phone Error Rate computation. Taken from:
# https://martin-thoma.com/word-error-rate-calculation/https://martin-thoma.com/word-error-rate-calculation
# ---------------------------------------------------------------------
def per(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list
    h : list

    Returns
    -------
    int

    Examples
    --------
    >>> per("who is there".split(), "is there".split())
    1
    >>> per("who is there".split(), "".split())
    3
    >>> per("".split(), "who is there".split())
    3
    """
    # initialisation
    import numpy

    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]


# Helper functions.
# ---------------------------------------------------------------------

def load_transcription(file_trans):
    retval = {}
    with open(file_trans, 'r') as fid:
        for line in fid:
            tokens = line.strip().split()
            retval[tokens[0]] = tokens[1:]
    return retval


def load_map(file_map):
    retval = {}
    with open(file_map, 'r') as fid:
        for line in fid:
            tokens = line.strip().split()
            retval[tokens[0]] = tokens[1]
    return retval


def map_seq(seq, phone_map):
    new_seq = []
    for label in seq:
        try:
            new_label = phone_map[label]
            new_seq.append(new_label)
        except KeyError:
            pass
    return new_seq


# Identity phone mapping.
# ---------------------------------------------------------------------
class IdentityPhoneMap(object):

    def __getitem__(self, key):
        return key


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

    # Group of options for the logger.
    # -----------------------------------------------------------------
    group = parser.add_argument_group('Logging')
    group.add_argument('--log_level', choices=['debug', 'info', 'warning'],
                       default='info',  help='file format of the features '
                       '(info)')

    # Group of options for the phone mapping.
    # -----------------------------------------------------------------
    group = parser.add_argument_group('Phone mapping')
    group.add_argument('--ref_phone_map',
                       help='phone map to apply on the reference (none)')
    group.add_argument('--hyp_phone_map',
                       help='phone map to apply on  the hypothesis (none)')

    # Mandatory arguments..
    # -----------------------------------------------------------------
    parser.add_argument('ref', help='reference transcription')
    parser.add_argument('hyp', help='hypothesis transcription')

    # Parse the command line.
    # -----------------------------------------------------------------
    args = parser.parse_args()

    # Set the logging level.
    # -----------------------------------------------------------------
    logging.getLogger('amdtk').setLevel(LOG_LEVELS[args.log_level])

    # Load the transcriptions.
    # -----------------------------------------------------------------
    ref_trans = load_transcription(args.ref)
    hyp_trans = load_transcription(args.hyp)

    # Load the phone mapping if provided.
    # -----------------------------------------------------------------
    if args.ref_phone_map is not None:
        logger.debug('using {phone_map} mapping for the reference'.format(
            phone_map=args.ref_phone_map))
        ref_phone_map = load_map(args.ref_phone_map)
    else:
        logger.debug('no phone map provided for the reference using '
                     'identity mapping')
        ref_phone_map = IdentityPhoneMap()

    if args.hyp_phone_map is not None:
        logger.debug('using {phone_map} mapping for the hypothesis'.format(
            phone_map=args.hyp_phone_map))
        hyp_phone_map = load_map(args.hyp_phone_map)
    else:
        logger.debug('no phone map provided for the hypothesis using '
                     'identity mapping')
        hyp_phone_map = IdentityPhoneMap()

    # Compute the phone error rate.
    # -----------------------------------------------------------------
    tot_per = 0.
    count = 0
    for utt in ref_trans:
        try:
            ref = map_seq(ref_trans[utt], ref_phone_map)
            hyp = map_seq(hyp_trans[utt], hyp_phone_map)
            tot_per += per(ref, hyp)
            count += len(ref)
        except KeyError:
            logger.warning('no hypothesis for utterance {key}'.format(
                key=utt))


    logger.info('phone error rate: {per:.2f} %'.format(
        per=100 * tot_per / count))


if __name__ == '__main__':
    main()
else:
    print('This script cannot be imported')
    exit(1)

