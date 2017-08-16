
"""Set the prior as the current posterior distribution."""

import logging
import argparse
import pickle
import numpy as np
import amdtk
import copy


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

    # Group of options for the logger.
    # -----------------------------------------------------------------
    group = parser.add_argument_group('Logging')
    group.add_argument('--log_level', choices=['debug', 'info', 'warning'],
                       default='info',  help='file format of the features '
                       '(info)')

    # Mandatory arguments..
    # -----------------------------------------------------------------
    parser.add_argument('model', help='the model to use')
    parser.add_argument('out_model', help='output model')

    # Parse the command line.
    # -----------------------------------------------------------------
    args = parser.parse_args()

    # Set the logging level.
    # -----------------------------------------------------------------
    logging.getLogger('amdtk').setLevel(LOG_LEVELS[args.log_level])

    # Load the model.
    # -----------------------------------------------------------------
    model = amdtk.load(args.model)

    # Change the prior.
    # -----------------------------------------------------------------
    #model.latent_prior = \
    #    copy.deepcopy(model.latent_posterior)
    #for i, posterior in enumerate(model.state_posteriors):
    #    model.state_priors[i] = copy.deepcopy(posterior)
    #for comp in model.components:
    #    comp.prior = copy.deepcopy(comp.posterior)
    #model.post_update()
    model.prior_latent.latent_prior = \
        copy.deepcopy(model.prior_latent.latent_posterior)
    for i, posterior in enumerate(model.prior_latent.state_posteriors):
        model.prior_latent.state_priors[i] = copy.deepcopy(posterior)
    for comp in model.prior_latent.components:
        comp.prior = copy.deepcopy(comp.posterior)
    model.prior_latent.post_update()

    # Store the model and the statistics of the data.
    # -------------------------------------------------------------
    model.save(args.out_model)

if __name__ == '__main__':
    main()
else:
    print('This script cannot be imported')
    exit(1)

