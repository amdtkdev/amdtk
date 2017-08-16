
"""Average the parameters of a set of phone-loop model."""

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
    parser.add_argument('models', nargs='+', help='the model to use')
    parser.add_argument('out_model', help='output model')

    # Parse the command line.
    # -----------------------------------------------------------------
    args = parser.parse_args()

    # Set the logging level.
    # -----------------------------------------------------------------
    logging.getLogger('amdtk').setLevel(LOG_LEVELS[args.log_level])

    # Load the models.
    # -----------------------------------------------------------------
    models = [amdtk.load(model_path) for model_path in args.models]

    # Accumulate the natural parameters of the models.
    # -----------------------------------------------------------------
    np1 = models[0].latent_posterior.natural_params
    np2 = [posterior.natural_params for posterior in
           models[0].state_posteriors]
    np3 = [comp.posterior.natural_params for comp in
           models[0].components]
    params = [np1, np2, np3]
    for model in models:
        params[0] = model.latent_posterior.natural_params
        for i, posterior in enumerate(model.state_posteriors):
            params[1][i] += posterior.natural_params
        for i, comp in enumerate(model.components):
            params[2][i] += comp.posterior.natural_params

    # Set the averaged natural parmaters to the output mdoel.
    # -----------------------------------------------------------------
    out_model = models[0]
    weight = 1. / len(models)
    model.latent_posterior.natural_params = weight * params[0]
    for i, posterior in enumerate(model.state_posteriors):
        posterior.natural_params = weight * params[1][i]
    for i, comp in enumerate(model.components):
        comp.posterior.natural_params = weight * params[2][i]

    # Store the model and the statistics of the data.
    # -------------------------------------------------------------
    model.save(args.out_model)

if __name__ == '__main__':
    main()
else:
    print('This script cannot be imported')
    exit(1)

