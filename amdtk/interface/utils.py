
"""
Helper functions..

The main purpose of this module is to provide a way to manage the
ipyparallel cluster within a python session..

Copyright (C) 2017, Lucas Ondel

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

"""

import logging
from ipyparallel.util import interactive
import numpy as np


# Create the module's logger.
logger = logging.getLogger(__name__)


@interactive
def _collect_stats_job(filename):
    # We re-import AMDTK because this code will run "remotely".
    import amdtk

    # features_loader is passed to the context by the caller.
    data = fea_loader.load(filename)['data']

    return (data.shape[0], data.sum(axis=0), (data ** 2).sum(axis=0))


def collect_stats(dview, fea_list, fea_loader):
    """Collect the mean and the variance of the data set.

    Parameters
    ----------
    dview : object
        Ipyparallel direct view interface.
    fea_list : list
        List of features file.
    fea_loader : :class:`FeaturesLoader`
        Features loader.

    Returns
    -------
    data_stats : dict
        Dictionary containing 3 keys:
          * 'mean'
          * 'variance'
          * counts

    """
    dview.push({'fea_loader':fea_loader})
    acc_stats = dview.map_sync(_collect_stats_job, fea_list)

    stats = np.asarray(acc_stats, dtype=object).sum(axis=0)
    count = stats[0]
    mean = stats[1] / count
    var = (stats[2] / count)  - mean**2

    return {'mean':mean, 'var':var, 'count':count}


