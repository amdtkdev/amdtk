"""
Proxy to load the features.

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

import abc
import logging
import os
import numpy as np
from .utils import read_htk


# Create the module's logger.
logger = logging.getLogger(__name__)


class FeaturesLoader(object):
    """Features loading.

    Load the features and may perform various operation at run time.
    The exact behavior of the features will depends on the list of the
    preprocessors given.

    """

    def __init__(self):
        self._preprocessors = []

    def add_preprocessor(self, new_preprocessor):
        """Adde a new preprocessor to the fetures loader.

        Parameters
        ----------
        new_preprocessor : :class:`FeaturesPreprocessor`

        """
        self._preprocessors.append(new_preprocessor)


    def load(self, fname):
        """Load a features file.

        Parameters
        ----------
        fname : string
            Path to the file to load.

        Returns
        -------
        fea_data : dictionary
            Dictionary containing the various value relative to the
            features. The actual pairs key, value of the dictionary
            will depends on the preprocessors given to the features
            loader.

        """
        fea_data = dict()
        for preprocessor in self._preprocessors:
            logger.debug('{name}'.format(name=preprocessor.name))
            fea_data = preprocessor.process(fname, fea_data)
        return fea_data

class FeaturesPreprocessor(metaclass=abc.ABCMeta):

    @abc.abstractproperty
    def name(self):
        """Readable name for logging."""
        pass

    def process(self, fname, fea_data):
        """Process the features.

        Parameters
        ----------
        fname : string
            Path to the file to load.

        fea_data : dictionary
            Dictionary containing the various value relative to the
            features. The actual pairs key, value of the dictionary
            will depends on the preprocessors called before.

        Returns
        -------
        fea_data : dictionary
            The previous `fea_data` with the new data extracted from
            current preprocessor.

        """
        pass

class FeaturesPreprocessorLoadHTK(FeaturesPreprocessor):
    """Load features stored in HTK binary format..

    The data loaded will be added to the key 'data'.

    """
    @property
    def name(self):
        return "load htk features"

    def process(self, fname, fea_data):
        fea_data['data'] = read_htk(fname)
        return fea_data


class FeaturesPreprocessorLoadNumpy(FeaturesPreprocessor):
    """Load features stored in numpy format.

    The data loaded will be added to the key 'data'.

    """
    @property
    def name(self):
        return "load numpy features"

    def process(self, fname, fea_data):
        fea_data['data'] = np.load(fname)
        return fea_data


class FeaturesPreprocessorMeanVarNorm(FeaturesPreprocessor):
    """Apply mean-variance normalization.

    The apply a mean variance normalization to the value mapped to
    the key 'data'.

    """

    def __init__(self, data_stats):
        self._data_stats = data_stats

    @property
    def name(self):
        return 'mean/variance normalization'

    def process(self, fname, fea_data):
        data = fea_data['data']
        data -= self._data_stats['mean']
        data /= np.sqrt(self._data_stats['var'])
        fea_data['data'] = data
        return fea_data


class FeaturesPreprocessorLoadAlignments(FeaturesPreprocessor):
    """Load the alignments associated to the features file..

    The alignments will be associated to the key 'ali' in `fea_data`.

    """
    def __init__(self, alignments):
        self._alignments = alignments

    @property
    def name(self):
        return 'load alignments'

    def process(self, fname, fea_data):
        FeaturesPreprocessor.process(self, fname, fea_data)

        # Extract the name of the features file (without extension)
        bname = os.path.basename(fname)
        key, ext = os.path.splitext(bname)

        # Look for the alignments corresponding to the file name.
        try:
            fea_data['ali'] = self._alignments[key]
        except KeyError:
            logger.warning('missing key {key} in the alignments'.format(
                key=key))

        return fea_data

