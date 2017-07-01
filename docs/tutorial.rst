********
Tutorial
********

This tutorial explains step by step how to build a model to do phoneme
recognition and/or discovery.

.. note:: All along this tutorial we assume that you have imported
    AMDTK in your python session as in the code below:
    ::

        >>> import amdtk



Basic operations
================


Settings AMDTK's verbosity
--------------------------

AMDTK uses the `logging <https://docs.python.org/3.6/library/logging.html>`_
python module to output messages. By default, the verbosity level is
set to the highest level `logging.DEBUG`. This behavior can be changed
as in the following example:

::

    >>> import logging
    >>> logger = logging.getLogger('amdtk')
    >>> logger.setLevel(logging.INFO)


Parallelization
---------------

Speech processing usually deals with big data set henceforth requiring
the parallelization of computationally demanding operations. AMDTK
provides a thin wrapper over `ipyparallel <https://github.com/ipython/ipyparallel>`_
which allows to create and shutdown a ipyparallel server within a python
session. Please refer to the `ipyparallel documentation <https://ipyparallel.readthedocs.io/en/latest>`_
to configure your parallel environment.

Starting the ipyparallel cluster is done as follows:
::

    >>> with amdtk.parallel('default', 4) as dview:
    ...        result = dview.map_sync(lambda x: x**2, [1, 2, 3])
    ...        logger.info('result = {result}'.format(result=result))

    DEBUG 2017-07-01 17:54:00,704 [parallel] starting ipyparallel server profile=default, njobs=4
    DEBUG 2017-07-01 17:54:01,571 [parallel] waiting 20 seconds for the server to start
    INFO 2017-07-01 17:54:21,681 [parallel] connected to 4 jobs
    INFO 2017-07-01 17:54:21,771 [<ipython-input-10-ce9e1b2a3303>] result = [1, 4, 9]
    DEBUG 2017-07-01 17:54:21,771 [parallel] shutting down the ipyparallel server

``profile`` is the name of the desired ipyparallel profile and
``njobs`` is the number of workers requested. Settings ``profile`` to
``'default'`` will use the default environment (i.e. it will start the
workers on you local machine) and should work "out-of-the-box".


Loading features
----------------

In AMDTK, the process of loading the acoustic features (MFCC,
filterbank...) is a sequence of various operations such as loading the
features into memory, whitening the features or loading the phonetic
transcription associated to the features. This sequence of operation
is encapsulated into the `FeaturesLoader` object.


