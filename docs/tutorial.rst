********
Tutorial
********

This tutorial explains step by step how to build a model to do phoneme
recognition and/or discovery.

.. note:: All along this tutorial we assume that you have imported
    AMDTK in your python session as in the code below:
    ::

        import amdtk



Basic operations
================

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
    import amdtk

    with amdtk.parallel(profile, njobs) as dview:
        # do something with dview

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


