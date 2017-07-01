********
Tutorial
********


Parallelization
===============

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


Parallelization
===============

Test

