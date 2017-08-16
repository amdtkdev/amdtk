********
Tutorial
********


Introduction
============

AMDTK is a python library to use Bayesian phone recognition models.
Whereas it is relatively easy to read and use python code, combining
all the separate tools to build a full system is more involved. The
``example`` directory provide an example on how to combine the multiple
functionalities of AMDTK to build a phone recognizer. Please, bare in
mind that the provided example is merely one way to do it and we
encourage experienced users to modify this example the way they see
fit.


Directory Structure
===================

The ``example`` directory defines the skeleton of a working directory to
build a model with AMDTK. A good practice is to copy the ``example``
directory so we can modify the original recipe to a database specific
recipe:

::

    $ cp -r /home/user/amdtk/example /home/user/timit
    $ cd /home/user/timit
    $ ls .
    run.sh steps utils

At the top level the ``example`` directory contain one bash script
``run.sh`` and two directories ``steps`` and ``utils``. ``run.sh`` is
the main script to build the system and for most of the cases,
preparing the data and chaning the configuration in this script will
be enough to have a phone recognizer. The ``utils`` directory contains
python scripts that are built upon AMDTK library to form specific
operation (create a model, decode, generate lattices...). Finally, the
``steps`` directory contains bash scripts that defines steps of the
current recipe. These steps generally call one or several of the
python scripts in ``utils``.


Data Preparation
================

.. note:: We assume that you have already generated the acoustic
   features for the database you want to work on. AMDTK can read
   features in `HTK <http://htk.eng.cam.ac.uk>`_ or binary
   `numpy array <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`_
   format. When using numpy format, we consider the number of rows to
   correspond to the number of frame and the number of columns to
   be the dimension of the features.


Before to run anything, we need to prepare the database in order to be
able to run the ``run.sh`` script. The data should be stored in a
directory named ``data`` with the following structure :

::

    data/
        idx_phone_map.txt
        subset1/
            [transcription.txt]
            [transcription_idx.txt]
            fea_type1/
                fea_list.txt
            [fea_type2]/
                fea_lists.txt
            [...]
        [subset2]/
            [transcription.txt]
            [transcription_idx.txt]
            fea_type1/
                fea_list.txt
            [fea_type2]/
                fea_list.txt
            [...]

directories or files surrounded by '[...]' are optional. Note however
that in their absence, some part of the recipe will not be executed.
At the top level, the ``idx_phone_map.txt`` is a mapping from an index
to a (pseudo-)phone. When dealing with phonetically transcribed data
the file will look like this:

::

    0 aa
    1 ae
    2 ah
    ...

Note that the first index starts at 0. When the data is not transcribed
and you wish to perform some acoustic unit discovery you are free to
label the units as you are pleased. A common convention is to put a
prefix to clearly identify indices vs. pseudo-phones as in this
example:

::

    0 a1
    1 a2
    2 a3
    ...

Subdirectories in ``data`` split the database into a multiple of
subsets. In most of the cases you will have two or three subsets, each
one corresponding to the training, development and testing set. AMDTK
relies on Bayesian generative models and therefore, usually does not
require any development set. Each subset directory contains one
directory per type of features. The features directory contain a
single file: ``fea_list.txt`` which is the list of paths of the
features file as following:

::

    /path/to/uttid1.fea
    /path/to/uttid2.fea
    /path/to/uttid3.fea
    ...

.. warning::
   If you store your features in several directories, make sure that
   your ``uttid1``, ``uttid2``, ...  are unique across the whole
   database as the recipe will rely on them to compute the phone error
   rate or other operations.

Subset directory may also contain a ``transcription.txt`` and a
``transcriptions_idx.txt`` file. The former is the phonetic
transcription of the subset in the following format:

::

    uttid1 t h i s
    uttid2 i s
    uttid3 a
    uttid4 t r a n s c r i p t i o n
    ...

It is used to compute the phone error rate on the specific subset. The
second has the same format but the phone symbols are replaced by the
index as defined in ``data/idx_phone_map.txt``:

::

    uttid1 19 7 8 18
    uttid2 8 18
    uttid3 0
    uttid4 19 17 0 13 18 2 17 8 15 19 8 14 13
    ...

This file is necessary only for supervised training of the phone
recognizer.


Preparing the recipe
====================


# ----------
# 2 Settings
# ----------
# Configuration of the experiment goes here. The output directory of
# the experiment is based on the settings values in order to separate
# and organize different experiments. The output directory after
# completion of the recipe will have the following structure:
#
#   model_type/
#     - model_conf.txt      # Summary of the model configuration.
#     fea_type/
#       - fea_conf.txt      # Summary of the features configuration.
#       train_type/
#         - train_conf.txt  # Summary of the training configuration.
#         - model.bin
#         - stats.bin
#         steps/
#           - model1_1.bin
#           - ...
#         data_set1/
#           labels/
#             - utt1.lab
#             - ...
#           lattices/
#             - utt1.lat
#             - ...
#           - report.txt
#         data_set2/
#           labels/
#             - utt1.lab
#             - ...
#           lattices/
#             - utt1.lat
#             - ...
#           - report.txt
#         ...
#




# Features preparation
# --------------------
# The first step is to configure the features extractor and compute the
# mean and variance of the data. These statistics will be use to
# apply mean/variance normalization (optional) and for the training.

if []


