****************
Installing AMDTK
****************

Dependencies
============

AMDTK relies upon several third party python packages, namely:
  * numpy
  * scipy
  * theano
  * ipyparallel
Moreover, AMDTK does not ship any features extractor so you will most
likely need to install so standard software like
`HTK <http://htk.eng.cam.ac.uk>`_ / `Kaldi <http://kaldi-asr.org>`_
to extract those.

Finally, AMDTK supports python 3 or greater.

Installing
==========

In a terminal type
::
    python setup.py install

