***************
What is AMDTK ?
***************

The Acoustic Model Discovery ToolKit, AMDTK for short, is a python
package providing basic tools for phoneme recognition and discovery.


How it differs from Kaldi, HTK or other speech recognition systems ?
====================================================================

Whereas Kaldi, HTK and similar toolkits aim at building a full
Speech-To-Text (STT) system, AMDTK focuses only on phoneme discovery
and recognition. Moreover standard speech recognition systems are
designed for languages rich in resources, more precisely
state-of-the-art STT systems require at least a fair amount of
transcribed speech and a lexicon. On the other hand, AMDTK targets
languages with little or no transcribed data, also known as
*under-resourced* languages in the speech research jargon.
Consequently, AMDTK is built upon advandced Bayesian generative models
as discriminative models cannot be used when there is no
transcriptions available.


To whom it is designed for ?
============================

AMDTK is a research project for people having some background in
speech and machine learning. However, despite the unavoidable
complexity of such a project, we try as much as possible to build a
toolkit easy to access and to modify.


Contact
=======

More questions ? Send an email at lucas.ondel@gmail.com

