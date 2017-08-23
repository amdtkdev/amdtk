# Helper commands to install AMDTK

doc :
	$(MAKE) html -C docs
	tar cjf docs.tar.bz2 docs


install :
	python setup.py install
