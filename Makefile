# Helper commands to install AMDTK

doc :
	$(MAKE) html -C sphinxdoc
	rm docs/* -fr
	cp sphinxdoc/build/html/* -r docs


install :
	python setup.py install


