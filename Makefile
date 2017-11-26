# Makefile for building distributions of niftypipe.

PYTHON ?= python
NOSETESTS ?= nosetests

.PHONY: zipdoc sdist egg upload_to_pypi clean-pyc clean-so clean-build clean-ctags clean in inplace test-code test-doc test-coverage test html specs check-before-commit check

zipdoc: html
	zip documentation.zip doc/_build/html

sdist: zipdoc
	@echo "Building source distribution..."
	python setup.py sdist
	@echo "Done building source distribution."
	# XXX copy documentation.zip to dist directory.
	# XXX Somewhere the doc/_build directory is removed and causes
	# this script to fail.

egg: zipdoc
	@echo "Building egg..."
	python setup.py bdist_egg
	@echo "Done building egg."

clean-pyc:
	find . -name "*.pyc" | xargs rm -f

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f

clean-build:
	rm -rf build

clean-ctags:
	rm -f tags

clean: clean-build clean-pyc clean-so clean-ctags

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

test-code: in
	$(NOSETESTS) -s niftypipe --with-doctest

test-doc:
	$(NOSETESTS) -s --with-doctest --doctest-tests --doctest-extension=rst \
	--doctest-fixtures=_fixture doc/

test-coverage:
	$(NOSETESTS) -s --with-doctest --with-coverage --cover-package=niftypipe \
	--config=.coveragerc

test: clean test-code

html:
	@echo "building docs"
	make -C doc clean html

specs:
	@echo "Checking specs and autogenerating spec tests"
	python tools/checkspecs.py

check: check-before-commit # just a shortcut
check-before-commit: specs html test
	@echo "removed spaces"
	@echo "built docs"
	@echo "ran test"
	@echo "generated spec tests"
