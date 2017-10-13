OpenFermion
===========

.. image:: https://travis-ci.org/quantumlib/OpenFermion.svg?branch=master
    :target: https://travis-ci.org/quantumlib/OpenFermion

.. image:: https://coveralls.io/repos/github/quantumlib/OpenFermion/badge.svg?branch=master
    :target: https://coveralls.io/github/quantumlib/OpenFermion

.. image:: https://readthedocs.org/projects/openfermion/badge/?version=latest
    :target: http://openfermion.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://badge.fury.io/py/openfermion.svg
    :target: https://badge.fury.io/py/openfermion

OpenFermion is an open source effort for compiling and analyzing quantum algorithms to simulate fermionic systems, including quantum chemistry. The current version is an alpha release which features data structures and tools for obtaining and manipulating representations of fermionic Hamiltonians.

Getting started
---------------

Installing OpenFermion requires pip. Make sure that you are using an up-to-date version of it.
To install the latest development version of OpenFermion, clone `this <http://github.com/quantumlib/OpenFermion>`__ git repo,
change directory to the top level folder and run:

.. code-block:: bash

  python -m pip install -e .

Alternatively, if using OpenFermion as a library, one can install the last official PyPI release with:

.. code-block:: bash

  python -m pip install --pre --user openfermion

One should then install the OpenFermion plugins (see below).
For a particularly robust method of installing OpenFermion together with select
plugins, we have provided a Docker image and usage instructions in the
`docker folder <https://github.com/quantumlib/OpenFermion/tree/master/docker>`__
(the Docker image provides a virtual environment configured with the OpenFermion
libraries pre-installed).
For other information about how to get started please see `intro <http://openfermion.readthedocs.io/en/latest/intro.html>`__ and  `code examples <http://openfermion.readthedocs.io/en/latest/examples.html>`__. Also take a look at the 
`ipython notebook demo <https://github.com/quantumlib/OpenFermion/blob/master/examples/openfermion_demo.ipynb>`__
as well as our detailed `code documentation <http://openfermion.readthedocs.io/en/latest/openfermion.html>`__.

Plugins
-------

In order to simulate and compile quantum circuits or perform other complicated electronic structure calculations, one can install OpenFermion plugins. We currently support a circuit simulation plugin for `ProjectQ <https://projectq.ch>`__, which you can find at `OpenFermion-ProjectQ <http://github.com/quantumlib/OpenFermion-ProjectQ>`__. We also support electronic structure plugins for `Psi4 <http://psicode.org>`__, which you can find at `OpenFermion-Psi4 <http://github.com/quantumlib/OpenFermion-Psi4>`__ (recommended), and for `PySCF <https://github.com/sunqm/pyscf>`__, which you can find at `OpenFermion-PySCF <http://github.com/quantumlib/OpenFermion-PySCF>`__.
We also provide a Docker image with all three of these plugins pre-installed in
the docker folder of this repository.

How to contribute
-----------------

We'd love to accept your contributions and patches to OpenFermion.
There are a few small guidelines you need to follow.
Contributions to OpenFermion must be accompanied by a Contributor License Agreement.
You (or your employer) retain the copyright to your contribution,
this simply gives us permission to use and redistribute your contributions as part of the project.
Head over to https://cla.developers.google.com/
to see your current agreements on file or to sign a new one.

All submissions, including submissions by project members, require review.
We use GitHub pull requests for this purpose. Consult
`GitHub Help <https://help.github.com/articles/about-pull-requests/>`__ for
more information on using pull requests.
Furthermore, please make sure your new code comes with extensive tests!
We use automatic testing to make sure all pull requests pass tests and do not
decrease overall test coverage by too much. Make sure you adhere to our style
guide. Just have a look at our code for clues. We mostly follow
`PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ and use
the corresponding `linter <https://pypi.python.org/pypi/pep8>`_ to check for it.
Code should always come with documentation, which is generated automatically and can be found
`here <http://openfermion.readthedocs.io/en/latest/openfermion.html>`_.

Authors
-------

`Ryan Babbush <http://ryanbabbush.com>`__ (Google),
`Jarrod McClean <http://jarrodmcclean.com>`__ (Google),
`Ian Kivlichan <http://aspuru.chem.harvard.edu/ian-kivlichan/>`__ (Harvard),
Damian Steiger (ETH Zurich),
Dave Bacon (Google),
Yudong Cao (Harvard),
Craig Gidney (Google),
Thomas Haener (ETH Zurich),
Vojtech Havlicek (Oxford),
Zhang Jiang (NASA),
Matthew Neeley (Google),
Nicholas Rubin (Rigetti),
Kanav Setia (Dartmouth),
Hannah Sim (Harvard),
Wei Sun (Google) and
Kevin Sung (University of Michigan).

Questions?
----------

If you have any other questions, please contact help@openfermion.org.

Disclaimer
----------
Copyright 2017 The OpenFermion Developers.
This is not an official Google product.
