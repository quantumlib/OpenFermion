OpenFermion
===========

.. image:: https://travis-ci.org/quantumlib/OpenFermion.svg?branch=develop
    :target: https://travis-ci.org/quantumlib/OpenFermion

.. image:: https://coveralls.io/repos/github/quantumlib/OpenFermion/badge.svg
    :target: https://coveralls.io/github/quantumlib/OpenFermion

.. image:: https://readthedocs.org/projects/openfermion/badge/?version=latest
    :target: http://openfermion.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://badge.fury.io/py/openfermion.svg
    :target: https://badge.fury.io/py/openfermion

OpenFermion is an open source effort for compiling and analyzing quantum algorithms to simulate fermionic systems, including quantum chemistry. The current version is an alpha release which features data structures and tools for obtaining and manipulating representations of fermionic Hamiltonians.

Getting started
---------------

To start using OpenFermion, clone this git repo, change directory to the top level folder and then run:

.. code-block:: bash

  python -m pip install -e .

Alternatively, one can install using pip with the command

.. code-block:: bash

  python -m pip install openfermion

For further information about how to get started please see `intro <http://openfermion.readthedocs.io/en/latest/intro.html>`__ and  `code examples <http://openfermion.readthedocs.io/en/latest/examples.html>`__. Also take a look at the the ipython notebook demo in the examples folder of this repository as well as our detailed `code documentation <http://openfermion.readthedocs.io/en/latest/openfermion.html>`__.

Plugins
-------

In order to simulate and compile quantum circuits or perform other complicated electronic structure calculations, one can install OpenFermion plugins. We currently support a circuit simulation plugin for `ProjectQ <https://projectq.ch>`__, which you can find at `OpenFermion-ProjectQ <http://github.com/quantumlib/OpenFermion-ProjectQ>`__. We also support electronic structure plugins for `Psi4 <http://psicode.org>`__, which you can find at `OpenFermion-Psi4 <http://github.com/quantumlib/OpenFermion-Psi4>`__ (recommended), and for `PySCF <https://github.com/sunqm/pyscf>`__, which you can find at `OpenFermion-PySCF <http://github.com/quantumlib/OpenFermion-PySCF>`__.

How to contribute
-----------------

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution,
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to https://cla.developers.google.com/ to see
your current agreements on file or to sign a new one. You generally only need
to submit a CLA once unless you change employers.

All submissions, including submissions by project members, require review.
We use GitHub pull requests for this purpose. Consult
`GitHub Help <https://help.github.com/articles/about-pull-requests/>`__ for
more information on using pull requests.

Furthermore, please make sure your new code comes with extensive tests! We
use automatic testing to make sure all pull requests pass tests and do not
decrease overall test coverage by too much. Make sure you adhere to our style
guide. Just have a look at our code for clues. We mostly follow pep8 and use
the pep8 linter to check for it. Code should always come with documentation,
which is generated automatically and can be found
`here <http://openfermion.readthedocs.io/en/latest/openfermion.html>`_.

Authors
-------

`Ryan Babbush <http://ryanbabbush.com>`__ (Google),
`Jarrod McClean <http://jarrodmcclean.com>`__ (Google),
Ian Kivlichan (Harvard),
Damian Steiger (ETH Zurich),
Wei Sun (Google),
Craig Gidney (Google),
Thomas Haner (ETH Zurich),
Hannah Sim (Harvard),
Vojtech Havlicek (Oxford),
Kanav Setia (Dartmouth),
Nicholas Rubin (Rigetti),
Matthew Neeley (Google) and
Dave Bacon (Google).

Questions?
----------

If you have any other questions, please contact help@openfermion.org.

Disclaimer
----------
Copyright 2017 The OpenFermion Developers. This is not an official Google product.
