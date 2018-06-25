.. image:: https://github.com/quantumlib/openfermion/blob/master/docs/logo.svg
    :align: left

OpenFermion is an open source effort for compiling and analyzing quantum
algorithms to simulate fermionic systems, including quantum chemistry. Among
other functionalities, the current version features data structures and tools
for obtaining and manipulating representations of fermionic and qubit
Hamiltonians. For more information, see our
`release paper <https://arxiv.org/abs/1710.07629>`__.

.. image:: https://travis-ci.org/quantumlib/OpenFermion.svg?branch=master
    :target: https://travis-ci.org/quantumlib/OpenFermion

.. image:: https://readthedocs.org/projects/openfermion/badge/?version=latest
    :target: http://openfermion.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://coveralls.io/repos/github/quantumlib/OpenFermion/badge.svg?branch=master
    :target: https://coveralls.io/github/quantumlib/OpenFermion

.. image:: https://badge.fury.io/py/openfermion.svg
    :target: https://badge.fury.io/py/openfermion

.. image:: https://img.shields.io/badge/python-2.7%2C%203.4%2C%203.5%2C%203.6-brightgreen.svg


Getting started
===============

Installing OpenFermion requires pip. Make sure that you are using an up-to-date version of it.
For information about getting started beyond what is provided below please see our
`tutorial <https://github.com/quantumlib/OpenFermion/blob/master/examples/openfermion_tutorial.ipynb>`__
in the
`examples <https://github.com/quantumlib/OpenFermion/blob/master/examples>`__ folder
as well as our detailed `code documentation <http://openfermion.readthedocs.io/en/latest/openfermion.html>`__.

Currently, OpenFermion is only tested on Mac and Linux for the reason that both
electronic structure plugins are only compatible with Mac and Linux. However,
for those who would like to use Windows, or for anyone having other difficulties
with installing OpenFermion or its plugins, we have provided a Docker image
and usage instructions in the
`docker folder <https://github.com/quantumlib/OpenFermion/tree/master/docker>`__.
The Docker image provides a virtual environment with OpenFermion and select plugins pre-installed.
The Docker installation should run on any operating system.

You might also want to explore the alpha release of the
`OpenFermion Cloud Library <https://github.com/quantumlib/OpenFermion/tree/master/cloud_library>`__
where users can share and download precomputed molecular benchmark files.


Developer install
-----------------

To install the latest version of OpenFermion (in development mode):

.. code-block:: bash

  git clone https://github.com/quantumlib/OpenFermion
  cd OpenFermion
  python -m pip install -e .

Library install
---------------

To install the latest PyPI release as a library (in user mode):

.. code-block:: bash

  python -m pip install --user openfermion


Plugins
=======

OpenFermion relies on modular plugin libraries for significant functionality.
Specifically, plugins are used to simulate and compile quantum circuits and to perform
classical electronic structure calculations.
Follow the links below to learn more about these useful plugins.

Circuit compilation and simulation plugins
------------------------------------------
* `OpenFermion-ProjectQ <http://github.com/quantumlib/OpenFermion-ProjectQ>`__ to support integration with `ProjectQ <https://projectq.ch>`__.

* `Forest-OpenFermion <https://github.com/rigetticomputing/forestopenfermion>`__ to support integration with `Forest <https://www.rigetti.com/forest>`__.

* `SFOpenBoson <https://github.com/XanaduAI/SFOpenBoson>`__ to support integration with `Strawberry Fields <https://github.com/XanaduAI/strawberryfields>`__.

Electronic structure package plugins
------------------------------------
* `OpenFermion-Psi4 <http://github.com/quantumlib/OpenFermion-Psi4>`__ to support integration with `Psi4 <http://psicode.org>`__ (recommended).

* `OpenFermion-PySCF <http://github.com/quantumlib/OpenFermion-PySCF>`__ to support integration with `PySCF <https://github.com/sunqm/pyscf>`__.


How to contribute
=================

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
=======

`Ryan Babbush <http://ryanbabbush.com>`__ (Google),
`Jarrod McClean <http://jarrodmcclean.com>`__ (Google),
`Ian Kivlichan <http://aspuru.chem.harvard.edu/ian-kivlichan/>`__ (Harvard),
`Kevin J. Sung <https://github.com/kevinsung>`__ (University of Michigan),
`Damian Steiger <https://github.com/damiansteiger>`__ (ETH Zurich),
`Dave Bacon <https://github.com/dabacon>`__ (Google),
`Yudong Cao <https://github.com/yudongcao>`__ (Harvard),
`Chengyu Dai <https://github.com/jdaaph>`__ (University of Michigan),
`E. Schuyler Fried <https://github.com/schuylerfried>`__ (Harvard),
`Craig Gidney <https://github.com/Strilanc>`__ (Google),
`Brendan Gimby <https://github.com/bgimby>`__ (University of Michigan),
`Pranav Gokhale <https://github.com/singular-value>`__ (University of Chicago),
`Thomas Häner <https://github.com/thomashaener>`__ (ETH Zurich),
`Tarini Hardikar <https://github.com/TariniHardikar>`__ (Dartmouth),
`Vojtĕch Havlíček <https://github.com/VojtaHavlicek>`__ (Oxford),
`Cupjin Huang <https://github.com/pertoX4726>`__ (University of Michigan),
`Josh Izaac <https://github.com/josh146>`__ (Xanadu),
`Zhang Jiang <https://ti.arc.nasa.gov/profile/zjiang3>`__ (NASA),
`Xinle Liu <https://github.com/sheilaliuxl>`__ (Google),
`Sam McArdle <https://github.com/sammcardle30>`__ (Oxford),
`Matthew Neeley <https://github.com/maffoo>`__ (Google),
`Thomas O'Brien <https://github.com/obriente>`__ (Leiden University),
`Isil Ozfidan <https://github.com/conta877>`__ (D-Wave Systems),
`Max Radin <https://github.com/max-radin>`__ (UC Santa Barbara),
`Jhonathan Romero <https://github.com/jromerofontalvo>`__ (Harvard),
`Nicholas Rubin <https://github.com/ncrubin>`__ (Rigetti),
`Daniel Sank <https://github.com/DanielSank>`__ (Google),
`Nicolas Sawaya <https://github.com/nicolassawaya>`__ (Harvard),
`Kanav Setia <https://github.com/kanavsetia>`__ (Dartmouth),
`Hannah Sim <https://github.com/hsim13372>`__ (Harvard),
`Mark Steudtner <https://github.com/msteudtner>`__  (Leiden University),
`Qiming Sun <https://github.com/sunqm>`__ (Caltech),
`Wei Sun <https://github.com/Spaceenter>`__ (Google),
`Daochen Wang <https://github.com/daochenw>`__ (River Lane Research),
`Chris Winkler <https://github.com/quid256>`__ (University of Chicago) and
`Fang Zhang <https://github.com/fangzh-umich>`__ (University of Michigan).


How to cite
===========
When using OpenFermion for research projects, please cite:

    Jarrod R. McClean, Ian D. Kivlichan, Kevin J. Sung, Damian S. Steiger,
    Yudong Cao, Chengyu Dai, E. Schuyler Fried, Craig Gidney, Brendan Gimby,
    Pranav Gokhale, Thomas Häner, Tarini Hardikar, Vojtĕch Havlíček,
    Cupjin Huang, Josh Izaac, Zhang Jiang, Xinle Liu, Matthew Neeley,
    Thomas O'Brien, Isil Ozfidan, Maxwell D. Radin, Jhonathan Romero,
    Nicholas Rubin, Nicolas P. D. Sawaya, Kanav Setia, Sukin Sim,
    Mark Steudtner, Qiming Sun, Wei Sun, Fang Zhang and Ryan Babbush.
    *OpenFermion: The Electronic Structure Package for Quantum Computers*.
    `arXiv:1710.07629 <https://arxiv.org/abs/1710.07629>`__. 2017.

We are happy to include future contributors as authors on later releases.


Disclaimer
==========

Copyright 2017 The OpenFermion Developers.
This is not an official Google product.
