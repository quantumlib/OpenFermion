.. image:: docs/images/logo_horizontal.svg

OpenFermion is an open source library for compiling and analyzing quantum
algorithms to simulate fermionic systems, including quantum chemistry. Among
other functionalities, this version features data structures and tools
for obtaining and manipulating representations of fermionic and qubit
Hamiltonians. For more information, see our
`release paper <https://arxiv.org/abs/1710.07629>`__.



.. image:: https://github.com/quantumlib/OpenFermion/workflows/Continuous%20Integration/badge.svg
    :target: https://github.com/quantumlib/OpenFermion/workflows/Continuous%20Integration/badge.svg

.. image:: https://readthedocs.org/projects/openfermion/badge/?version=latest
    :target: http://openfermion.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://badge.fury.io/py/openfermion.svg
    :target: https://badge.fury.io/py/openfermion

.. image:: https://img.shields.io/pypi/dm/openfermion
   :target: https://img.shields.io/pypi/dm/openfermion


Run the interactive Jupyter Notebooks in Colab_ or MyBinder_:

.. _Colab: https://colab.research.google.com/github/quantumlib/OpenFermion
.. _MyBinder:  https://mybinder.org/v2/gh/quantumlib/OpenFermion/master?filepath=examples


Installation and Documentation
==============================

Installing the latest **stable** OpenFermion requires pip. Make sure that you are using an up-to-date version of it.

Documentation can be found at `quantumai.google/openfermion <https://quantumai.google/openfermion>`__ and the following links

* `Installation <https://quantumai.google/openfermion/install>`__

* `API Docs <https://quantumai.google/reference/python/openfermion/all_symbols>`__

* `Tutorials <https://quantumai.google/openfermion/tutorials/intro_to_openfermion>`__

Currently, OpenFermion is tested on Mac, Windows, and Linux. We recommend using Mac or Linux because
the electronic structure plugins are only compatible on these platforms. However,
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
Follow the links below to learn more!

High performance simulators
------------------------------------------
* `OpenFermion-FQE <https://github.com/quantumlib/OpenFermion-FQE>`__ is
  a high performance emulator of fermionic quantum evolutions specified
  by a sequence of fermion operators, which can exploit fermionic
  symmetries such as spin and particle number.

Circuit compilation plugins
------------------------------------------
* `Forest-OpenFermion <https://github.com/rigetticomputing/forestopenfermion>`__ to support integration with `Forest <https://www.rigetti.com/forest>`__.

* `SFOpenBoson <https://github.com/XanaduAI/SFOpenBoson>`__ to support integration with `Strawberry Fields <https://github.com/XanaduAI/strawberryfields>`__.

Electronic structure package plugins
------------------------------------
* `OpenFermion-Psi4 <http://github.com/quantumlib/OpenFermion-Psi4>`__ to support integration with `Psi4 <http://psicode.org>`__.

* `OpenFermion-PySCF <http://github.com/quantumlib/OpenFermion-PySCF>`__ to support integration with `PySCF <https://github.com/sunqm/pyscf>`__.

* `OpenFermion-Dirac <https://github.com/bsenjean/Openfermion-Dirac>`__ to support integration with `DIRAC <http://diracprogram.org/doku.php>`__.

* `OpenFermion-QChem <https://github.com/qchemsoftware/OpenFermion-QChem>`__ to support integration with `Q-Chem <https://www.q-chem.com>`__.

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

We use `Github issues <https://github.com/quantumlib/OpenFermion/issues>`__
for tracking requests and bugs. Please post questions to the
`Quantum Computing Stack Exchange <https://quantumcomputing.stackexchange.com/>`__ with an 'openfermion' tag.

Authors
=======

`Ryan Babbush <http://ryanbabbush.com>`__ (Google),
`Jarrod McClean <http://jarrodmcclean.com>`__ (Google),
`Nicholas Rubin <https://github.com/ncrubin>`__ (Google),
`Kevin Sung <https://github.com/kevinsung>`__ (University of Michigan),
`Ian Kivlichan <http://aspuru.chem.harvard.edu/ian-kivlichan/>`__ (Harvard),
`Dave Bacon <https://github.com/dabacon>`__ (Google),
`Xavier Bonet-Monroig <https://github.com/xabomon>`__  (Leiden University),
`Yudong Cao <https://github.com/yudongcao>`__ (Harvard),
`Chengyu Dai <https://github.com/jdaaph>`__ (University of Michigan),
`E. Schuyler Fried <https://github.com/schuylerfried>`__ (Harvard),
`Craig Gidney <https://github.com/Strilanc>`__ (Google),
`Brendan Gimby <https://github.com/bgimby>`__ (University of Michigan),
`Pranav Gokhale <https://github.com/singular-value>`__ (University of Chicago),
`Thomas Häner <https://github.com/thomashaener>`__ (ETH Zurich),
`Tarini Hardikar <https://github.com/TariniHardikar>`__ (Dartmouth),
`Vojtĕch Havlíček <https://github.com/VojtaHavlicek>`__ (Oxford),
`Oscar Higgott <https://github.com/oscarhiggott>`__ (University College London),
`Cupjin Huang <https://github.com/pertoX4726>`__ (University of Michigan),
`Josh Izaac <https://github.com/josh146>`__ (Xanadu),
`Zhang Jiang <https://ti.arc.nasa.gov/profile/zjiang3>`__ (NASA),
`William Kirby <https://williammkirby.com>`__ (Tufts University),
`Xinle Liu <https://github.com/sheilaliuxl>`__ (Google),
`Sam McArdle <https://github.com/sammcardle30>`__ (Oxford),
`Matthew Neeley <https://github.com/maffoo>`__ (Google),
`Thomas O'Brien <https://github.com/obriente>`__ (Leiden University),
`Bryan O'Gorman <https://ti.arc.nasa.gov/profile/bogorman>`__ (UC Berkeley, NASA),
`Isil Ozfidan <https://github.com/conta877>`__ (D-Wave Systems),
`Max Radin <https://github.com/max-radin>`__ (UC Santa Barbara),
`Jhonathan Romero <https://github.com/jromerofontalvo>`__ (Harvard),
`Daniel Sank <https://github.com/DanielSank>`__ (Google),
`Nicolas Sawaya <https://github.com/nicolassawaya>`__ (Harvard),
`Bruno Senjean <https://github.com/bsenjean>`__ (Leiden University),
`Kanav Setia <https://github.com/kanavsetia>`__ (Dartmouth),
`Hannah Sim <https://github.com/hsim13372>`__ (Harvard),
`Damian Steiger <https://github.com/damiansteiger>`__ (ETH Zurich),
`Mark Steudtner <https://github.com/msteudtner>`__  (Leiden University),
`Qiming Sun <https://github.com/sunqm>`__ (Caltech),
`Wei Sun <https://github.com/Spaceenter>`__ (Google),
`Daochen Wang <https://github.com/daochenw>`__ (River Lane Research),
`Chris Winkler <https://github.com/quid256>`__ (University of Chicago),
`Fang Zhang <https://github.com/fangzh-umich>`__ (University of Michigan) and
`Emiel Koridon <https://github.com/Emieeel>`__ (Leiden University).



How to cite
===========
When using OpenFermion for research projects, please cite:

    Jarrod R McClean, Nicholas C Rubin, Kevin J Sung, Ian D Kivlichan, Xavier Bonet-Monroig,
    Yudong Cao, Chengyu Dai, E Schuyler Fried, Craig Gidney, Brendan Gimby, Pranav Gokhale,
    Thomas Häner, Tarini Hardikar, Vojtěch Havlíček, Oscar Higgott, Cupjin Huang, Josh Izaac,
    Zhang Jiang, Xinle Liu, Sam McArdle, Matthew Neeley, Thomas O'Brien, Bryan O'Gorman,
    Isil Ozfidan, Maxwell D Radin, Jhonathan Romero, Nicolas P D Sawaya, Bruno Senjean,
    Kanav Setia, Sukin Sim, Damian S Steiger, Mark Steudtner, Qiming Sun, Wei Sun, Daochen Wang,
    Fang Zhang, and Ryan Babbush
    *OpenFermion: The Electronic Structure Package for Quantum Computers*.
    `Quantum Science and Technology 5.3 (2020): 034014 <https://iopscience.iop.org/article/10.1088/2058-9565/ab8ebc/meta>`__.

We are happy to include future contributors as authors on later releases.

Disclaimer
==========

Copyright 2017 The OpenFermion Developers.
This is not an official Google product.
