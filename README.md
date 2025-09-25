<div align="center">

# OpenFermion

Electronic structure package for quantum computers.

[![Compatible with Python versions 3.10 and higher](https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Licensed under the Apache 2.0 license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/OpenFermion/blob/main/LICENSE)
[![OpenFermion project on PyPI](https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c)](https://pypi.org/project/OpenFermion)
[![OpenFermion downloads per month from PyPI](https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads)](https://img.shields.io/pypi/dm/OpenFermion)

[Features](#features) &ndash;
[Installation](#installation) &ndash;
[Plugins](#plugins) &ndash;
[Documentation](#cirq-documentation) &ndash;
[Contributing](#how-to-contribute) &ndash;
[Citing](#citing-openfermion) &ndash;
[Authors](#authors) &ndash;
[Contact](#contact)

</div>

## Features

OpenFermion is an open-source Python package for compiling and analyzing quantum algorithms to
simulate fermionic systems, including quantum chemistry. Among other features, it includes data
structures and tools for obtaining and manipulating representations of fermionic and qubit
Hamiltonians. More information can be found in the [release
paper](https://arxiv.org/abs/1710.07629).

## Installation

Installing the latest **stable** OpenFermion requires the Python package installer
[pip](https://pip.pypa.io). (Make sure that you are using an up-to-date version of it.)

Currently, OpenFermion is tested on Mac, Windows, and Linux. We recommend using Mac or Linux because
the electronic structure plugins are only compatible on these platforms. However, for those who
would like to use Windows, or for anyone having other difficulties with installing OpenFermion or
its plugins, we provide instructions for creating and using a Docker image – see the
[`docker/`](https://github.com/quantumlib/OpenFermion/tree/master/docker) subdirectory. The Docker
image provides a virtual environment with OpenFermion and select plugins pre-installed. The Docker
installation should run on any operating system where Docker can be used.

### User installation

To install the latest PyPI release of OpenFermion as a Python package in user mode, run the
following commands:

```shell
python -m pip install --user openfermion
```

### Developer installation

To install the latest version of OpenFermion in development mode, run the following commands:

```shell
git clone https://github.com/quantumlib/OpenFermion
cd OpenFermion
python -m pip install -e .
```

## Plugins

OpenFermion relies on modular plugin packages for significant functionality. Specifically, plugins
are used to simulate and compile quantum circuits and to perform classical electronic structure
calculations. Follow the links below to learn more!

#### High-performance simulators

*   [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE) is a high-performance emulator
    of fermionic quantum evolutions specified by a sequence of fermion operators, which can exploit
    fermionic symmetries such as spin and particle number.

#### Circuit compilation plugins

*   [Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion) to support
    integration with [Forest](https://www.rigetti.com/forest).
*   [SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson) to support integration with [Strawberry
    Fields](https://github.com/XanaduAI/strawberryfields).

#### Electronic structure package plugins

*   [OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4) to support integration with
    [Psi4](http://psicode.org).
*   [OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF) to support integration with
    [PySCF](https://github.com/sunqm/pyscf).
*   [OpenFermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac) to support integration with
    [DIRAC](http://diracprogram.org/doku.php).
*   [OpenFermion-QChem](https://github.com/qchemsoftware/OpenFermion-QChem) to support integration
    with [Q-Chem](https://www.q-chem.com).

## Documentation

Documentation for OpenFermion can be found at
[quantumai.google/openfermion](https://quantumai.google/openfermion) and the following links:

*   [Installation](https://quantumai.google/openfermion/install)
*   [API Docs](https://quantumai.google/reference/python/openfermion/all_symbols)
*   [Tutorials](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

You can run OpenFermion's interactive Jupyter Notebooks in
[Colab](https://colab.research.google.com/github/quantumlib/OpenFermion) or
[MyBinder](https://mybinder.org/v2/gh/quantumlib/OpenFermion/master?filepath=examples).

## Contributing to OpenFermion

We'd love to accept your contributions and patches to OpenFermion. There are a few small guidelines
you need to follow.

*   Contributions to OpenFermion must be accompanied by a Contributor License Agreement (CLA). You
    (or your employer) retain the copyright to your contribution; the CLA simply gives us permission
    to use and redistribute your contributions as part of the OpenFermion project. Please visit
    https://cla.developers.google.com/ to see your current agreements on file or to sign a new one.
*   All submissions, including submissions by project members, require review. We use GitHub pull
    requests for this purpose. Consult the appropriate [GitHub Help
    documentation](https://help.github.com/articles/about-pull-requests/) for more information on
    using pull requests.
*   Please make sure your new code comes with extensive tests! We use automatic testing to make sure
    all pull requests pass tests and do not decrease overall test coverage by too much.
*   Please also make sure to follow the OpenFermion source code style. We mostly follow Python's
    [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines and use the corresponding
    [linter](https://pypi.python.org/pypi/pep8) to check for it.
*   Code should always be accompanied by documentation. Formatted OpenFermion documentation is
    generated automatically and can be found [on the Quantum AI web
    site](http://openfermion.readthedocs.io/en/latest/openfermion.html).
*   We use [Github issues](https://github.com/quantumlib/OpenFermion/issues) for tracking requests
    and bugs. Please post questions to the [Quantum Computing Stack
    Exchange](https://quantumcomputing.stackexchange.com/) with an 'openfermion' tag.

## Citing OpenFermion<a name="how-to-cite-openfermion"></a><a name="how-to-cite"></a>

When using OpenFermion for research projects, please cite:

```bibtex
@article{McClean_2020,
  author    = {Jarrod R. McClean and Nicholas C. Rubin and Kevin J. Sung
              and Ian D. Kivlichan and Xavier Bonet-Monroig and Yudong Cao
              and Chengyu Dai and E. Schuyler Fried and Craig Gidney
              and Brendan Gimby and Pranav Gokhale and Thomas Häner
              and Tarini Hardikar and Vojtěch Havlíček and Oscar Higgott
              and Cupjin Huang and Josh Izaac and Zhang Jiang
              and Xinle Liu and Sam McArdle and Matthew Neeley
              and Thomas O’Brien and Bryan O’Gorman and Isil Ozfidan
              and Maxwell D. Radin and Jhonathan Romero
              and Nicolas P. D. Sawaya and Bruno Senjean and Kanav Setia
              and Sukin Sim and Damian S. Steiger and Mark Steudtner
              and Qiming Sun and Wei Sun and Daochen Wang and Fang Zhang
              and Ryan Babbush},
  year      = 2020,
  month     = jun,
  publisher = {IOP Publishing},
  volume    = 5,
  number    = 3,
  pages     = {034014},
  title     = {{OpenFermion}: The Electronic Structure Package for
              Quantum Computers},
  journal   = {Quantum Science and Technology},
}
```

We are happy to include future contributors as authors on later releases.

## Authors

[Ryan Babbush](https://ryanbabbush.com) (Google),
[Jarrod McClean](https://jarrodmcclean.com) (Google),
[Nicholas Rubin](https://github.com/ncrubin) (Google),
[Kevin Sung](https://github.com/kevinsung) (University of Michigan),
[Ian Kivlichan](https://aspuru.chem.harvard.edu/ian-kivlichan/) (Harvard),
[Dave Bacon](https://github.com/dabacon) (Google),
[Xavier Bonet-Monroig](https://github.com/xabomon) (Leiden University),
[Yudong Cao](https://github.com/yudongcao) (Harvard),
[Chengyu Dai](https://github.com/jdaaph) (University of Michigan),
[E. Schuyler Fried](https://github.com/schuylerfried) (Harvard),
[Craig Gidney](https://github.com/Strilanc) (Google),
[Brendan Gimby](https://github.com/bgimby) (University of Michigan),
[Pranav Gokhale](https://github.com/singular-value) (University of Chicago),
[Thomas Häner](https://github.com/thomashaener) (ETH Zurich),
[Tarini Hardikar](https://github.com/TariniHardikar) (Dartmouth),
[Vojtĕch Havlíček](https://github.com/VojtaHavlicek) (Oxford),
[Oscar Higgott](https://github.com/oscarhiggott) (University College London),
[Cupjin Huang](https://github.com/pertoX4726) (University of Michigan),
[Josh Izaac](https://github.com/josh146) (Xanadu),
[Zhang Jiang](https://ti.arc.nasa.gov/profile/zjiang3) (NASA),
[William Kirby](https://williammkirby.com) (Tufts University),
[Xinle Liu](https://github.com/sheilaliuxl) (Google),
[Sam McArdle](https://github.com/sammcardle30) (Oxford),
[Matthew Neeley](https://github.com/maffoo) (Google),
[Thomas O'Brien](https://github.com/obriente) (Leiden University),
[Bryan O'Gorman](https://ti.arc.nasa.gov/profile/bogorman) (UC Berkeley, NASA),
[Isil Ozfidan](https://github.com/conta877) (D-Wave Systems),
[Max Radin](https://github.com/max-radin) (UC Santa Barbara),
[Jhonathan Romero](https://github.com/jromerofontalvo) (Harvard),
[Daniel Sank](https://github.com/DanielSank) (Google),
[Nicolas Sawaya](https://github.com/nicolassawaya) (Harvard),
[Bruno Senjean](https://github.com/bsenjean) (Leiden University),
[Kanav Setia](https://github.com/kanavsetia) (Dartmouth),
[Hannah Sim](https://github.com/hsim13372) (Harvard),
[Damian Steiger](https://github.com/damiansteiger) (ETH Zurich),
[Mark Steudtner](https://github.com/msteudtner) (Leiden University),
[Qiming Sun](https://github.com/sunqm) (Caltech),
[Wei Sun](https://github.com/Spaceenter) (Google),
[Daochen Wang](https://github.com/daochenw) (River Lane Research),
[Chris Winkler](https://github.com/quid256) (University of Chicago),
[Fang Zhang](https://github.com/fangzh-umich) (University of Michigan),
and [Emiel Koridon](https://github.com/Emieeel) (Leiden University).

## Contact

For any questions or concerns not addressed here, please email quantum-oss-maintainers@google.com.

## Disclaimer

This is not an officially-supported Google product. This project is not eligible for the [Google
Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).

Copyright 2017 The OpenFermion Developers.

<div align="center">
  <a href="https://quantumai.google">
    <img width="15%" alt="Google Quantum AI"
         src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/quantum-ai-vertical.svg">
  </a>
</div>
