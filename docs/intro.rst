.. _intro:

Tutorial
========

.. toctree::
   :maxdepth: 2

Getting started with OpenFermion
-----------------------------

Installing OpenFermion requires pip. Make sure that you are using an up-to-date
version of it by running:

.. code-block:: bash

  python -m pip --upgrade pip

To install the latest development version of OpenFermion,
clone `this <http://github.com/quantumlib/OpenFermion>`__ git repo,
change directory to the top level folder and run:

.. code-block:: bash

  python -m pip install -e .

Alternatively, if using OpenFermion as a library, one can install the last official PyPI release with:

.. code-block:: bash

  python -m pip install --pre --user openfermion

For further information about how to get started please see `intro <http://openfermion.readthedocs.io/en/latest/intro.html>`__ and  `code examples <http://openfermion.readthedocs.io/en/latest/examples.html>`__. Also take a look at the the ipython notebook demo in the examples folder of this repository as well as our detailed `code documentation <http://openfermion.readthedocs.io/en/latest/openfermion.html>`__.


Basic OpenFermion example
----------------------

To see a basic example with both fermion and qubit operators as well as whether the installation worked, try to run the following code.

.. code-block:: python

        from openfermion.ops import FermionOperator, hermitian_conjugated
        from openfermion.transforms import jordan_wigner, bravyi_kitaev
        from openfermion.utils import eigenspectrum

        # Initialize an operator.
        fermion_operator = FermionOperator('2^ 0', 3.17)
        fermion_operator += hermitian_conjugated(fermion_operator)
        print(fermion_operator)

        # Transform to qubits under the Jordan-Wigner transformation and print its spectrum.
        jw_operator = jordan_wigner(fermion_operator)
        jw_spectrum = eigenspectrum(jw_operator)
        print(jw_operator)
        print(jw_spectrum)

        # Transform to qubits under the Bravyi-Kitaev transformation and print its spectrum.
        bk_operator = bravyi_kitaev(fermion_operator)
        bk_spectrum = eigenspectrum(bk_operator)
        print(bk_operator)
        print(bk_spectrum)


This code creates the fermionic operator :math:`a^\dagger_2 a_0` and adds its Hermitian conjugate :math:`a^\dagger_0 a_2` to it. It then maps the resulting fermionic operator to qubit operators using two transforms included in OpenFermion, the Jordan-Wigner and Bravyi-Kitaev transforms. Despite the different representations, these operators are isospectral. The example also shows some of the intuitive string methods included in OpenFermion.

Further examples can be found in the docs (`Examples` in the panel on the left) and in the OpenFermion examples folder on `GitHub <http://github.com/quantumlib/OpenFermion>`_.

Plugins
-------

In order to simulate and compile quantum circuits or perform other complicated electronic structure calculations, one can install OpenFermion plugins. We currently support a circuit simulation plugin for `ProjectQ <https://projectq.ch>`__, which you can find at `OpenFermion-ProjectQ <http://github.com/quantumlib/OpenFermion-ProjectQ>`__. We also support electronic structure plugins for `Psi4 <http://psicode.org>`__, which you can find at `OpenFermion-Psi4 <http://github.com/quantumlib/OpenFermion-Psi4>`__ (recommended), and for `PySCF <https://github.com/sunqm/pyscf>`__, which you can find at `OpenFermion-PySCF <http://github.com/quantumlib/OpenFermion-PySCF>`__.
