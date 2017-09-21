.. _intro:

Tutorial
========

.. toctree::
   :maxdepth: 2	

Getting started with OpenFermion
-----------------------------

Installing OpenFermion requires pip. Make sure that you are using an up-to-date version of it. Then, install OpenFermion, by running

.. code-block:: bash

	python -m pip install --pre --user openfermion

Alternatively, clone/download `this repo <www.openfermion.org>`_ (e.g., to your /home directory) and run

.. code-block:: bash

	cd /home/openfermion
	python -m pip install --pre --user .

This will install OpenFermion and all its dependencies automatically. OpenFermion is compatible with both Python 2 and 3.


Basic openfermion example
----------------------

To see a basic example with both fermionic and qubit operators as well as whether the installation worked, try to run the following code.

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

Further examples can be found in the docs (`Examples` in the panel on the left) and in the OpenFermion examples folder on `GitHub <www.openfermion.org>`_.

Plugins
-------

In order to generate molecular hamiltonians in Gaussian basis sets and perform other complicated electronic structure calculations, one can install plugins. We currently support Psi4 (plugin `here <www.openfermion.org>`__, recommended) and PySCF (plugin `here <www.openfermion.org>`__).
