This page contains a partial list of important functionality currently implemented
in OpenFermion and its plugins.


OpenFermion core functionality
==============================

Transforms
----------

* Standard Jordan-Wigner transform: method for mapping FermionOperators to QubitOperators.

* Optimized molecular Jordan-Wigner transform: numerically performant method of applying
  Jordan-Wigner to InteractionOperator that is specialized for two-body number-conserving operators.

* Reverse Jordan-Wigner transform: method for mapping QubitOperators to FermionOperators.

* Bravyi-Kitaev transform: method for mapping FermionOperators to QubitOperators with
  logarithmic many-body order. Implementation from arXiv:1208.5986.

* Bravyi-Kitaev Fenwick tree transform: method for mapping FermionOperators to QubitOperators with
  logarithmic many-body order basedon Fenwick Trees. Implementation from arXiv:1701.07072.

* Bravyi-Kitaev Superfast transform: method for mapping FermionOperators to QubitOperators
  with constant many-body order. Number of qubits required scales as number of terms.
  Implementation from arXiv:1712.00446.

* Verstraete-Cirac transform: maps a FermionOperator on a d-dimensional lattice to a
  QubitOperator on a d-dimensional lattice with twice as many degrees of freedom.
  Implementation from arXiv:cond-mat/0508353.

* Fermion transforms from binary codes: generalized machinery for implementing an
  arbitrary (possibly non-linear) fermion transform (and decoding). Can be used to reduce
  qubit requirements by exploiting symmetries in FermionOperator.
  Implementation from arXiv:1712.07067.

* Routines for converting between InteractionOperator, QuadraticHamiltonian,
  FermionOperator, QubitOperator as well sparse matrix representations.


Hamiltonians
------------

* Fermi-Hubbard model: generate Fermi-Hubbard models as FermionOperators.
  Can generate in 1D or 2D, with or without chemical potential and/or magnetic field,
  with or without particle/hole symmetry, with or without periodic boundary conditions,
  and with or without spin.

* Plane wave Hamiltonian: generate molecule electronic structure Hamiltonians
  in the plane wave basis. Can generate with an arbitrary nuclear potential, with
  nuclei having arbitrary spins at arbitrary lattice sites. Can choose basis by
  an energy cutoff or by taking frequencies on a grid. Can generate in 1D, 2D or 3D.
  Can generate with or without spins.

* Dual basis Hamiltonian: generate molecule electronic structure Hamiltonians
  in the plane wave dual basis from arXiv:1706.00023. Can generate with an arbitrary
  nuclear potential, with nuclei having arbitrary spins at arbitrary lattice sites.
  Can generate in 1D, 2D or 3D. Can generate with or without spins.

* Jellium Hamiltonian: can generate in the plane wave or dual formalism with all
  of the options for those basis sets mentioned above.

* Mean-field D-Wave Hamiltonian: can generate the Hamiltonian for BCS mean-field
  description of superconductivity. Can generate in 1D or 2D, with or without
  chemical_potential, with or without periodic boundary conditions and with
  adjustable superconducting gap and tunneling parameters.

* Generate atomic rings: can automatically generate initial MolecularData
  for atomic rings with variable atom spacing, basis, ring size and atom type.

* Generate atomic lattices: can automatically generate initial MolecularData
  for atomic lattices in 1D, 2D or 3D with variable atom spacing, lattice size and atom type.

* Generate atoms: can automatically generate initial MolecularData
  for single atom calculations. Function chooses the correct spin assignment automatically.
  Can generate for first half of periodic table with any basis set.


Core Data Structures
--------------------

* SymbolicOperator: A weighted sum of terms, each of which is a product of factors specifying
  an index (a non-negative integer) and an action on that index.
  Parent class of FermionOperator and QubitOperator.

* FermionOperator: A sum of products of fermionic ladder operators.

* QubitOperator: A sum of products of Pauli operators.

* MolecularData: Structure for storing electronic structure calculations on molecules.

* PolynomialTensor: Efficient tensor representation of an operator that corresponds
  with a multilinear polynomial in the fermionic ladder operators.
  Parent class of InteractionOperator, InteractionRDM, and QuadraticHamiltonian.

* InteractionOperator: A numerically performant class for storing 2-body operators.

* InteractionRDM: A numerically performant class for storying 2-body RDMs

* QuadraticHamiltonian: A Hamiltonian that is quadratic in the fermionic ladder operators.

* CodeOperator:

* BinaryOperator:


Utilities
---------

* Special fermion operators: helper functions generate Sz, S+, S-, S^2, majorana and number operators.

* Slater determinants: compile quantum circuits for preparing Slater determinants and
  fermionic Gaussian states using Givens rotations

* BCH expansion: compute the logarithm of a product of exponentials of operators
  up to a specified order using the Baker-Campbell-Hausdorff formula


Measurements
------------

* Generate RDM Equality Constraints:

* RDM Equality Constraint Projection:


OpenFermion-PySCF
=================


OpenFermion-Psi4
================


OpenFermion-ProjectQ
====================


Forest-OpenFermion
==================
