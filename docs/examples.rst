.. _examples:

Examples
========

All of these examples (and more!) are explained in detail in the ipython notebook
openfermion_demo.ipynb located at `GitHub <http://github.com/quantumlib/OpenFermion>`_.
Note these examples demonstrate only a few (introductory) aspects of OpenFermion's full functionality.
To get the most out of OpenFermion, one must also install some of our plugins.
We currently support a circuit simulation plugin for `ProjectQ <https://projectq.ch>`__, which you can find at `OpenFermion-ProjectQ <http://github.com/quantumlib/OpenFermion-ProjectQ>`__. We also support electronic structure plugins for `Psi4 <http://psicode.org>`__, which you can find at `OpenFermion-Psi4 <http://github.com/quantumlib/OpenFermion-Psi4>`__ (recommended), and for `PySCF <https://github.com/sunqm/pyscf>`__, which you can find at `OpenFermion-PySCF <http://github.com/quantumlib/OpenFermion-PySCF>`__.


.. toctree::
   :maxdepth: 2

Fermionic Operators
-------------------

Fermionic systems are often treated in second quantization, where arbitrary operators can be expressed using the fermionic creation and annihilation operators, :math:`a_k^\dagger` and :math:`a_k`. Any weighted sum of products of these operators can be represented with the FermionOperator data structure in OpenFermion.

.. code-block:: python

        from openfermion.ops import FermionOperator

        my_term = FermionOperator(((3, 1), (1, 0)))
        print(my_term)

        my_term = FermionOperator('3^ 1')
        print(my_term)

These two examples yield the same fermionic operator, :math:`a_3^\dagger a_1`.

The preferred way to specify the coefficient in OpenFermion is to provide an optional coefficient argument. If not provided, the coefficient defaults to 1. In the code below, the first method is preferred. The multiplication in the last method actually creates a copy of the term, which introduces some additional cost. All inplace operands (such as +=) modify classes whereas binary operands such as + create copies. Important caveats are that the empty tuple FermionOperator(()) and the empty string FermionOperator('') initialize identity. The empty initializer FermionOperator() initializes the zero operator. We demonstrate some of these below.

.. code-block:: python

        from openfermion.ops import FermionOperator

        good_way_to_initialize = FermionOperator('3^ 1', -1.7)
        print(good_way_to_initialize)

        bad_way_to_initialize = -1.7 * FermionOperator('3^ 1')
        print(bad_way_to_initialize)

        identity = FermionOperator('')
        print(identity)

        zero_operator = FermionOperator()
        print(zero_operator)

This creates the previous FermionOperator with a coefficient -1.7, as well as the identity and zero operators.

FermionOperator has only one attribute: .terms. This attribute is the dictionary which stores the term tuples.

.. code-block:: python

        from openfermion.ops import FermionOperator

        my_operator = FermionOperator('4^ 1^ 3 9', 1. + 2.j)
        print(my_operator)
        print(my_operator.terms)

FermionOperator supports a wide range of builtins including str(), repr(), =, , /, /=, +, +=, -, -=, - and \*\*. Note that instead of supporting != and ==, we have the method .isclose(), since FermionOperators involve floats.

Qubit Operators
---------------

The QubitOperator data structure is another essential part of OpenFermion. As the name suggests, QubitOperator is used to
store qubit operators in almost exactly the same way that FermionOperator is
used to store fermion operators. For instance :math:`X_0 Z_3 Y_4` is a
QubitOperator. The internal representation of this as a terms tuple would be
:math:`((0, X),(3, Z),(4, Y))`. Note that one important difference between QubitOperator and FermionOperator is that the terms in QubitOperator are always sorted in order of tensor factor. In some cases, this enables faster manipulation. We initialize some QubitOperators below.

.. code-block:: python

        from openfermion.ops import QubitOperator

        my_first_qubit_operator = QubitOperator('X1 Y2 Z3')
        print(my_first_qubit_operator)
        print(my_first_qubit_operator.terms)

        operator_2 = QubitOperator('X3 Z4', 3.17)
        operator_2 -= 77. * my_first_qubit_operator
        print(operator_2)

Transformations
---------------

OpenFermion also provides functions for mapping FermionOperators to QubitOperators, including the Jordan-Wigner and Bravyi-Kitaev transforms.

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

We see that despite the different representation, these operators are iso-spectral. We can also apply the Jordan-Wigner transform in reverse to map arbitrary QubitOperators to FermionOperators. Note that we also demonstrate the .compress() method (a method on both FermionOperators and QubitOperators) which removes zero entries.

.. code-block:: python

        from openfermion.ops import QubitOperator
        from openfermion.transforms import jordan_wigner, reverse_jordan_wigner

        # Initialize QubitOperator.
        my_operator = QubitOperator('X0 Y1 Z2', 88.)
        my_operator += QubitOperator('Z1 Z4', 3.17)
        print(my_operator)

        # Map QubitOperator to a FermionOperator.
        mapped_operator = reverse_jordan_wigner(my_operator)
        print(mapped_operator)

        # Map the operator back to qubits and make sure it is the same.
        back_to_normal = jordan_wigner(mapped_operator)
        back_to_normal.compress()
        print(back_to_normal)

Sparse matrices and the Hubbard model
-------------------------------------

Often, one would like to obtain a sparse matrix representation of an operator which can be analyzed numerically. There is code in both openfermion.transforms and openfermion.utils which facilitates this. The function get_sparse_operator converts either a FermionOperator, a QubitOperator or other more advanced classes such as InteractionOperator to a scipy.sparse.csc matrix. There are numerous functions in openfermion.utils which one can call on the sparse operators such as "get_gap", "get_hartree_fock_state", "get_ground_state", ect. We show this off by computing the ground state energy of the Hubbard model. To do that, we use code from the openfermion.hamiltonians module which constructs lattice models of fermions such as Hubbard models.

.. code-block:: python

        from openfermion.hamiltonians import fermi_hubbard
        from openfermion.transforms import get_sparse_operator, jordan_wigner
        from openfermion.utils import get_ground_state

        # Set model.
        x_dimension = 2
        y_dimension = 2
        tunneling = 2.
        coulomb = 1.
        magnetic_field = 0.5
        chemical_potential = 0.25
        periodic = 1
        spinless = 1

        # Get fermion operator.
        hubbard_model = fermi_hubbard(
            x_dimension, y_dimension, tunneling, coulomb, chemical_potential,
            magnetic_field, periodic, spinless)
        print(hubbard_model)

        # Get qubit operator under Jordan-Wigner.
        jw_hamiltonian = jordan_wigner(hubbard_model)
        jw_hamiltonian.compress()
        print(jw_hamiltonian)

        # Get scipy.sparse.csc representation.
        sparse_operator = get_sparse_operator(hubbard_model)
        print(sparse_operator)
        print('\nEnergy of the model is {} in units of T and J.'.format(
            get_ground_state(sparse_operator)[0]))


Basics of MolecularData class
-----------------------------

Data from electronic structure calculations can be saved in a OpenFermion data structure called MolecularData, which makes it easy to access within our library. Often, one would like to analyze a chemical series or look at many different Hamiltonians and sometimes the electronic structure calculations are either expensive to compute or difficult to converge (e.g. one needs to mess around with different types of SCF routines to make things converge). Accordingly, we anticipate that users will want some way to automatically database the results of their electronic structure calculations so that important data (such as the SCF intergrals) can be looked up on-the-fly if the user has computed them in the past. OpenFermion supports a data provenance strategy which saves key results of the electronic structure calculation (including pointers to files containing large amounts of data, such as the molecular integrals) in an HDF5 container.

The MolecularData class stores information about molecules. One initializes a MolecularData object by specifying parameters of a molecule such as its geometry, basis, multiplicity, charge and an optional string describing it. One can also initialize MolecularData simply by providing a string giving a filename where a previous MolecularData object was saved in an HDF5 container. One can save a MolecularData instance by calling the class's .save() method. This automatically saves the instance in a data folder specified during OpenFermion installation. The name of the file is generated automatically from the instance attributes and optionally provided description. Alternatively, a filename can also be provided as an optional input if one wishes to manually name the file.

When electronic structure calculations are run, the data files for the molecule can be automatically updated. If one wishes to later use that data they either initialize MolecularData with the instance filename or initialize the instance and then later call the .load() method.

Basis functions are provided to initialization using a string such as "6-31g". Geometries can be specified using a simple txt input file (see geometry_from_file function in molecular_data.py) or can be passed using a simple python list format demonstrated below. Atoms are specified using a string for their atomic symbol. Distances should be provided in atomic units (Bohr). Below we initialize a simple instance of MolecularData without performing any electronic structure calculations.

.. code-block:: python

        from openfermion.hamiltonians import MolecularData

        # Set parameters to make a simple molecule.
        diatomic_bond_length = .7414
        geometry = [('H', (0., 0., 0.)), ('H', (0., 0., diatomic_bond_length))]
        basis = 'sto-3g'
        multiplicity = 1
        charge = 0
        description = str(diatomic_bond_length)

        # Make molecule and print out a few interesting facts about it.
        molecule = MolecularData(geometry, basis, multiplicity,
                                 charge, description)
        print('Molecule has automatically generated name {}'.format(
            molecule.name))
        print('Information about this molecule would be saved at:\n{}\n'.format(
            molecule.filename))
        print('This molecule has {} atoms and {} electrons.'.format(
            molecule.n_atoms, molecule.n_electrons))
        for atom, atomic_number in zip(molecule.atoms, molecule.protons):
            print('Contains {} atom, which has {} protons.'.format(
                atom, atomic_number))

If we had previously computed this molecule using an electronic structure package, we can call molecule.load() to populate all sorts of interesting fields in the data structure. Though we make no assumptions about what electronic structure packages users might install, we assume that the calculations are saved in Fermilib's MolecularData objects. There may be plugins available in future. For the purposes of this example, we will load data that ships with OpenFermion to make a plot of the energy surface of hydrogen. Note that helper functions to initialize some interesting chemical benchmarks are found in openfermion.utils.

.. code-block:: python

        # Set molecule parameters.
        basis = 'sto-3g'
        multiplicity = 1
        bond_length_interval = 0.1
        n_points = 25

        # Generate molecule at different bond lengths.
        hf_energies = []
        fci_energies = []
        bond_lengths = []
        for point in range(3, n_points + 1):
            bond_length = bond_length_interval * point
            bond_lengths += [bond_length]
            description = str(round(bond_length,2))
            print(description)
            geometry = [('H', (0., 0., 0.)), ('H', (0., 0., bond_length))]
            molecule = MolecularData(
                geometry, basis, multiplicity, description=description)

            # Load data.
            molecule.load()

            # Print out some results of calculation.
            print('\nAt bond length of {} Bohr, molecular hydrogen has:'.format(
                bond_length))
            print('Hartree-Fock energy of {} Hartree.'.format(molecule.hf_energy))
            print('MP2 energy of {} Hartree.'.format(molecule.mp2_energy))
            print('FCI energy of {} Hartree.'.format(molecule.fci_energy))
            print('Nuclear repulsion energy between protons is {} Hartree.'.format(
                molecule.nuclear_repulsion))
            for orbital in range(molecule.n_orbitals):
                print('Spatial orbital {} has energy of {} Hartree.'.format(
                    orbital, molecule.orbital_energies[orbital]))
            hf_energies += [molecule.hf_energy]
            fci_energies += [molecule.fci_energy]


InteractionOperator and InteractionRDM for efficient numerical representations
------------------------------------------------------------------------------

Fermion Hamiltonians can be expressed as :math:`H=h_0+\sum_{pq} h_{pq} a^\dagger_p a_q + \frac12 \sum_{pqrs} h_{pqrs} a^\dagger_p a^\dagger_q a_r a_s`, where :math:`h_0` is a constant shift due to the nuclear repulsion and :math:`h_{pq}`  and  :math:`h_{pqrs}` are the famous molecular integrals. Since fermions interact pairwise, their energy is thus a unique function of the one-particle and two-particle reduced density matrices which are expressed in second quantization as :math:`\rho_{pq} = \langle p | a^\dagger_p a_q | q\rangle` and :math:`\rho_{pqrs} = \langle pq | a^\dagger_p a^\dagger_q a_r a_s | rs \rangle`, respectively.

Because the RDMs and molecular Hamiltonians are both compactly represented and manipulated as 2- and 4- index tensors, we can represent them in a particularly efficient form using similar data structures. The InteractionOperator data structure can be initialized for a Hamiltonian by passing the constant  h0h0  (or 0), as well as numpy arrays representing :math:`h_{pq}` (or :math:`\rho_{pq}`) and :math:`h_{pqrs}` (or :math:`\rho_{pqrs}`). Importantly, InteractionOperators can also be obtained by calling MolecularData.get_molecular_hamiltonian() or by calling the function get_interaction_operator() (found in openfermion.transforms) on a FermionOperator. The InteractionRDM data structure is similar but represents RDMs. For instance, one can get a molecular RDM by calling MolecularData.get_molecular_rdm(). When generating Hamiltonians from the MolecularData class, one can choose to restrict the system to an active space.

These classes inherit from the same base class, PolynomialTensor. This data structure overloads the slice operator [] so that one can get or set the key attributes of the InteractionOperator: .constant, .one_body_coefficients and .two_body_coefficients. For instance, InteractionOperator[(p, 1), (q, 1), (r, 0), (s, 0)] would return :math:`h_{pqrs}` and InteractionRDM would return :math:`\rho_{pqrs}`. Importantly, the class supports fast basis transformations using the method PolynomialTensor.rotate_basis(rotation_matrix). But perhaps most importantly, one can map the InteractionOperator to any of the other data structures we've described here.

Below, we load MolecularData from a saved calculation of LiH. We then obtain an InteractionOperator representation of this system in an active space. We then map that operator to qubits. We then demonstrate that one can rotate the orbital basis of the InteractionOperator using random angles to obtain a totally different operator that is still iso-spectral.

.. code-block:: python

        from openfermion.hamiltonians import MolecularData
        from openfermion.transforms import get_fermion_operator, get_sparse_operator, jordan_wigner
        from openfermion.utils import get_ground_state
        import numpy
        import scipy
        import scipy.linalg

        # Load saved file for LiH.
        diatomic_bond_length = 1.45
        geometry = [('Li', (0., 0., 0.)), ('H', (0., 0., diatomic_bond_length))]
        basis = 'sto-3g'
        multiplicity = 1

        # Set Hamiltonian parameters.
        active_space_start = 1
        active_space_stop = 3

        # Generate and populate instance of MolecularData.
        molecule = MolecularData(geometry, basis, multiplicity, description=\"1.45\")
        molecule.load()

        # Get the Hamiltonian in an active space.
        molecular_hamiltonian = molecule.get_molecular_hamiltonian(
            occupied_indices=range(active_space_start),
            active_indices=range(active_space_start, active_space_stop))

        # Map operator to fermions and qubits.
        fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
        qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
        qubit_hamiltonian.compress()
        print('The Jordan-Wigner Hamiltonian in canonical basis follows:\n{}'.format(qubit_hamiltonian))

        # Get sparse operator and ground state energy.
        sparse_hamiltonian = get_sparse_operator(qubit_hamiltonian)
        energy, state = get_ground_state(sparse_hamiltonian)
        print('Ground state energy before rotation is {} Hartree.\n'.format(energy))

        # Randomly rotate.
        n_orbitals = molecular_hamiltonian.n_qubits // 2
        n_variables = int(n_orbitals * (n_orbitals - 1) / 2)
        random_angles = numpy.pi * (1. - 2. * numpy.random.rand(n_variables))
        kappa = numpy.zeros((n_orbitals, n_orbitals))
        index = 0
        for p in range(n_orbitals):
            for q in range(p + 1, n_orbitals):
                kappa[p, q] = random_angles[index]
                kappa[q, p] = -numpy.conjugate(random_angles[index])
                index += 1

            # Build the unitary rotation matrix.
            difference_matrix = kappa + kappa.transpose()
            rotation_matrix = scipy.linalg.expm(kappa)

            # Apply the unitary.
            molecular_hamiltonian.rotate_basis(rotation_matrix)

        # Get qubit Hamiltonian in rotated basis.
        qubit_hamiltonian = jordan_wigner(molecular_hamiltonian)
        qubit_hamiltonian.compress()
        print('The Jordan-Wigner Hamiltonian in rotated basis follows:\n{}'.format(qubit_hamiltonian))

        # Get sparse Hamiltonian and energy in rotated basis.
        sparse_hamiltonian = get_sparse_operator(qubit_hamiltonian)
        energy, state = get_ground_state(sparse_hamiltonian)
        print('Ground state energy after rotation is {} Hartree.'.format(energy))
