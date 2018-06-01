""" Module to remove two qubits from the problem space
    using conservation of electron number and conservation
    of electron spin. As described in arXiv:1701.08213.
"""


import openfermion
from openfermion.ops import FermionOperator, QubitOperator
from openfermion.transforms import get_fermion_operator, bravyi_kitaev_tree
from openfermion.utils import reorder, up_then_down, prune_unused_indices
import numpy







def remove_symmetry_qubits(fermion_hamil, active_orbitals, active_electrons):
    """ Returns the qubit Hamiltonian for the molecular fermionic
        Hamiltonian supplied, with two qubits removed using conservation
        of electron spin and number, as described in arXiv:1701.08213.

        Args:
            fermion_hamil: A fermionic molecular hamiltonian obtained
                           using OpenFermion. An instance of the
                           FermionOperator class.

            active_orbitals: Int type object. The number of active orbitals
                             being considered for the molecule.

            active_electrons: Int type object. The number of active electrons
                              being considered for the molecule (note, this
                              is less than the number of electrons in the
                              molecule if some orbitals have been assumed
                              filled).

        Returns:
                qubit_hamil: The qubit Hamiltonian corresponding to
                             the supplied fermionic Hamiltonian, with
                             two qubits removed using spin symmetries.

        Notes: This function reorders the spin orbitals as all spin-up, then
               all spin-down. It uses the OpenFermion bravyi_kitaev_tree
               mapping, rather than the bravyi-kitaev mapping.

    """

    def edit_hamil_for_spin(qubit_hamiltonian, spin_orbital, orbital_parity):
        """ Removes the Z terms acting on the orbital from the Hamiltonian.
        """

        new_qubit_dict = {}
        for key, val in qubit_hamiltonian.terms.items():
            # If Z acts on the specified orbital, precompute its effect and
            # remove it from the Hamiltonian.
            if (spin_orbital-1, 'Z') in key:
                new_val = val*orbital_parity
                new_key = tuple(i for i in key if i != (spin_orbital-1, 'Z'))
                # Make sure to combine terms comprised of the same operators.
                if new_qubit_dict.get(new_key) is None:
                    new_qubit_dict[new_key] = new_val
                else:
                    old_val = new_qubit_dict.get(new_key)
                    new_qubit_dict[new_key] = new_val + old_val
            else:
                # Make sure to combine terms comprised of the same operators.
                if new_qubit_dict.get(key) is None:
                    new_qubit_dict[key] = val
                else:
                    old_val = new_qubit_dict.get(key)
                    new_qubit_dict[key] = val + old_val

        qubit_hamiltonian.terms = new_qubit_dict
        qubit_hamiltonian.compress()

        return qubit_hamiltonian


    # Catch errors if inputs are of wrong type.
    if (type(fermion_hamil) is not
        openfermion.ops._fermion_operator.FermionOperator):
        raise ValueError('Supplied operator should be an instance'
                         'of FermionOperator class')
    if type(active_orbitals) is not int:
        raise ValueError('Number of active orbitals should be an integer.')
    if type(active_electrons) is not int:
        raise ValueError('Number of active electrons should be an integer.')

    # Arrange spins up then down, then BK map to qubit Hamiltonian.
    ferm_hamil_reorder = reorder(fermion_hamil, up_then_down)
    qubit_hamil = bravyi_kitaev_tree(ferm_hamil_reorder)
    qubit_hamil.compress()

    # Allocates the parity factors for the orbitals as in arXiv:1704.05018.
    remainder = active_electrons % 4
    if remainder == 0:
        parity_final_orb = 1
        parity_middle_orb = 1
    elif remainder == 1:
        parity_final_orb = -1
        parity_middle_orb = 1
    elif remainder == 2:
        parity_final_orb = 1
        parity_middle_orb = -1
    elif remainder == 3:
        parity_final_orb = -1
        parity_middle_orb = 1

    # Removes the final qubit, then the middle qubit.
    qubit_hamil = edit_hamil_for_spin(qubit_hamil, active_orbitals,
                                      parity_final_orb)
    qubit_hamil = edit_hamil_for_spin(qubit_hamil, active_orbitals/2,
                                      parity_middle_orb)
    qubit_hamil = prune_unused_indices(qubit_hamil)


    return qubit_hamil
