#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
""" Module to remove two qubits from the problem space using conservation
    of electron number and conservation of electron spin. As described in
    arXiv:1701.08213 and Phys. Rev. X 6, 031007.
"""

import copy

from openfermion.ops.operators import FermionOperator
from openfermion.transforms.opconversions import bravyi_kitaev_tree, reorder
from openfermion.utils.indexing import up_then_down


def symmetry_conserving_bravyi_kitaev(fermion_hamiltonian, active_orbitals, active_fermions):
    """Returns the qubit Hamiltonian for the fermionic Hamiltonian
    supplied, with two qubits removed using conservation of electron
    spin and number, as described in arXiv:1701.08213.

    Args:
        fermion_hamiltonian: A fermionic hamiltonian obtained
                             using OpenFermion. An instance
                             of the FermionOperator class.

        active_orbitals: Int type object. The number of active orbitals
                         being considered for the system.

        active_fermions: Int type object. The number of active fermions
                          being considered for the system (note, this
                          is less than the number of electrons in a
                          molecule if some orbitals have been assumed
                          filled).
    Returns:
            qubit_hamiltonian: The qubit Hamiltonian corresponding to
                               the supplied fermionic Hamiltonian, with
                               two qubits removed using spin symmetries.
    WARNING:
            Reorders orbitals from the default even-odd ordering to all
            spin-up orbitals, then all spin-down orbitals.
    Raises:
            ValueError if fermion_hamiltonian isn't of the type
            FermionOperator, or active_orbitals isn't an integer,
            or active_fermions isn't an integer.

    Notes: This function reorders the spin orbitals as all spin-up, then
           all spin-down. It uses the OpenFermion bravyi_kitaev_tree
           mapping, rather than the bravyi-kitaev mapping.
           Caution advised when using with a Fermi-Hubbard Hamiltonian;
           this technique correctly reduces the Hamiltonian only for the
           lowest energy even and odd fermion number states, not states
           with an arbitrary number of fermions.
    """
    # Catch errors if inputs are of wrong type.
    if not isinstance(fermion_hamiltonian, FermionOperator):
        raise ValueError("Supplied operator should be an instance of FermionOperator class")
    if not isinstance(active_orbitals, int):
        raise ValueError("Number of active orbitals should be an integer.")
    if not isinstance(active_fermions, int):
        raise ValueError("Number of active fermions should be an integer.")

    # Arrange spins up then down, then BK map to qubit Hamiltonian.
    fermion_hamiltonian_reorder = reorder(
        fermion_hamiltonian, up_then_down, num_modes=active_orbitals
    )
    qubit_hamiltonian = bravyi_kitaev_tree(fermion_hamiltonian_reorder, n_qubits=active_orbitals)
    qubit_hamiltonian.compress()

    # Allocates the parity factors for the orbitals as in arXiv:1704.05018.
    remainder = active_fermions % 4
    if remainder == 0:
        parity_final_orb = 1
        parity_middle_orb = 1
    elif remainder == 1:
        parity_final_orb = -1
        parity_middle_orb = -1
    elif remainder == 2:
        parity_final_orb = 1
        parity_middle_orb = -1
    else:
        parity_final_orb = -1
        parity_middle_orb = 1

    # Removes the final qubit, then the middle qubit.
    qubit_hamiltonian = edit_hamiltonian_for_spin(
        qubit_hamiltonian, active_orbitals, parity_final_orb
    )
    qubit_hamiltonian = edit_hamiltonian_for_spin(
        qubit_hamiltonian, active_orbitals / 2, parity_middle_orb
    )
    qubit_hamiltonian = remove_indices(qubit_hamiltonian, (active_orbitals / 2, active_orbitals))

    return qubit_hamiltonian


def edit_hamiltonian_for_spin(qubit_hamiltonian, spin_orbital, orbital_parity):
    """Removes the Z terms acting on the orbital from the Hamiltonian."""
    new_qubit_dict = {}
    for term, coefficient in qubit_hamiltonian.terms.items():
        # If Z acts on the specified orbital, precompute its effect and
        # remove it from the Hamiltonian.
        if (spin_orbital - 1, "Z") in term:
            new_coefficient = coefficient * orbital_parity
            new_term = tuple(i for i in term if i != (spin_orbital - 1, "Z"))
            # Make sure to combine terms comprised of the same operators.
            if new_qubit_dict.get(new_term) is None:
                new_qubit_dict[new_term] = new_coefficient
            else:
                old_coefficient = new_qubit_dict.get(new_term)
                new_qubit_dict[new_term] = new_coefficient + old_coefficient
        else:
            # Make sure to combine terms comprised of the same operators.
            if new_qubit_dict.get(term) is None:
                new_qubit_dict[term] = coefficient
            else:
                old_coefficient = new_qubit_dict.get(term)
                new_qubit_dict[term] = coefficient + old_coefficient

    qubit_hamiltonian.terms = new_qubit_dict
    qubit_hamiltonian.compress()

    return qubit_hamiltonian


def remove_indices(symbolic_operator, indices):
    """Returns the symbolic operator from which the operator with the specified
    index was removed.

    Args:
        symbolic_operator: An instance of the SymbolicOperator class.
        indices: A sequence of Int type object. The indices to be removed.

    Returns:
        The symbolic operator. The removed indices will be filled by shifting.
    """
    map = {}

    def new_index(index):
        if index in map:
            return map[index]
        map[index] = index - len(list(i for i in indices if (i - 1) < index))
        return map[index]

    new_operator = copy.deepcopy(symbolic_operator)
    new_operator.terms.clear()

    for term in symbolic_operator.terms:
        new_term = [(new_index(op[0]), op[1]) for op in term]
        new_operator.terms[tuple(new_term)] = symbolic_operator.terms[term]

    return new_operator
