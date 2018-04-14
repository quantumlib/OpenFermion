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
"""Slow but simple way to generate RDMs for unit tests"""
import os
import numpy
from itertools import product
import h5py
from openfermion.hamiltonians import MolecularData
from openfermion.ops import FermionOperator
from openfermion.transforms import get_sparse_operator
from openfermion.utils import get_ground_state
from openfermion.config import DATA_DIRECTORY


def generate_tqdm(molecule, state):
    """
    Generate the 2-hole-RDM

    Args:
        molecule (MolecularData):
        state (numpy.ndarray):

    Returns:
        tqdm (numpy.ndarray): 2-hole-RDM
    """
    true_tqdm = numpy.zeros_like(molecule.fci_two_rdm)
    for p, q, r, s in product(range(molecule.n_qubits), repeat=4):
        if (p < q and r < s and
           p * molecule.n_qubits + q <= r * molecule.n_qubits + s):
            fermi_op = FermionOperator(((p, 0), (q, 0), (r, 1), (s, 1)))
            sparse_fermi_op = get_sparse_operator(fermi_op,
                                                  n_qubits=molecule.n_qubits)
            true_tqdm[p, q, r, s] = numpy.conj(state).T.dot(
                sparse_fermi_op.dot(state)).real
            true_tqdm[s, r, q, p] = true_tqdm[p, q, r, s]

            true_tqdm[q, p, r, s] = -1 * true_tqdm[p, q, r, s]
            true_tqdm[s, r, p, q] = -1 * true_tqdm[p, q, r, s]

            true_tqdm[p, q, s, r] = -1 * true_tqdm[p, q, r, s]
            true_tqdm[r, s, q, p] = -1 * true_tqdm[p, q, r, s]

            true_tqdm[q, p, s, r] = true_tqdm[p, q, r, s]
            true_tqdm[r, s, p, q] = true_tqdm[p, q, r, s]

    return true_tqdm


def generate_phdm(molecule, state):
    """
    Generate the particle-hole-RDM

    Args:
        molecule (MolecularData):
        state (numpy.ndarray):

    Returns:
        phdm (numpy.ndarray): particle-hole-RDM
    """
    true_phdm = numpy.zeros_like(molecule.fci_two_rdm)
    for p, q, r, s in product(range(molecule.n_qubits), repeat=4):
        if p * molecule.n_qubits + q >= s * molecule.n_qubits + r:
            fermi_op = FermionOperator(((p, 1), (q, 0), (r, 1), (s, 0)))
            sparse_fermi_op = get_sparse_operator(fermi_op,
                                                  n_qubits=molecule.n_qubits)
            true_phdm[p, q, r, s] = numpy.conj(state).T.dot(
                sparse_fermi_op.dot(state)).real
            true_phdm[s, r, q, p] = true_phdm[p, q, r, s]

    return true_phdm


if __name__ == "__main__":
    # NOTE: It will take a very long time to generate RDMs for LiH like this
    files = ["H1-Li1_sto-3g_singlet_1.45.hdf5", "H2_sto-3g_singlet_1.4.hdf5",
             "H2_6-31g_singlet_0.75.hdf5"]
    for file in files:
        print("generating marginals for: ", file)
        molecule = MolecularData(filename=os.path.join(DATA_DIRECTORY, file))
        sparse_hamiltonian = get_sparse_operator(
            molecule.get_molecular_hamiltonian())
        energy, state = get_ground_state(sparse_hamiltonian)

        print("generating tqdm")
        tqdm = generate_tqdm(molecule, state)
        print("generating phdm")
        phdm = generate_phdm(molecule, state)

        with h5py.File('tpdm_' + file, 'w') as fid:
            fid.create_dataset('tqdm', data=tqdm)
        with h5py.File('phdm_' + file, 'w') as fid:
            fid.create_dataset('phdm', data=phdm)
