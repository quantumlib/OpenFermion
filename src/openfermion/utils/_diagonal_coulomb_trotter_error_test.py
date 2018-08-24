import time
import unittest

from openfermion import count_qubits, FermionOperator, Grid, normal_ordered

from openfermion.hamiltonians import (
    dual_basis_jellium_model,
    fermi_hubbard,
    hypercube_grid_with_given_wigner_seitz_radius_and_filling,
    jellium_model)
from openfermion.utils._low_depth_trotter_error import (
    low_depth_second_order_trotter_error_bound,
    low_depth_second_order_trotter_error_operator)
from openfermion.utils._diagonal_coulomb_trotter_error import (
    potential_and_kinetic_terms_as_arrays,
    bit_mask_of_modes_acted_on_by_fermionic_terms,
    split_operator_trotter_error_operator_diagonal_two_body,
    fermionic_swap_trotter_error_operator_diagonal_two_body)


class BreakHamiltonianIntoPotentialKineticArraysTest(unittest.TestCase):

    def test_simple_hamiltonian(self):
        hamiltonian = (FermionOperator('3^ 1^ 3 1') +
                       FermionOperator('1^ 1') - FermionOperator('1^ 2')
                       - FermionOperator('2^ 1'))

        potential_terms, kinetic_terms = potential_and_kinetic_terms_as_arrays(
            hamiltonian)

        potential = sum(potential_terms, FermionOperator.zero())
        kinetic = sum(kinetic_terms, FermionOperator.zero())

        self.assertEqual(potential, (FermionOperator('1^ 1') +
                                     FermionOperator('3^ 1^ 3 1')))
        self.assertEqual(kinetic, (-FermionOperator('1^ 2') -
                                   FermionOperator('2^ 1')))

    def test_jellium_hamiltonian_correctly_broken_up(self):
        grid = Grid(2, 3, 1.)

        hamiltonian = jellium_model(grid, spinless=True, plane_wave=False)

        potential_terms, kinetic_terms = potential_and_kinetic_terms_as_arrays(
            hamiltonian)

        potential = sum(potential_terms, FermionOperator.zero())
        kinetic = sum(kinetic_terms, FermionOperator.zero())

        true_potential = dual_basis_jellium_model(grid, spinless=True,
                                                  kinetic=False)
        true_kinetic = dual_basis_jellium_model(grid, spinless=True,
                                                potential=False)
        for i in range(count_qubits(true_kinetic)):
            coeff = true_kinetic.terms.get(((i, 1), (i, 0)))
            if coeff:
                true_kinetic -= FermionOperator(((i, 1), (i, 0)), coeff)
                true_potential += FermionOperator(((i, 1), (i, 0)), coeff)

        self.assertEqual(potential, true_potential)
        self.assertEqual(kinetic, true_kinetic)

    def test_identity_recognized_as_potential_term(self):
        potential_terms, kinetic_terms = potential_and_kinetic_terms_as_arrays(
            FermionOperator.identity())

        self.assertListEqual(list(potential_terms),
                             [FermionOperator.identity()])
        self.assertListEqual(list(kinetic_terms), [])


if __name__ == '__main__':
    unittest.main()
    #compare_with_old_code = True

    #dimension = 2
    #side_length = int(sys.argv[1])
    #tunneling = 1.0
    #periodic = True

    #jellium = False

    #coulomb_data_points = numpy.array([4.])
    #fs_error_bounds = numpy.zeros(len(coulomb_data_points))
    #fs_n_terms = numpy.zeros(len(coulomb_data_points), dtype=int)
    #so_error_bounds = numpy.zeros(len(coulomb_data_points))
    #so_n_terms = numpy.zeros(len(coulomb_data_points), dtype=int)

    #for i, coulomb in enumerate(coulomb_data_points):
        #print('For 2D Fermi-Hubbard with side length %i, coulomb = %f:' % (
            #side_length, coulomb))

        #start = time.time()
        #if jellium:
            #hamiltonian = normal_ordered(jellium_model(
                #hypercube_grid_with_given_wigner_seitz_radius_and_filling(
                    #dimension, side_length, wigner_seitz_radius=10.,
                    #spinless=True), spinless=True,
                #plane_wave=False))
            #order = 'T+V'
        #else:
            #hamiltonian = normal_ordered(
                #fermi_hubbard(side_length, side_length,
                              #tunneling, coulomb, periodic=periodic))
            #order = 'V+T'

        #hamiltonian.compress()
        #print('Got Hamiltonian in %s' % str(time.time() - start))

        #start = time.time()
        #error_operator = (
            #split_operator_trotter_error_operator_diagonal_two_body(
                #hamiltonian, order))
        #error_operator.compress()

        #so_norm_bound = numpy.sum(numpy.absolute(
            #list(error_operator.terms.values())))

        #so_n_terms[i] = len(error_operator.terms)
        #so_error_bounds[i] = so_norm_bound
        #print('Took ' + str(time.time() - start) +
              #' to compute SO error operator and info')

        #start = time.time()
        #error_operator = (
            #fermionic_swap_trotter_error_operator_diagonal_two_body(
                #hamiltonian))
        #error_operator.compress()

        #fs_norm_bound = numpy.sum(numpy.absolute(
            #list(error_operator.terms.values())))

        #fs_n_terms[i] = len(error_operator.terms)
        #fs_error_bounds[i] = fs_norm_bound
        #print('Took ' + str(time.time() - start) +
              #' to compute FS error operator and info')

        #print('SO mask bound: ' + str(so_norm_bound))
        #print('FS mask bound: ' + str(fs_norm_bound))

        #if compare_with_old_code:
            #start = time.time()

            ## Unpack result into terms, indices they act on, and whether
            ## they're hopping operators.
            #result = simulation_ordered_grouped_low_depth_terms_with_info(
                #hamiltonian)
            #terms, indices, is_hopping = result

            #old_error_operator = low_depth_second_order_trotter_error_operator(
                #terms, indices, is_hopping, jellium_only=True)

            #print('Regular FS bound: ',
                  #low_depth_second_order_trotter_error_bound(
                      #terms, indices, is_hopping, jellium_only=True))
            #print('Took ' + str(time.time() - start) +
                  #' to compute FS with old code')

            #old_error_operator -= error_operator
            #print('Difference between old and new methods:',
                  #numpy.sum(numpy.absolute(list(
                      #old_error_operator.terms.values()))))
