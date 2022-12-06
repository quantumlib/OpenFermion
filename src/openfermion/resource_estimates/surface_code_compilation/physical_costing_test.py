from openfermion.resource_estimates.surface_code_compilation.physical_costing import cost_estimator
from datetime import timedelta


def test_against_numbers_in_paper():
    # TODO: I just picked the blue highlighted row in the table. Which one is it?
    cost, params = cost_estimator(num_logical_qubits=1434, num_toffoli=7.8e9)
    assert cost.physical_qubit_count == 4_624_440
    assert timedelta(hours=72) < cost.duration < timedelta(hours=74)
    assert cost.algorithm_failure_probability < 0.1

    assert params.factory_count == 4
    assert params.physical_error_rate == 1e-3

    assert params.logical_data_qubit_distance == 29
    factory = params.magic_state_factory

    # FIXME: this doesn't match the paper
    assert factory.details == 'AutoCCZ(physical_error_rate=0.001,l1_distance=19,l2_distance=31'
