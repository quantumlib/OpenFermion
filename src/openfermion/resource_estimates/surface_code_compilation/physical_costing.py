#coverage:ignore
import dataclasses
import datetime
import math
from typing import Tuple, Iterator


@dataclasses.dataclass(frozen=True, unsafe_hash=True)
class MagicStateFactory:
    details: str
    physical_qubit_footprint: int
    rounds: int
    failure_rate: int


@dataclasses.dataclass(frozen=True, unsafe_hash=True)
class CostEstimate:
    physical_qubit_count: int
    duration: datetime.timedelta
    algorithm_failure_probability: float


@dataclasses.dataclass(frozen=True, unsafe_hash=True)
class AlgorithmParameters:
    physical_error_rate: float
    surface_code_cycle_time: datetime.timedelta
    logical_data_qubit_distance: int
    magic_state_factory: MagicStateFactory
    toffoli_count: int
    max_allocated_logical_qubits: int
    factory_count: int
    routing_overhead_proportion: float
    proportion_of_bounding_box: float = 1

    def estimate_cost(self) -> CostEstimate:
        """Determine algorithm single-shot layout and costs for given params.

        ASSUMES:
            - There is enough routing area to get needed data qubits to the
                factories as fast as the factories run.
            - The bottleneck time cost is waiting for magic states.
        """
        logical_storage = int(
            math.ceil(self.max_allocated_logical_qubits *
                      (1 + self.routing_overhead_proportion)))
        storage_area = logical_storage * _physical_qubits_per_logical_qubit(
            self.logical_data_qubit_distance)
        distillation_area = self.factory_count \
                          * self.magic_state_factory.physical_qubit_footprint
        rounds = int(self.toffoli_count / self.factory_count *
                     self.magic_state_factory.rounds)
        distillation_failure = self.magic_state_factory.failure_rate \
                             * self.toffoli_count
        data_failure = self.proportion_of_bounding_box \
                     * _topological_error_per_unit_cell(
            self.logical_data_qubit_distance,
            gate_err=self.physical_error_rate) * logical_storage * rounds

        return CostEstimate(physical_qubit_count=storage_area +
                            distillation_area,
                            duration=rounds * self.surface_code_cycle_time,
                            algorithm_failure_probability=min(
                                1., data_failure + distillation_failure))


def _topological_error_per_unit_cell(code_distance: int,
                                     gate_err: float) -> float:
    return 0.1 * (100 * gate_err)**((code_distance + 1) / 2)


def _total_topological_error(code_distance: int, gate_err: float,
                             unit_cells: int) -> float:
    return unit_cells * _topological_error_per_unit_cell(
        code_distance, gate_err)


def iter_known_factories(physical_error_rate: float
                        ) -> Iterator[MagicStateFactory]:
    if physical_error_rate == 0.001:
        yield _two_level_t_state_factory_1p1000(
            physical_error_rate=physical_error_rate)
    yield from iter_auto_ccz_factories(physical_error_rate)


def _two_level_t_state_factory_1p1000(physical_error_rate: float
                                     ) -> MagicStateFactory:
    assert physical_error_rate == 0.001
    return MagicStateFactory(
        details="https://arxiv.org/abs/1808.06709",
        failure_rate=4 * 9 * 10**-17,
        physical_qubit_footprint=(12 * 8) * (4) *
        _physical_qubits_per_logical_qubit(31),
        rounds=6 * 31,
    )


def _autoccz_factory_dimensions(l1_distance: int,
                                l2_distance: int) -> Tuple[int, int, float]:
    """Determine the width, height, depth of the magic state factory."""
    t1_height = 4 * l1_distance / l2_distance
    t1_width = 8 * l1_distance / l2_distance
    t1_depth = 5.75 * l1_distance / l2_distance

    ccz_depth = 5
    ccz_height = 6
    ccz_width = 3
    storage_width = 2 * l1_distance / l2_distance

    ccz_rate = 1 / ccz_depth
    t1_rate = 1 / t1_depth
    t1_factories = int(math.ceil((ccz_rate * 8) / t1_rate))
    t1_factory_column_height = t1_height * math.ceil(t1_factories / 2)

    width = int(math.ceil(t1_width * 2 + ccz_width + storage_width))
    height = int(math.ceil(max(ccz_height, t1_factory_column_height)))
    depth = max(ccz_depth, t1_depth)

    return width, height, depth


def iter_auto_ccz_factories(physical_error_rate: float
                           ) -> Iterator[MagicStateFactory]:
    for l1_distance in range(5, 25, 2):
        for l2_distance in range(l1_distance + 2, 41, 2):
            w, h, d = _autoccz_factory_dimensions(l1_distance=l1_distance,
                                                  l2_distance=l2_distance)
            f = _compute_autoccz_distillation_error(
                l1_distance=l1_distance,
                l2_distance=l2_distance,
                physical_error_rate=physical_error_rate)

            yield MagicStateFactory(
                details=
                f"AutoCCZ({physical_error_rate=},{l1_distance=},{l2_distance=}",
                physical_qubit_footprint=w * h *
                _physical_qubits_per_logical_qubit(l2_distance),
                rounds=d * l2_distance,
                failure_rate=f,
            )


def _compute_autoccz_distillation_error(l1_distance: int, l2_distance: int,
                                        physical_error_rate: float) -> float:
    # Level 0
    L0_distance = l1_distance // 2
    L0_distillation_error = physical_error_rate
    L0_topological_error = _total_topological_error(
        unit_cells=100,  # Estimated 100 for T injection.
        code_distance=L0_distance,
        gate_err=physical_error_rate)
    L0_total_T_error = L0_distillation_error + L0_topological_error

    # Level 1
    L1_topological_error = _total_topological_error(
        unit_cells=1100,  # Estimated 1000 for factory, 100 for T injection.
        code_distance=l1_distance,
        gate_err=physical_error_rate)
    L1_distillation_error = 35 * L0_total_T_error**3
    L1_total_T_error = L1_distillation_error + L1_topological_error

    # Level 2
    L2_topological_error = _total_topological_error(
        unit_cells=1000,  # Estimated 1000 for factory.
        code_distance=l2_distance,
        gate_err=physical_error_rate)
    L2_distillation_error = 28 * L1_total_T_error**2
    L2_total_CCZ_or_2T_error = L2_topological_error + L2_distillation_error

    return L2_total_CCZ_or_2T_error


def _physical_qubits_per_logical_qubit(code_distance: int) -> int:
    return (code_distance + 1)**2 * 2


def cost_estimator(num_logical_qubits,
                   num_toffoli,
                   physical_error_rate=1.0E-3,
                   portion_of_bounding_box=1.):
    """
    Produce best cost in terms of physical qubits and real run time based on
    number of toffoli, number of logical qubits, and physical error rate.

    """
    best_cost = None
    best_params = None
    for factory in iter_known_factories(
            physical_error_rate=physical_error_rate):
        for logical_data_qubit_distance in range(7, 35, 2):
            params = AlgorithmParameters(
                physical_error_rate=physical_error_rate,
                surface_code_cycle_time=datetime.timedelta(microseconds=1),
                logical_data_qubit_distance=logical_data_qubit_distance,
                magic_state_factory=factory,
                toffoli_count=num_toffoli,
                max_allocated_logical_qubits=num_logical_qubits,
                factory_count=4,
                routing_overhead_proportion=0.5,
                proportion_of_bounding_box=portion_of_bounding_box)
            cost = params.estimate_cost()
            if cost.algorithm_failure_probability > 0.1:
                continue
            if best_cost is None or cost.physical_qubit_count * cost.duration \
                < best_cost.physical_qubit_count * best_cost.duration:
                best_cost = cost
                best_params = params
    return best_cost, best_params
