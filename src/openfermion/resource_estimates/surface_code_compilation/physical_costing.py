# coverage:ignore
import dataclasses
import datetime
import math
from typing import Tuple, Iterator


@dataclasses.dataclass(frozen=True)
class MagicStateFactory:
    details: str
    physical_qubit_footprint: int
    rounds: float
    failure_rate: float


@dataclasses.dataclass(frozen=True)
class CostEstimate:
    physical_qubit_count: int
    duration: datetime.timedelta
    algorithm_failure_probability: float

    @property
    def spacetime_volume(self):
        return self.physical_qubit_count * self.duration


@dataclasses.dataclass(frozen=True)
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

    @property
    def logical_storage(self) -> int:
        return math.ceil(self.max_allocated_logical_qubits * (1 + self.routing_overhead_proportion))

    @property
    def n_rounds(self) -> int:
        """Number of error correction rounds.

        Assumptions:
            - There is enough routing area to get needed data qubits to the factories as fast
              as the factories run.
            - The bottleneck time cost is waiting for magic states.
        """
        return int(self.toffoli_count / self.factory_count * self.magic_state_factory.rounds)

    @property
    def physical_qubits_per_logical_qubit(self) -> int:
        return _physical_qubits_per_logical_qubit(self.logical_data_qubit_distance)

    @property
    def storage_area(self) -> int:
        return self.logical_storage * self.physical_qubits_per_logical_qubit

    @property
    def distillation_area(self) -> int:
        return self.factory_count * self.magic_state_factory.physical_qubit_footprint

    @property
    def distillation_failures(self) -> float:
        return self.magic_state_factory.failure_rate * self.toffoli_count

    @property
    def topological_error_per_unit_cell(self) -> float:
        return _topological_error_per_unit_cell(self.logical_data_qubit_distance,
                                                gate_err=self.physical_error_rate)

    @property
    def data_failures(self):
        return (self.proportion_of_bounding_box * self.topological_error_per_unit_cell *
                self.logical_storage * self.n_rounds)

    @property
    def physical_qubit_count(self) -> int:
        return self.storage_area + self.distillation_area

    @property
    def duration(self) -> datetime.timedelta:
        return self.n_rounds * self.surface_code_cycle_time

    @property
    def algorithm_failure_probability(self) -> float:
        return min(1.0, self.data_failures + self.distillation_failures)

    def estimate_cost(self) -> CostEstimate:
        """Determine algorithm single-shot layout and costs for these params.

        Assumptions:
            - There is enough routing area to get needed data qubits to the factories as fast
              as the factories run.
            - The bottleneck time cost is waiting for magic states.
        """
        return CostEstimate(
            physical_qubit_count=self.physical_qubit_count,
            duration=self.duration,
            algorithm_failure_probability=self.algorithm_failure_probability,
        )


def _topological_error_per_unit_cell(code_distance: int, gate_err: float) -> float:
    return 0.1 * (100 * gate_err)**((code_distance + 1) / 2)


def _total_topological_error(code_distance: int, gate_err: float,
                             unit_cells: int) -> float:
    return unit_cells * _topological_error_per_unit_cell(code_distance, gate_err)


def iter_known_factories(physical_error_rate: float) -> Iterator[MagicStateFactory]:
    if physical_error_rate == 0.001:
        yield _two_level_t_state_factory_1p1000(physical_error_rate=physical_error_rate)
    yield from iter_auto_ccz_factories(physical_error_rate)


def _two_level_t_state_factory_1p1000(physical_error_rate: float) -> MagicStateFactory:
    assert physical_error_rate == 0.001
    return MagicStateFactory(
        details="https://arxiv.org/abs/1808.06709",
        failure_rate=4 * 9 * 10**-17,
        physical_qubit_footprint=(12 * 8) * (4) * _physical_qubits_per_logical_qubit(31),
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


def iter_auto_ccz_factories(physical_error_rate: float) -> Iterator[MagicStateFactory]:
    for l1_distance in range(5, 25, 2):
        for l2_distance in range(l1_distance + 2, 41, 2):
            w, h, d = _autoccz_factory_dimensions(l1_distance=l1_distance,
                                                  l2_distance=l2_distance)
            f = _compute_autoccz_distillation_error(
                l1_distance=l1_distance,
                l2_distance=l2_distance,
                physical_error_rate=physical_error_rate)

            yield MagicStateFactory(
                details=f"AutoCCZ({physical_error_rate=},{l1_distance=},{l2_distance=}",
                physical_qubit_footprint=w * h * _physical_qubits_per_logical_qubit(l2_distance),
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


def cost_estimator(
    num_logical_qubits: int,
    num_toffoli: int,
    *,
    physical_error_rate: float = 1.0e-3,
    portion_of_bounding_box: float = 1.0,
    failure_probability_threshold: float = 0.1,
    factory_count=4,
    routing_overhead_proportion=0.5,
) -> Tuple[CostEstimate, AlgorithmParameters]:
    """Try different factories and code distances to find the optimal cost.

    This function uses AlgorithmParameters.estimate_cost to grid-search for optimal
    `magical_state_factory` and `logical_data_qubit_distance` assignments from a
    reasonable grid. We search for parameters that keep `algorithm_failure_probability`
    below `failure_probability_threshold` and minimize `CostEstimate.spacetime_volume`.

    Please see `AlgorithmParameters` for more information about cost estimates. This
    code follows the discussion in [arXiv:2202.01244](https://arxiv.org/abs/2202.01244)
    section IIIB.
    """
    best_cost = None
    best_params = None
    for factory in iter_known_factories(physical_error_rate=physical_error_rate):
        for logical_data_qubit_distance in range(7, 35, 2):
            params = AlgorithmParameters(
                physical_error_rate=physical_error_rate,
                surface_code_cycle_time=datetime.timedelta(microseconds=1),
                logical_data_qubit_distance=logical_data_qubit_distance,
                magic_state_factory=factory,
                toffoli_count=num_toffoli,
                max_allocated_logical_qubits=num_logical_qubits,
                factory_count=factory_count,
                routing_overhead_proportion=routing_overhead_proportion,
                proportion_of_bounding_box=portion_of_bounding_box,
            )
            cost = params.estimate_cost()
            if cost.algorithm_failure_probability > failure_probability_threshold:
                continue
            if best_cost is None or cost.spacetime_volume < best_cost.spacetime_volume:
                best_cost = cost
                best_params = params
    return best_cost, best_params
