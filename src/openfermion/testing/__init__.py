# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from openfermioncirq.testing.example_classes import (
    ExampleAlgorithm,
    ExampleAnsatz,
    ExampleBlackBox,
    ExampleBlackBoxNoisy,
    ExampleStatefulBlackBox,
    ExampleVariationalObjective,
    ExampleVariationalObjectiveNoisy,
    LazyAlgorithm,
)

from openfermioncirq.testing.random import (
    random_interaction_operator_term,
)

from openfermioncirq.testing.wrapped import (
    assert_eigengate_implements_consistent_protocols,
    assert_equivalent_repr,
    assert_implements_consistent_protocols,
)
