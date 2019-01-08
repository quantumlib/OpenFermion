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


from openfermion import MajoranaOperator


def test_majorana_operator_init():
    op = MajoranaOperator((0, 7, 4, 1, 10))
    assert op.terms == {(0, 1, 4, 7, 10): -1.0}

    op = MajoranaOperator((0, 1, 0, 1, 0))
    assert op.terms == {(0,): -1.0}

    op = MajoranaOperator((3, 2, 5, 1, 3, 4))
    assert op.terms == {(1, 2, 4, 5): 1.0}

    op = MajoranaOperator((5, 10, 4, 3, 6, 9, 6))
    assert op.terms == {(3, 4, 5, 9, 10): -1.0}
