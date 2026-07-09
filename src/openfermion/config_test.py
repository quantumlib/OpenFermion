# Copyright 2026 Google LLC
#
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

"""Tests for config.py."""

import os
import unittest


class MockOS:
    def __init__(
        self, process_cpu_count_val=None, sched_getaffinity_val=None, cpu_count_val=4, environ=None
    ):
        if process_cpu_count_val is not None:
            self.process_cpu_count = lambda: process_cpu_count_val
        if sched_getaffinity_val is not None:
            if isinstance(sched_getaffinity_val, Exception):

                def raise_exc():
                    raise sched_getaffinity_val

                self.sched_getaffinity = lambda x: raise_exc()
            else:
                self.sched_getaffinity = lambda x: sched_getaffinity_val
        self.cpu_count = lambda: cpu_count_val
        self.environ = environ if environ is not None else {}


class GetAvailableCpuCountTest(unittest.TestCase):

    def test_get_available_cpu_count_various_os(self):
        from unittest.mock import patch
        from openfermion.config import get_available_cpu_count
        import openfermion.config as config

        # Test Case 1: Python 3.13+ (has process_cpu_count).
        mock_os = MockOS(process_cpu_count_val=8)
        with patch.object(config, "os", mock_os):
            self.assertEqual(get_available_cpu_count(), 8)

        # Test Case 1b: Python 3.13+ returns 0/None -> fallback to 1.
        mock_os = MockOS(process_cpu_count_val=0)
        with patch.object(config, "os", mock_os):
            self.assertEqual(get_available_cpu_count(), 1)

        # Test Case 2: Linux/Unix (no process_cpu_count, has sched_getaffinity).
        mock_os = MockOS(sched_getaffinity_val=[1, 2, 3, 4])
        with patch.object(config, "os", mock_os):
            self.assertEqual(get_available_cpu_count(), 4)

        # Test Case 3: Linux/Unix but sched_getaffinity raises exception.
        mock_os = MockOS(sched_getaffinity_val=ValueError("error"), cpu_count_val=4)
        with patch.object(config, "os", mock_os):
            self.assertEqual(get_available_cpu_count(), 4)

        # Test Case 4: Windows/macOS/Older Python Fallback.
        mock_os = MockOS(cpu_count_val=6)
        with patch.object(config, "os", mock_os):
            self.assertEqual(get_available_cpu_count(), 6)

        # Test Case 4b: Fallback when cpu_count returns None/0.
        mock_os = MockOS(cpu_count_val=0)
        with patch.object(config, "os", mock_os):
            self.assertEqual(get_available_cpu_count(), 1)

        # Test Case 5: pytest-xdist active with valid worker count.
        mock_os = MockOS(cpu_count_val=8, environ={"PYTEST_XDIST_WORKER_COUNT": "4"})
        with patch.object(config, "os", mock_os):
            self.assertEqual(get_available_cpu_count(), 2)

        # Test Case 6: pytest-xdist with invalid worker count.
        mock_os = MockOS(cpu_count_val=8, environ={"PYTEST_XDIST_WORKER_COUNT": "abc"})
        with patch.object(config, "os", mock_os):
            self.assertEqual(get_available_cpu_count(), 8)


class SetThreadpoolLimitsTest(unittest.TestCase):

    def test_set_threadpool_limits_fixture(self):
        import conftest
        import sys
        from unittest.mock import patch, MagicMock

        # Test Case 1: threadpoolctl is not installed.
        with patch.dict(sys.modules, {"threadpoolctl": None}):
            gen = conftest.set_threadpool_limits.__wrapped__()
            val = next(gen)
            self.assertIsNone(val)
            with self.assertRaises(StopIteration):
                next(gen)

        # Mock threadpoolctl and os.environ for the remaining cases.
        mock_threadpoolctl = MagicMock()
        mock_threadpoolctl.threadpool_limits.return_value = MagicMock()

        with patch.dict(sys.modules, {"threadpoolctl": mock_threadpoolctl}):
            # Test Case 2: PYTEST_XDIST_WORKER_COUNT is in os.environ.
            with patch.dict(os.environ, {"PYTEST_XDIST_WORKER_COUNT": "4"}):
                with patch("openfermion.config.get_available_cpu_count", return_value=2):
                    gen = conftest.set_threadpool_limits.__wrapped__()
                    val = next(gen)
                    self.assertIsNone(val)
                    mock_threadpoolctl.threadpool_limits.assert_called_once()
                    mock_threadpoolctl.threadpool_limits.assert_called_with(limits=2)
                    with self.assertRaises(StopIteration):
                        next(gen)
                    mock_threadpoolctl.reset_mock()

            # Test Case 3: PYTEST_XDIST_WORKER_COUNT is in os.environ but invalid.
            with patch.dict(os.environ, {"PYTEST_XDIST_WORKER_COUNT": "abc"}):
                with patch("openfermion.config.get_available_cpu_count", return_value=8):
                    gen = conftest.set_threadpool_limits.__wrapped__()

                    val = next(gen)
                    self.assertIsNone(val)
                    mock_threadpoolctl.threadpool_limits.assert_called_once()
                    mock_threadpoolctl.threadpool_limits.assert_called_with(limits=8)
                    with self.assertRaises(StopIteration):
                        next(gen)
                    mock_threadpoolctl.reset_mock()

            # Test Case 4: PYTEST_XDIST_WORKER_COUNT is not in os.environ.
            with patch.dict(os.environ, {}, clear=True):
                gen = conftest.set_threadpool_limits.__wrapped__()
                val = next(gen)
                self.assertIsNone(val)
                mock_threadpoolctl.threadpool_limits.assert_not_called()
                with self.assertRaises(StopIteration):
                    next(gen)


class SetThreadingLimitsTest(unittest.TestCase):

    def test_set_threading_limits(self):
        from openfermion.config import set_threading_limits
        from unittest.mock import patch

        # Clean/mocked environment dictionary.
        env = {}
        with patch.dict(os.environ, env, clear=True):
            with patch("openfermion.config.get_available_cpu_count", return_value=8):
                set_threading_limits()
                self.assertEqual(os.environ.get("MKL_NUM_THREADS"), "7")
                self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "7")
                self.assertEqual(os.environ.get("OPENBLAS_NUM_THREADS"), "7")

        # Test respecting existing values.
        env = {"OMP_NUM_THREADS": "3"}
        with patch.dict(os.environ, env, clear=True):
            with patch("openfermion.config.get_available_cpu_count", return_value=8):
                set_threading_limits()
                self.assertEqual(os.environ.get("MKL_NUM_THREADS"), "7")
                self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "3")  # Respected!
                self.assertEqual(os.environ.get("OPENBLAS_NUM_THREADS"), "7")
