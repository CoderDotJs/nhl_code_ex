"""Unified Test Runner for Algorithmic Samples.

Discovers and executes all unit tests across both sample directories.
"""

import sys
import unittest
from pathlib import Path


def run_all_tests() -> bool:
    """Discovers and runs test cases in sample_1 and sample_2 directories.

    Returns:
        True if all tests passed successfully, False otherwise.
    """
    root_dir = Path(__file__).resolve().parent

    dir_1 = root_dir / "sample_1_push_relabel"
    dir_2 = root_dir / "sample_2_kd_tree"

    sys.path.insert(0, str(dir_1))
    sys.path.insert(0, str(dir_2))

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Discover sample 1 tests
    suite_1 = loader.discover(start_dir=str(dir_1), pattern="test_*.py", top_level_dir=str(dir_1))
    # Discover sample 2 tests
    suite_2 = loader.discover(start_dir=str(dir_2), pattern="test_*.py", top_level_dir=str(dir_2))

    suite.addTests(suite_1)
    suite.addTests(suite_2)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
