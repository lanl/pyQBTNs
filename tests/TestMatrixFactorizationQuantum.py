"""
Run using:

python -m unittest TestMatrixFactorizationQuantum.py
"""

import numpy as np
import unittest
from pyQBTNs import QBTNs

class TestMatrixFactorizationQuantum(unittest.TestCase):
    def test_rank_2_Quantum_Annealing(self):
        qbtns = QBTNs(factorization_method="Matrix_Factorization", solver_method="d-wave")
        A = np.load("../data/unittest_fixed_problems/exact_factorization_matrices/10_10_2_0.5_A.npy")
        A = np.array(A, dtype=bool)
        B = np.load("../data/unittest_fixed_problems/exact_factorization_matrices/10_10_2_0.5_B.npy")
        B = np.array(B, dtype=bool)
        qbtns.fit(np.matmul(A, B), 2)
        self.assertEqual(0, qbtns.get_score())
