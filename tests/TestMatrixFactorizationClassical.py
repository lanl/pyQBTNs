"""
Run using:

python -m unittest TestMatrixFactorizationClassical.py
"""

import numpy as np
import unittest
from pyQBTNs import QBTNs

class TestMatrixFactorizationClassical(unittest.TestCase):
    def test_rank_2_Simulated_Annealing(self):
        qbtns = QBTNs(factorization_method="Matrix_Factorization", solver_method="classical-simulated-annealing")
        A = np.load("../data/unittest_fixed_problems/exact_factorization_matrices/10_10_2_0.5_A.npy")
        B = np.load("../data/unittest_fixed_problems/exact_factorization_matrices/10_10_2_0.5_B.npy")
        qbtns.fit(np.matmul(A, B), 2)
        self.assertEqual(0, qbtns.get_score())

    def test_rank_3_Simulated_Annealing(self):
        qbtns = QBTNs(factorization_method="Matrix_Factorization", solver_method="classical-simulated-annealing")
        A = np.load("../data/unittest_fixed_problems/exact_factorization_matrices/10_10_3_0.5_A.npy")
        B = np.load("../data/unittest_fixed_problems/exact_factorization_matrices/10_10_3_0.5_B.npy")
        qbtns.fit(np.matmul(A, B), 3)
        self.assertEqual(0, qbtns.get_score())

    def test_rank_8_Simulated_Annealing(self):
        qbtns = QBTNs(factorization_method="Matrix_Factorization", solver_method="classical-simulated-annealing")
        A = np.load("../data/unittest_fixed_problems/exact_factorization_matrices/10_10_8_0.5_A.npy")
        B = np.load("../data/unittest_fixed_problems/exact_factorization_matrices/10_10_8_0.5_B.npy")
        qbtns.fit(np.matmul(A, B), 8)
        self.assertEqual(0, qbtns.get_score())

    def test_rank_2_Steepest_Descent(self):
        qbtns = QBTNs(factorization_method="Matrix_Factorization", solver_method="classical-steepest-descent")
        A = np.load("../data/unittest_fixed_problems/exact_factorization_matrices/10_10_2_0.5_A.npy")
        B = np.load("../data/unittest_fixed_problems/exact_factorization_matrices/10_10_2_0.5_B.npy")
        qbtns.fit(np.matmul(A, B), 2)
        self.assertEqual(0, qbtns.get_score())

    def test_rank_2_Tabu_Sampler(self):
        qbtns = QBTNs(factorization_method="Matrix_Factorization", solver_method="classsical-tabu-sampler")
        A = np.load("../data/unittest_fixed_problems/exact_factorization_matrices/10_10_2_0.5_A.npy")
        B = np.load("../data/unittest_fixed_problems/exact_factorization_matrices/10_10_2_0.5_B.npy")
        qbtns.fit(np.matmul(A, B), 2)
        self.assertEqual(0, qbtns.get_score())
