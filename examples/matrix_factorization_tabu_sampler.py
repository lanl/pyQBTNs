import numpy as np
from pyQBTNs import QBTNs

qbtns = QBTNs(factorization_method="Matrix_Factorization", solver_method="classsical-tabu-sampler")

p = 0.5
N1 = 10
N2 = 10
RANK = 8
A = np.random.choice(a=[False, True], size=(N1, RANK), p=[p, 1-p])
B = np.random.choice(a=[False, True], size=(RANK, N2), p=[p, 1-p])

print(np.matmul(A, B))

qbtns.fit(np.matmul(A, B), RANK)

print(qbtns.get_score())

print(qbtns.get_factors())
