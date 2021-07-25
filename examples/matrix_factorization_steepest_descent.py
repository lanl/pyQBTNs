import numpy as np
from pyQBTNs import QBTNs

### Steepest descent solver takes an arduous amount of time to compute. It should run fairly quickly for small ranks though.

qbtns = QBTNs(factorization_method="Matrix_Factorization", solver_method="classical-steepest-descent")

p = 0.5
N1 = 10
N2 = 10
RANK = 8
A = np.random.choice(a=[False, True], size=(N1, RANK), p=[p, 1-p])
B = np.random.choice(a=[False, True], size=(RANK, N2), p=[p, 1-p])

print(np.matmul(A, B))

qbtns.fit(np.matmul(A, B), RANK)

print("Hamming distance =", qbtns.get_score())

print(qbtns.get_factors())
