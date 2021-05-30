import numpy as np
from pyQBTNs import QBTNs
import sys
sys.path.append("../pyQBTNs/src/")
from tensor_utils import construct_tucker_tensor

### Set the tensor dimensions and properties

ORDER = 4
RANK = 3
N = 4
dims = [N for i in range(ORDER)]
ranks =  [RANK for i in range(ORDER)]

### Set solver_method="d-wave" in order to factorize using the default D-Wave solver in your config file
qbtns = QBTNs(factorization_method="Tucker_Iterative", solver_method="classical-simulated-annealing")

core_original, factors_original, T = construct_tucker_tensor(dims, ranks, 0.5)

print(T)

qbtns.fit(T, RANK)

print(qbtns.get_score())

factors = qbtns.get_factors()
T_prime = qbtns.get_reconstructed_tensor()
