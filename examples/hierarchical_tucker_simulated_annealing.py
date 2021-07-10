from pyQBTNs import QBTNs
import sys
sys.path.append("../pyQBTNs/src/")
from tensor_utils import construct_HT, reconstruct_HT

### Set the tensor dimensions and properties
ORDER = 4
RANK = 3
N = 4
dims = [N for i in range(ORDER)]
ranks =  [RANK for i in range(ORDER)]

### Set solver_method="d-wave" in order to factorize using the default D-Wave solver in your config file
qbtns = QBTNs(factorization_method="Hierarchical_Tucker", solver_method="classical-simulated-annealing")

ORDER = 4
RANK = 3
N = 4

dims = [N for i in range(ORDER)]
ranks =  [RANK for i in range(2*len(dims))]

HT = construct_HT(dims, ranks, 0.5)
T = reconstruct_HT(HT)
print(T)

qbtns.fit(T, RANK, dimensions=dims, ranks=ranks)

print(qbtns.get_score())

factors = qbtns.get_factors()
T_prime = qbtns.get_reconstructed_tensor()
