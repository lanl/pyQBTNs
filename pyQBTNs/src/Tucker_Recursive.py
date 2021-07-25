"""Tucker Recursive."""
import numpy as np
import math
import tensorly as tl
from .tensor_utils import split_tucker
from .Matrix_Factorization import Matrix_Factorization
from .Tucker_Iterative import Tucker_Iterative


class Tucker_Recursive():

    def __init__(self, **parameters):
        """


        Parameters
        ----------
        **parameters : dictionary
            Passed from pyQBTNs from initialization.

        Returns
        -------
        None.

        """
        if "minimum_recursive_order" in parameters:
            self.minimum_recursive_order = parameters["minimum_recursive_order"]
            del parameters["minimum_recursive_order"]
        else:
            self.minimum_recursive_order = 4
        self.MF = Matrix_Factorization(**parameters)
        self.Tucker_Iterative_solve = Tucker_Iterative(**parameters)

    def train(self, T, dimensions, ranks):
        """
        Factor the input tensor using the Tucker_Recursive algorithm

        Parameters
        ----------
        T : numpy array
            Tensor to be factored.
        dimensions : list
            tensor dimensions.
        ranks : list
            factorization ranks.

        Returns
        -------
        core : numpy array
            tensor core.
        factors : list
            list of matrix factors.

        """
        ord = len(dimensions)
        split_point = math.ceil((ord-int(dimensions[-1] == ranks[-1]))/2)
        d1, dims1, ranks1 = split_tucker(dimensions, ranks, range(split_point))
        d2, dims2, ranks2 = split_tucker(dimensions, ranks, range(split_point, ord))
        reshaped_M = np.reshape(T, (d1, d2))
        q = max(ranks1[-1], ranks2[0])

        # M1(n0*..*n_(split-1),r), M2(r,n_split*...*n_ord)
        M1, M2 = self.MF.train(reshaped_M, q)

        dims1.append(q)
        ranks1.append(0)
        # construct Tucker for the left part
        if (split_point) >= self.minimum_recursive_order:
            core1, factors1 = self.train(M1, dims1, ranks1)
        else:
            core1, factors1 = self.Tucker_Iterative_solve.train(M1, dims1, ranks1)
        dims2 = [q] + dims2
        ranks2 = [0] + ranks2
        # construct Tucker for the right part
        if len(dims2) >= self.minimum_recursive_order:
            core2, factors2 = self.train(M2, dims2, ranks2)
        else:
            core2, factors2 = self.Tucker_Iterative_solve.train(M2, dims2, ranks2)
        # merge the two Tuckers
        core = tl.tenalg.contract(core1, core1.ndim-1, core2, 0)
        factors = factors1 + factors2
        return core, factors
