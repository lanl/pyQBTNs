"""Tensor Train Recursive."""
import numpy as np
import math
from .Matrix_Factorization import Matrix_Factorization
from .tensor_utils import split_TT


class Tensor_Train_Recursive():

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
        self.MF = Matrix_Factorization(**parameters)

    def train(self, T, dimensions, ranks):
        """
        Factor the input tensor using the Tensor_Train_Recursive algorithm

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
        TTlist : list
            List of factors.

        """
        ord = len(dimensions)
        split_point = math.ceil((ord-int(dimensions[-1]==ranks[-1]))/2)
        d1, dims1, ranks1 = split_TT(T, dimensions, ranks, range(split_point))
        d2, dims2, ranks2 = split_TT(T, dimensions, ranks, range(split_point,ord))
        reshaped_M = np.reshape(T, (d1, d2))
        M1, M2 = self.MF.train(reshaped_M, ranks1[-1])  # M1(n0*..*n_(split-1),r), M2(r,n_split*...*n_ord)
        dims1.append(ranks[split_point-1])
        ranks1.append(ranks[split_point-1])
        if (split_point) > 2 or (split_point>1 and dims1[0]>ranks[0]):
            TTlist1 = self.train(M1, dims1, ranks1)
        else:
            TTlist1 = [np.reshape(M1, dims1)]

        dims2 = [ranks[split_point-1]] + dims2
        ranks2 = [ranks[split_point-1]] + ranks2
        if (len(dims2) > 3) or (len(dims2) > 2 and dims2[-1]>ranks[-1]):
            TTlist2 = self.train(M2, dims2, ranks2)
        else:
            TTlist2 = [np.reshape(M2, dims2)]
        return(TTlist1 + TTlist2)
