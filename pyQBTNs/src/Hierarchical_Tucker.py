"""Hierarchical Tucker."""
import numpy as np
from .tensor_utils import split_HT
from .Matrix_Factorization import Matrix_Factorization


class Hierarchical_Tucker():

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
        Factor the input tensor using the Tucker_Recursive algorithm.

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
        HT : dictionary
            dictionary of the tensor.

        """

        HT = {}
        ord = len(dimensions)+1
        q = ranks.pop() # the top rank of core
        d1, dims1 = split_HT(dimensions+[q], range(ord//2))
        d2, dims2 = split_HT(dimensions+[q], range(ord//2, ord))
        reshaped_M = np.reshape(T, (d1, d2))
        r_left  = ranks.pop()
        M1, M2 = self.MF.train(reshaped_M, r_left)
        if len(dims1)>1: #HT type node
            HT1 = self.train(np.reshape(M1,dims1+[r_left]), dims1, ranks+[r_left])
            HT['child1type'] = 'HT'
        else: # matrix
            HT1 = np.moveaxis(M1, 0, -1)
            HT['child1type']='M'
        HT['child1'] = HT1
        M2 = np.reshape(M2,[r_left]+dims2)
        M2 = np.moveaxis(M2, -1, 0)
        d3 = r_left*q
        M3 = np.reshape(M2 , (d3 , d2//q))
        r_right  = ranks.pop()
        M31, M32 = self.MF.train(M3, r_right)
        core = M31.reshape(q, r_left, r_right)
        HT['core'] = core
        dims2.pop()
        if len(dims2) > 1:
            M32 = np.moveaxis(M32, 0, -1)
            HT2 = self.train(np.reshape(M32,dims2+[r_left]), dims2, ranks+[r_left])
            HT['child2type'] = 'HT'
        else:
            HT2 = M32
            HT['child2type']='M'
        HT['child2'] = HT2
        return HT
