"""Tuckeer Iterative."""
import tensorly as tl
from .Matrix_Factorization import Matrix_Factorization


class Tucker_Iterative():

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
        Factor the input tensor using the Tucker_Iterative algorithm.

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
        T : numpy array
            tensor core.
        matrixList : list
            list of matrices.

        """
        T = tl.reshape(T, tuple(dimensions))
        ord = len(dimensions)
        matrixList = []
        newDims = [d for d in dimensions]
        for n in range(ord):
            if ranks[n] == 0:
                continue  # dims[n] is not an original dimension
            reshaped_T = tl.unfold(T, n)
            M1, M2 = self.MF.train(reshaped_T, ranks[n])
            matrixList.append(M1)
            newDims[n] = ranks[n]
            T = tl.reshape(M2, tuple(newDims))
        return T, matrixList
