"""
© 2021. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.
"""


"""Tensor Utilities."""

import numpy as np
import tensorly as tl

def construct_tensor_TT(dimensions, RANK, p):

    """
    Constructs a tensor train tensor

    Parameters
    ----------
    dimensions : list
        list of dimensions. The length of this list is the order.
    RANK : int
        factorization rank.
    p : float
        proportion of True and False elements for generating random boolean arrays.

    Returns
    -------
    reconstruct_TT(TT_list) : numpy array
        reconstructed Tensor Train tensor.

    """

    TT_list = []
    M1 = np.random.choice(a=[False, True], size=(dimensions[0], RANK), p=[p, 1-p])
    TT_list.append(M1)

    for idx in range(1, len(dimensions)-1):
        T_ = np.random.choice(a=[False, True], size=(RANK, dimensions[idx], RANK), p=[p, 1-p])
        TT_list.append(T_)

    M_end = np.random.choice(a=[False, True], size=(RANK, dimensions[len(dimensions)-1]), p=[p, 1-p])
    TT_list.append(M_end)
    return reconstruct_TT(TT_list)

def reconstruct_TT(factors):
    """
    Reconstructs a tensor given an input of factors generated from running the Tensor Train algorithm.

    Parameters
    ----------
    factors : list
        list of matrices.

    Returns
    -------
    prod : numpy array
        reconstructed tensor.

    """
    M1 = tl.tensor(factors[0])
    T2 = tl.tensor(factors[1])
    prod = tl.tenalg.contract(M1, 1, T2, 0)
    for i in range(2, len(factors)):
        f = tl.tensor(factors[i])
        prod = tl.tenalg.contract(prod, i, f, 0)
    prod = np.array(prod, dtype=bool)
    return prod


def reconstruct_HT(HT):
    """
    Reconstructs a tensor given an input of factors generated from running the
    Hierarchical Tucker algorithm.


    Parameters
    ----------
    HT : dict
        factors from Hierarchical Tucker algorithm.

    Returns
    -------
    prod : numpy array
        tensor.

    """
    core = HT['core']
    HT1 = HT['child1']
    HT2 = HT['child2']
    if HT['child1type'] == 'HT':
        HT1 = reconstruct_HT(HT1)
    if HT['child2type'] == 'HT':
        HT2 = reconstruct_HT(HT2)
    prod = tl.tenalg.contract(core, 1, HT1, 0)
    prod = tl.tenalg.contract(prod, 1, HT2, 0)
    return prod


def boolArray(l, p):
    """
    Generate random boolean array of size l with proportion of True/False according to p

    Parameters
    ----------
    l : list
        list of sizes.
    p : float
        proportion of False entries for constructing random boolean numpy arrays.

    Returns
    -------
    t : numpy array
        numpy array of order length of l.

    """
    t = np.random.choice(a=[False, True], size=l, p=[p, 1-p])
    return t

def construct_HT(dims, ranks, p):
    """
    Construct HT dictionary

    Parameters
    ----------
    dims : list
        list of dimensions.
    ranks : list
        list of ranks.
    p : float
        proportion of True and False elements for generating random boolean arrays.

    Returns
    -------
    HT : dict
        Reconstructed HT network.

    """
    coreDims = [ranks.pop(), ranks.pop(), ranks.pop()]
    l = len(dims)//2
    core = boolArray(coreDims, p)
    HT = {'core': core}
    if len(dims) >= 4:
        HT['child1type'] = 'HT'
        HT['child1'] = construct_HT(dims[:l], ranks + [coreDims[1]], p)
    else:
        HT['child1type'] = 'M'
        HT['child1'] = boolArray([coreDims[1],dims[0]], p)
    if len(dims) >= 3:
        HT['child2type'] = 'HT'
        HT['child2'] = construct_HT(dims[l:], ranks + [coreDims[2]], p)
    else:
        HT['child2type'] = 'M'
        HT['child2'] = boolArray([coreDims[2],dims[1]], p)
    return HT

def reconstruct_tucker(core, factors):
    """
    Reconstructs a tensor given an input of factors generated from running the Tucker algorithm.

    Parameters
    ----------
    core : numpy array
        core tensor.
    factors : list
        list of matrices.

    Returns
    -------
    prod : numpy array
        tensor.

    """
    prod = core
    for i in range(len(factors)):
        prod = tl.tenalg.contract(prod, 0, factors[i], 1)
    return prod


def construct_tucker_tensor(dims, ranks, p, random_state=42):
    """


    Parameters
    ----------
    dims : list
        list of dimensions.
    ranks : list
        list of ranks.
    p : float
        proportion of True and False elements for generating random boolean arrays.
    random_state : int, optional
        random state. The default is 42.

    Returns
    -------
    core : numpy array
        DESCRIPTION.
    factors : list
        List of numpy arrays, which are the constructed factors of the tensor
    tensor : numpy array
        Tensor in the form of a boolean numpy array

    """
    core = np.array(np.random.rand(*tuple(ranks)).round(), dtype=bool)
    factors = []
    for i in range(len(ranks)):
        M = np.random.choice(a=[False, True], size=(dims[i], ranks[i]), p=[p, 1-p])
        factors.append(M)
    return core, factors, reconstruct_tucker(core, factors)


def split_HT(dims, rng):
    """


    Parameters
    ----------
    dims : list
        list of dimensions.
    rng : range
        range of indexes in the dimension list.

    Returns
    -------
    dim : int
        product of dimensions in rng.
    list
        list of a subset (in rng) of dimensionss.

    """
    dim = 1
    for i in rng:
        dim *= dims[i]
    return dim, [dims[i] for i in rng]


def split_TT(M, dims, ranks, rng):
    """


    Parameters
    ----------
    M : numpy array
        Tensor or possible matrix. Boolean numpy array.
    dims : list
        list of dimensions.
    ranks : list
        list of ranks.
    rng : range
        range of indexes in the dimension list.

    Returns
    -------
    dim : int 
        product of dimensions in rng.
    list
        list of a subset (in rng) of dimensionss.
    list
        list of a subset (in rng) of ranks.

    """
    dim = 1
    for i in rng:
        dim *= dims[i]
    return dim, [dims[i] for i in rng], [ranks[i] for i in rng]


def split_tucker(dims, ranks, rng):
    """


    Parameters
    ----------
    dims : list
        list of dimensions.
    ranks : list
        list of ranks.
    rng : range
        range of indexes in the dimension list.

    Returns
    -------
    dim : int
        product of dimensions in rng.
    list
        list of a subset (in rng) of dimensionss.
    list
        list of a subset (in rng) of ranks.

    """
    dim = 1
    for i in rng:
        dim *= dims[i]
    return dim, [dims[i] for i in rng], [ranks[i] for i in rng]
