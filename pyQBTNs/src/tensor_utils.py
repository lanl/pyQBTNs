import numpy as np
import tensorly as tl


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
    HT : dictionary
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
        t = np.random.choice(a=[False, True], size=l, p=[p, 1-p])
        return t

def construct_HT(dims, ranks, p):
    """


    Parameters
    ----------
    dims : TYPE
        DESCRIPTION.
    ranks : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.

    Returns
    -------
    T : TYPE
        DESCRIPTION.

    """
    coreDims = [ranks.pop(),ranks.pop(),ranks.pop()]
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
    dims : TYPE
        DESCRIPTION.
    ranks : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.
    random_state : TYPE, optional
        DESCRIPTION. The default is 42.

    Returns
    -------
    TYPE
        DESCRIPTION.

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
    dims : TYPE
        DESCRIPTION.
    rng : TYPE
        DESCRIPTION.

    Returns
    -------
    dim : TYPE
        DESCRIPTION.
    list
        DESCRIPTION.

    """
    dim = 1
    for i in rng:
        dim *= dims[i]
    return dim, [dims[i] for i in rng]


def split_TT(M, dims, ranks, rng):
    """


    Parameters
    ----------
    M : TYPE
        DESCRIPTION.
    dims : TYPE
        DESCRIPTION.
    ranks : TYPE
        DESCRIPTION.
    rng : TYPE
        DESCRIPTION.

    Returns
    -------
    dim : TYPE
        DESCRIPTION.
    list
        DESCRIPTION.
    list
        DESCRIPTION.

    """
    dim = 1
    for i in rng:
        dim *= dims[i]
    return dim, [dims[i] for i in rng], [ranks[i] for i in rng]


def split_tucker(dims, ranks, rng):
    """


    Parameters
    ----------
    dims : TYPE
        DESCRIPTION.
    ranks : TYPE
        DESCRIPTION.
    rng : TYPE
        DESCRIPTION.

    Returns
    -------
    dim : TYPE
        DESCRIPTION.
    list
        DESCRIPTION.
    list
        DESCRIPTION.

    """
    dim = 1
    for i in rng:
        dim *= dims[i]
    return dim, [dims[i] for i in rng], [ranks[i] for i in rng]
