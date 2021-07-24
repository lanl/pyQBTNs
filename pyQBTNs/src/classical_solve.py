import random
import time
import itertools
import neal
from tabu import TabuSampler
import greedy
from .utils import get_qubo, delete_keys_from_dict, get_bcols_from_samples, filter_out_stored_QUBOs, combine_QUBO_storage, column_solve_postprocess, remove_duplicate_QUBO


def call_simulated_annealing(QUBO, random_state=42):
    """
    Call the simulated annealing method from the DWave API

    Parameters
    ----------
    QUBO : dictionary
        quadratic unconstrained binary optimization problem. 
    random_state : integer, optional
        random seed. The default is 42.

    Returns
    -------
    out_vector : list
        list of dictionaries, where each dictionary is a good solution to the QUBO.
    CPU_TIME : float
        CPU process time.

    """
    sampler = neal.SimulatedAnnealingSampler()
    start = time.process_time()
    sampleset = sampler.sample_qubo(QUBO, seed=random_state)
    CPU_TIME = time.process_time()-start
    data = sampleset.samples()
    out_vector = []
    for s in data:
        s = dict(s)
        out_vector.append(s)
    return out_vector, CPU_TIME


def call_steepest_descent(QUBO, random_state=42):
    """


    Parameters
    ----------
    QUBO : dictionary
        quadratic unconstrained binary optimization problem.
    random_state : TYPE, optional
        random seed. The default is 42.

    Returns
    -------
    out_vector : list
        list of dictionaries, where each dictionary is a good solution to the QUBO.
    CPU_TIME : float
        CPU process time in seconds.

    """
    sampler = greedy.SteepestDescentSolver()
    start = time.process_time()
    sampleset = sampler.sample_qubo(QUBO, seed=random_state)
    CPU_TIME = time.process_time()-start
    data = sampleset.samples()
    out_vector = []
    for s in data:
        s = dict(s)
        out_vector.append(s)
    return out_vector, CPU_TIME


def call_tabu_sampler(QUBO, random_state=42):
    """


    Parameters
    ----------
    QUBO : dictionary
        quadratic unconstrained binary optimization problem.
    random_state : integer, optional
        random seed. The default is 42.

    Returns
    -------
    out_vector : list
        list of dictionaries, where each dictionary is a good solution to the QUBO.
    CPU_TIME : float
        CPU process time.

    """
    start = time.process_time()
    sampleset = TabuSampler().sample_qubo(QUBO, seed=random_state)
    CPU_TIME = time.process_time()-start
    data = sampleset.samples()
    out_vector = []
    for s in data:
        s = dict(s)
        out_vector.append(s)
    return out_vector, CPU_TIME


def classical_single_QUBO(As, xs, all_QUBOS, solver_method, random_state=42):
    """
    Uses classical QUBO solvers to solve individual QUBOs at a time

    Parameters
    ----------
    As : dictionary
        In this case the dictionary has a single entry because we are only solving one QUBO at a time.
        The single value is a numpy array A from x=Ab (we are solving for the column vector b). 
        The key is tracking which column-factorization sub-problem this A is from.
    xs : dictionary
        In this case the dictionary has a single entry because we are only solving one QUBO at a time.
        The only value is a numpy array (vector) of x in x=Ab. The only key is tracking which column-
        factorization sub-problem this x is from.
    all_QUBOS : dictionary
        In this case the dictionary has a single entry because we are only solving one QUBO at a time.
        The QUBO is the only value, and the key is the QUBO integer label from the embedding.
    solver_method : string
        QUBO solver method. Allowed values are "classical-simulated-annealing", 
        "classical-steepest-descent",
        "classsical-tabu-sampler",
        "d-wave"
    random_state : integer, optional
        random state seed. The default is 42.

    Returns
    -------
    bcol_solution_dict : dictionary
        DESCRIPTION.
    TOTAL_CPU_TIME : float
        total CPU process time.

    """
    assert len(list(As.keys())) == 1, "Something went wrong"
    A = As[list(As.keys())[0]]
    x = xs[list(xs.keys())[0]]
    QUBO = all_QUBOS[list(all_QUBOS.keys())[0]]
    RANK = A.shape[1]

    if solver_method == "classical-simulated-annealing":
        vectors, TOTAL_CPU_TIME = call_simulated_annealing(QUBO, random_state=random_state)
    elif solver_method == "classical-steepest-descent":
        vectors, TOTAL_CPU_TIME = call_steepest_descent(QUBO, random_state=random_state)
    elif solver_method == "classsical-tabu-sampler":
        vectors, TOTAL_CPU_TIME = call_tabu_sampler(QUBO, random_state=random_state)

    bcols = get_bcols_from_samples(vectors, A.shape[1])
    solved_bcol = column_solve_postprocess(bcols, x, A)
    bcol_solution_dict = {list(As.keys())[0]: solved_bcol}
    return bcol_solution_dict, TOTAL_CPU_TIME


def batch_classical_single_QUBO(X, N, A, B, solver_method, random_state=42):
    """
    Solves the individual column factorization problems using classical algorithms such as
    simulated anealing.

    Parameters
    ----------
    X : 2-d numpy array
        matrix to be factored.
    N : integer
        column index.
    A : 2-d numpy array
        Initial state.
    B : 2-d numpy array
        Initial state. Not used. Here for the logical consistency.
    random_state : integer, optional
        random state. The default is 42.

    Returns
    -------
    out : list
        list of (b) columns which solve the matrix factorization problem of X=AB.

    """
    random.seed(random_state)

    QUBO_storage = []
    results = {}
    RANK = A.shape[1]
    all_QUBOS = {}
    all_xcols = {}
    all_Amatrices = {}
    no_dwave_counter = []
    for col_index in range(N):
        xcol = X[:, col_index]
        QUBO = get_qubo(xcol, A, RANK)
        if QUBO == "NA":
            tmp = []
            for i in range(RANK):
                tmp.append(random.choice([0, 1]))
            results[col_index] = tmp
            no_dwave_counter.append(col_index)
        else:
            all_QUBOS[col_index] = QUBO
            all_xcols[col_index] = xcol
            all_Amatrices[col_index] = A
    assert len(all_QUBOS) == len(all_xcols), "Something went wrong"
    assert len(all_xcols) == len(all_Amatrices), "Something went wrong"
    while len(all_QUBOS) != 0:
        assert len(all_QUBOS) == len(all_xcols), "Something went wrong"
        assert len(all_xcols) == len(all_Amatrices), "Something went wrong"
        tmp = len(all_QUBOS)
        stored_bcols, all_QUBOS, all_xcols, all_Amatrices = filter_out_stored_QUBOs(
            QUBO_storage, all_QUBOS, all_xcols, all_Amatrices)
        results = {**results, **stored_bcols}
        if len(all_QUBOS) == 0:
            break
        top_QUBOs = dict(itertools.islice(all_QUBOS.items(), 1))
        top_xs = dict(itertools.islice(all_xcols.items(), 1))
        top_As = dict(itertools.islice(all_Amatrices.items(), 1))
        all_QUBOS = delete_keys_from_dict(all_QUBOS, list(top_QUBOs.keys()))
        all_xcols = delete_keys_from_dict(all_xcols, list(top_xs.keys()))
        all_Amatrices = delete_keys_from_dict(all_Amatrices, list(top_As.keys()))

        b_columns_solved, CPU_time = classical_single_QUBO(top_As, top_xs, top_QUBOs, solver_method)
        QUBO_storage = combine_QUBO_storage(QUBO_storage, top_QUBOs, b_columns_solved)
        QUBO_storage = remove_duplicate_QUBO(QUBO_storage)
        results = {**results, **b_columns_solved}
    out = []
    for i in range(N):
        out.append(results[i])
    return out
