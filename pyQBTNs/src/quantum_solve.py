import time
import random
import itertools
import dimod
from dwave import embedding
from .utils import Start_DWave_connection, read_rank3_parallel_QA_embeddings, read_complete_embedding, map_embedding_to_QUBO, get_fixed_embedding, majority_vote, delete_keys_from_dict, get_bcols_from_samples, column_solve_postprocess, get_qubo, combine_QUBO_storage, remove_duplicate_QUBO, filter_out_stored_QUBOs


def parallel_quantum_annealing(all_embs, As, xs, all_QUBOS, connectivity_graph, DWave_solver):
    """
    Calls the DWave solver using the parallel quantum annealing method. Solving the boolean column factorization problem x=Ab, where x and b are columns and A is a matrix.

    Parameters
    ----------
    all_embs : dict
        Parallel QA embeddings. Keys are unique embedding identifiers. Values are the small clique embeddings
    As : dict
        Keys are problem indexes. Value is the initial state boolean array (matrix) A.
    xs : dict
        Keys are problem indexes. Value is the x-column vector to be factored. 
    all_QUBOS : dict
        Keys are problem indexes. Values are QUBO dictionaries.
    connectivity_graph : networkx.Graph()
        Undirected hardware connectivity graph of the DWave solver.
    DWave_solver : dwave.cloud.client.solver
        DWave solver object.

    Returns
    -------
    resulting_columns : dict
        solved b-columns. Each key is the index for that column (so we can stitch together the results into our B matrix).
        Each value is a list of 0 and 1 (boolean) vectors.

    """
    RANK = 3
    DWAVE_NUMBER_OF_ANNEALS_FOR_EACH_RANK = {
        2: 200, 3: 400, 4: 600, 5: 800, 6: 1000, 7: 4000, 8: 5000}
    UTC_PREFACTOR = 1.5
    params = {"num_reads": DWAVE_NUMBER_OF_ANNEALS_FOR_EACH_RANK[RANK], "annealing_time": 1}
    combine_QUBOs = {}
    all_QUBO_embeddings = {}
    index = -1
    for test in all_QUBOS:
        index += 1
        QUBO = all_QUBOS[test]
        emb = all_embs[index][1]
        C = connectivity_graph.subgraph(all_embs[index][0])
        QUBO_embedding = get_fixed_embedding(QUBO, emb)
        all_QUBO_embeddings[test] = QUBO_embedding
        bqm = dimod.BinaryQuadraticModel.from_qubo(QUBO)
        chain_strength_fixed = embedding.chain_strength.uniform_torque_compensation(
            bqm, prefactor=UTC_PREFACTOR)
        embedded_qubo = embedding.embed_qubo(
            QUBO, QUBO_embedding, C, chain_strength=chain_strength_fixed)
        combine_QUBOs = {**combine_QUBOs, **embedded_qubo}
    while True:
        try:
            sampleset = DWave_solver.sample_qubo(combine_QUBOs, answer_mode='raw', **params)
            vectors = sampleset.samples
            break
        except:
            print("D-Wave connection failed, trying again in 1 second...")
            time.sleep(1)
            continue
    resulting_columns = {}
    for test in all_QUBO_embeddings:
        A = As[test]
        x = xs[test]
        QUBO_embedding = all_QUBO_embeddings[test]
        unembedded = majority_vote(vectors, QUBO_embedding)
        bcols = get_bcols_from_samples(unembedded, RANK)
        solved_bcol = column_solve_postprocess(bcols, x, A)
        resulting_columns[test] = solved_bcol
    return resulting_columns


def batch_parallel_quantum_annealing(X, N, A, B, random_state=42):
    """
    Submits multiple column factorizaiton problems at once to D-Wave
    (Up to however many small cliques were found in the embedding stage).
    This method is called parallel quantum annealing.

    Parameters
    ----------
    X : 2-d numpy array
        matrix to be factored.
    N : int
        column index.
    A : 2-d numpy array
        Initial state.
    B : 2-d numpy array
        Initial state. Not used. Here for the logical consistency.
    random_state : int, optional
        random state. The default is 42.

    Returns
    -------
    out : list
        list of solved columns.

    """
    random.seed(random_state)

    QUBO_storage = []
    connectivity_graph, DWave_solver, solver_name = Start_DWave_connection()
    rank_3_embeddings = read_rank3_parallel_QA_embeddings(random_state=random_state)

    results = {}
    RANK = A.shape[1]
    all_QUBOS = {}
    all_xcols = {}
    all_Amatrices = {}
    no_dwave_counter = []
    for col_index in range(N):
        xcol = X[:, col_index]
        QUBO = get_qubo(xcol, A, A.shape[1])
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
        stored_bcols, all_QUBOS, all_xcols, all_Amatrices = filter_out_stored_QUBOs(
            QUBO_storage, all_QUBOS, all_xcols, all_Amatrices)
        results = {**results, **stored_bcols}
        if len(all_QUBOS) == 0:
            break
        top_QUBOs = dict(itertools.islice(all_QUBOS.items(), len(rank_3_embeddings)))
        top_xs = dict(itertools.islice(all_xcols.items(), len(rank_3_embeddings)))
        top_As = dict(itertools.islice(all_Amatrices.items(), len(rank_3_embeddings)))
        all_QUBOS = delete_keys_from_dict(all_QUBOS, list(top_QUBOs.keys()))
        all_xcols = delete_keys_from_dict(all_xcols, list(top_xs.keys()))
        all_Amatrices = delete_keys_from_dict(all_Amatrices, list(top_As.keys()))
        b_columns_solved = parallel_quantum_annealing(
            rank_3_embeddings, top_As, top_xs, top_QUBOs, connectivity_graph, DWave_solver)
        QUBO_storage = combine_QUBO_storage(QUBO_storage, top_QUBOs, b_columns_solved)
        QUBO_storage = remove_duplicate_QUBO(QUBO_storage)
        results = {**results, **b_columns_solved}
    out = []
    for i in range(N):
        out.append(results[i])
    return out



def quantum_annealing(As, xs, all_QUBOS, connectivity_graph, DWave_solver, complete_embedding):
    """
    Non-Parallel Quantum Annealing
    For ranks that are not 3

    Parameters
    ----------
    As : dict
        Keys are problem indexes. Value is the initial state boolean array (matrix) A.
    xs : dict
        Keys are problem indexes. Value is the x-column vector to be factored.
    all_QUBOS : dict
        Keys are problem indexes. Values are QUBO dictionaries.
    connectivity_graph : networkx.Graph()
        Undirected hardware connectivity graph of the DWave solver.
    DWave_solver : dwave.cloud.client.solver
        DWave solver object.
    complete_embedding : dict
        Keys are the variable indexes (for the LANL 2000Q this is 0-64). Values are lists of the physical qubits (chain) for that variable index. 

    Returns
    -------
    bcol_solution_dict : dict
        keys are the problem index. Values are the best found b-column for that particular column factorization problem.

    """
    assert len(list(As.keys())) == 1, "Something went wrong"
    DWAVE_NUMBER_OF_ANNEALS_FOR_EACH_RANK = {
        2: 200, 3: 400, 4: 600, 5: 800, 6: 1000, 7: 4000, 8: 5000}
    UTC_PREFACTOR = 1.5
    A = As[list(As.keys())[0]]
    x = xs[list(xs.keys())[0]]
    QUBO = all_QUBOS[list(all_QUBOS.keys())[0]]
    RANK = A.shape[1]
    params = {"num_reads": DWAVE_NUMBER_OF_ANNEALS_FOR_EACH_RANK[RANK], "annealing_time": 1}
    bqm = dimod.BinaryQuadraticModel.from_qubo(QUBO)
    QUBO_EMBEDDING = map_embedding_to_QUBO(QUBO, complete_embedding)
    chain_strength_fixed = embedding.chain_strength.uniform_torque_compensation(
        bqm, prefactor=UTC_PREFACTOR)
    embedded_qubo = embedding.embed_qubo(
        QUBO, QUBO_EMBEDDING, connectivity_graph, chain_strength=chain_strength_fixed)
    while True:
        try:
            sampleset = DWave_solver.sample_qubo(embedded_qubo, answer_mode='raw', **params)
            vectors = sampleset.samples
            break
        except:
            print("D-Wave connection failed, trying again in 1 second...")
            time.sleep(1)
            continue
    unembedded = majority_vote(vectors, QUBO_EMBEDDING)
    bcols = get_bcols_from_samples(unembedded, RANK)
    solved_bcol = column_solve_postprocess(bcols, x, A)
    bcol_solution_dict = {list(As.keys())[0]: solved_bcol}
    return bcol_solution_dict


def batch_quantum_annealing(X, N, A, B, random_state=42):
    """
    Submits one column factorization problem to D-Wave at a time;
    i.e. each column factorization problem is solved sequentially.

    Parameters
    ----------
    X : 2-d numpy array
        matrix to be factored.
    N : int
        column index.
    A : 2-d numpy array
        Initial state.
    B : 2-d numpy array
        Initial state. Not used. Here for the logical consistency.
    random_state : int, optional
        random state. The default is 42.

    Returns
    -------
    out : list
        list of (b) columns.

    """
    random.seed(random_state)

    QUBO_storage = []
    connectivity_graph, DWave_solver, solver_name = Start_DWave_connection()
    complete_embedding = read_complete_embedding()
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
        b_columns_solved = quantum_annealing(
            top_As, top_xs, top_QUBOs, connectivity_graph, DWave_solver, complete_embedding)
        QUBO_storage = combine_QUBO_storage(QUBO_storage, top_QUBOs, b_columns_solved)
        QUBO_storage = remove_duplicate_QUBO(QUBO_storage)
        results = {**results, **b_columns_solved}
    out = []
    for i in range(N):
        out.append(results[i])
    return out
