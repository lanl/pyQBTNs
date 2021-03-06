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


"""Utility methods."""

import random
import json
import numpy as np
import networkx as nx
import dimod
from dwave.cloud import Client
from symengine import var

def Start_DWave_connection():
    """
    Creates a connection to a D-Wave quantum annealer using the defualt information in the user's
    D-Wave config file.

    Returns
    -------
    connectivity_graph : networkx.Graph()
        Undirected hardware connectivity map of the quantum annealer.
    DWave_solver : DWave solver object
        D-Wave solver object. You can pass problems to this object in order to submit them to be
        solved on the quantum annealer.
    solver_name : str
        Name of the quantum annealer.

    """
    client = Client.from_config()
    solver_name = client.default_solver
    DWave_solver = client.get_solver(list(solver_name.values())[0])
    A = DWave_solver.undirected_edges
    connectivity_graph = nx.Graph(list(A))
    return connectivity_graph, DWave_solver, list(solver_name.values())[0]

####################################
# QUBO methods
####################################
def get_T_F_vecs(xcol):
    T = []
    F = []
    for (i, data) in enumerate(xcol):
        if data == True:
            T.append(i)
        if data == False:
            F.append(i)
    return T, F
def get_QUBO(A, x):
	true_vec, false_vec = get_T_F_vecs(x)
	variables = ""
	for idx in range(len(x)):
		variables += "x"+str(idx)+" "
	all_symbol_variables = list(var(variables))
	true_polynomial = 0
	for col_index in true_vec:
		row = A[col_index]
		used_variables = []
		for (idx, value) in enumerate(row):
			if value == True:
				used_variables.append(idx)
		if len(used_variables) == 0:
			continue
		symbols = [all_symbol_variables[i] for i in used_variables]
		combined = 1
		for symb in symbols:
			combined = combined*(1-symb)
		combined = 1-combined
		try:
			combined = combined.expand()
		except:
			pass
		combined = -1*combined
		true_polynomial += combined
	false_polynomial = 0
	for col_index in false_vec:
		row = A[col_index]
		used_variables = []
		for (idx, value) in enumerate(row):
			if value == True:
				used_variables.append(idx)
		if len(used_variables) == 0:
			continue
		symbols = [all_symbol_variables[i] for i in used_variables]
		combined = 1
		for symb in symbols:
			combined = combined*(1-symb)
		combined = 1-combined
		try:
			combined = combined.expand()
		except:
			pass
		false_polynomial += combined
	polynomial = true_polynomial + false_polynomial
	try:
		polynomial = polynomial.expand()
	except:
		return "EMPTY"
	HUBO_dictionary = polynomial.as_coefficients_dict()
	HUBO = {}
	for k in HUBO_dictionary:
		try:
			if type(int(k)) is int:
				continue
		except:
			term = str(k)
			term = term.replace("x", "")
			if "*" not in term:
				HUBO[(int(term),)] = HUBO_dictionary[k]
			else:
				terms = list(term.split("*"))
				terms = tuple([int(a) for a in terms])
				HUBO[terms] = HUBO_dictionary[k]
	if len(HUBO) == 0:
		return "EMPTY"
	coefs = [abs(a) for a in list(HUBO.values())]
	HUBO_TO_QUBO_PENALTY_FACTOR = max(coefs)
	QUBO = dimod.make_quadratic(HUBO, HUBO_TO_QUBO_PENALTY_FACTOR, dimod.BINARY).to_qubo()[0]
	return QUBO

def read_complete_embedding():
    """
    reads in the all-to-all connectivity embedding that has been strored as a Json.

    Returns
    -------
    complete_embedding : dictionary
        Keys are logical variables numbereed 0-N. Values are lists of physical qubits that represent
        the logical state of the that variable (that variable being the key to that value).

    """
    CLIQUE = 65
    _, _, solver_name = Start_DWave_connection()
    file = open("../data/fixed_embeddings/"+solver_name+"_" +
                str(CLIQUE)+"_node_complete_embedding.json", "r")
    complete_embedding = json.load(file)
    file.close()
    return complete_embedding


def read_rank3_parallel_QA_embeddings(random_state=42):
    """


    Parameters
    ----------
    random_state : int, optional
        random seed. The default is 42.

    Returns
    -------
    rank_3_embeddings : list
        List of disjoint clique-4 embeddings covering most of the quantum annealing hardware
        connectivity graph.

    """
    random.seed(random_state)
    small_clique_embeddings = 4
    _, _, solver_name = Start_DWave_connection()
    file = open("../data/fixed_embeddings/"+solver_name+"_size_" +
                str(small_clique_embeddings)+"_clique_parallel_QA_embeddings.json", "r")
    rank_3_embeddings = json.load(file)
    random.shuffle(rank_3_embeddings)
    file.close()
    return rank_3_embeddings


def get_hamming_distance(M1, M2):
    """
    Gets the number of dis-similar entries betweeen two numpy arrays.
    Here the application is for boolean numpy array, whichis why this metric
    is actually hamming distance. Since this comparison is element-wise, the
    number of entries in each array needs to be equal.

    Parameters
    ----------
    M1 : numpy.array
        Boolean numpy array of arbitrary dimensions. Could be a 1-d, 2-d
        array (matrix) or higher order (tensor).
    M2 : numpy.array
        Boolean numpy array of arbitrary dimensions. Could be a 1-d, 2-d
        array (matrix) or higher order (tensor).

    Returns
    -------
    ham : int
        Number of unequal elements between the two boolean numpy arrays.

    """

    ham = 0
    vec1 = list(M1.ravel())
    vec2 = list(M2.ravel())
    assert len(vec1) == len(vec2)
    for i in range(len(vec1)):
        if vec1[i] != vec2[i]:
            ham += 1
    return ham


def majority_vote(vectors, problem_embedding, random_state=42):
    """
    Unembeds raw D-Wave samples using the majority vote function.
    Unbiased majority vote in that if the chain is split 50-50, then random
    choice is used.

    Parameters
    ----------
    vectors : list
        Raw vectors from the D-Wave solver. The length is equal
        to the number of anneals. Each element in vectors is a list of
        length equal to the number of qubits on thee D-Wave device. For the
        case of a D-Wave 2000Q, the number of qubits  is 2048 (including active and inactive).
    problem_embedding : dict
        Logical embedding for the problem that D-Wave solved. Keys are
        variable names, and values are a list of physical qubits representing
        the logical state of the variable (key).

    Returns
    -------
    all_vectors_unembedded : list
        List of dictionaries. The number of dictionaries is equal to
        the length of thee input variable vectors. Each dictionary is
        has the same keys as problem_embedding, and the values
        are either 0 1 since this function handles QUBOS.

    """
    random.seed(random_state)

    all_vectors_unembedded = []
    for v in vectors:
        unembedded = {}
        for chain in problem_embedding:
            count_1 = 0
            count_0 = 0
            for qubit in problem_embedding[chain]:
                if v[qubit] == 0:
                    count_0 += 1
                if v[qubit] == 1:
                    count_1 += 1
            if count_0 == count_1:
                unembedded[chain] = random.choice([0, 1])
            if count_0 > count_1:
                unembedded[chain] = 0
            if count_1 > count_0:
                unembedded[chain] = 1
        all_vectors_unembedded.append(unembedded)
    return all_vectors_unembedded


def delete_keys_from_dict(dictionary, keys_to_remove):
    """
    Utility to remove specific keys from a dictionary.


    Parameters
    ----------
    dictionary : dict
        Input dictionary.
    keys_to_remove : list
        list of keys to remove from the dictionary. If this list is
        non-unique (i.e. has repeats), then there will be an error. Also, if
        any element in this list is not a key in the dictionary, there will
        be an error as well.

    Returns
    -------
    dictionary : dict
        Dictionary with keys removed.

    """
    for a in keys_to_remove:
        del dictionary[a]
    return dictionary



def get_fixed_embedding(QUBO, complete_embedding, random_state=42):
    """
    Given an input of a QUBO and an embedding, this function maps the variables from
    the qubo onto the embedding.

    Parameters
    ----------
    QUBO : dict
        dictionary where the keys are linear or quadratic terms, and the values are real numbers.
        Represents a Quadratic Unconstrained Binary Optimization problem.
    complete_embedding : dict
        all-to-all connectivity embedding for the given QUBO.
    random_state : integer, optional
        random seed parameter. The default is 42.

    Returns
    -------
    QUBO_embedding : dict
        remapped embedding for QUBO.

    """
    random.seed(random_state)

    linear_variables = []
    quadratic_variables = []
    for a in QUBO:
        if a[0] == a[1]:
            linear_variables.append(a[0])
        else:
            quadratic_variables.append(a)
    QUBO_embedding = {}
    complete_vars = list(complete_embedding.keys())
    random.shuffle(complete_vars)
    for i in range(len(linear_variables)):
        QUBO_embedding[linear_variables[i]] = complete_embedding[complete_vars[i]]
    return QUBO_embedding

def column_solve_postprocess(b_cols, xcol, A):
    """
    Solving x_col = A*b_col for b_col
    This method chooses the best b_col out of however many post-processed solutions were returned by the probabilistic sampler.

    Parameters
    ----------
    b_cols : list
        List of b-column solutions found by the probabilistic sampler.
    xcol : list
        target x-column we want to get the factorization of.
    A : 2-d Boolean numpy array (matrix)
        The A in x=Ab.

    Returns
    -------
    selection : list
        b-column in the form of a list.

    """
    hamming_distances = []
    solutions = []
    for b_solved in b_cols:
        b_solved = np.transpose(np.array(b_solved, dtype=bool))
        solved_x = np.matmul(A, b_solved)
        ham = get_hamming_distance(solved_x, xcol)
        hamming_distances.append(ham)
        solutions.append([b_solved.tolist(), ham])
    minimum = min(hamming_distances)
    good_solns = []  # There might be many possible solutions
    for a in solutions:
        if a[1] == minimum:
            good_solns.append(a[0])
    selection = random.choice(good_solns)
    return selection


def get_bcols_from_samples(vectors, rank):
    """
    From the solved vectors, compute the corresponding b-columns

    Parameters
    ----------
    vectors : list
        post-processed vectors.
    rank : int
        rank of the column factorization problem.

    Returns
    -------
    bcols : list
        List of solutions found by the probabilistic sampler, in the form of b-columns.

    """
    bcols = []
    for sample in vectors:
        vec = []
        for i in range(rank):
            try:
                vec.append(sample[i])
            except:#Empty QUBO case
                vec.append(random.choice([0, 1]))
        bcols.append(vec)
    return bcols


def combine_QUBO_storage(QUBO_storage, solved_QUBOs, column_solutions):
    """
    Merges previous QUBO storage with new solutions.

    Parameters
    ----------
    QUBO_storage : list
        Input QUBO storage list.
    solved_QUBOs : dict
        The QUBOs that were solved in this latest iteration.
    column_solutions : dict
        The associated b-column solutions for each of the solved QUBOs.

    Returns
    -------
    QUBO_storage : list
        Updated QUBO storage with more QUBOs and their recorded solutions.

    """
    assert set(list(column_solutions.keys())) == set(
        list(solved_QUBOs.keys())), "Something has gone wrong"
    indices = list(solved_QUBOs.keys())
    for i in indices:
        QUBO_storage.append([solved_QUBOs[i], column_solutions[i]])
    return QUBO_storage


def filter_out_stored_QUBOs(QUBO_storage, all_QUBOS, all_xcols, all_Amatrices):
    """
    Removes QUBOs which have been solved previously (as tracked by QUBO_storage).
    The best solutions that have been stored are then used instead of calling the solver again on the same QUBO.

    Parameters
    ----------
    QUBO_storage : list
        Stored QUBOs (and their solutions) that have already been solved.
    all_QUBOS : dict
        QUBOs to be (potentially) factored in this iteration.
    all_xcols : dict
        The associated x-columns for each of these QUBOs to be solved.
    all_Amatrices : dict
        The associated A matrices for each of these QUBOs to be solved..

    Returns
    -------
    storage_solutions : dict
        Any solved QUBOs that we do not need to solve again.
    all_QUBOS : dict
        Updated QUBOs to solve.
    all_xcols : dict
        Updated x-columns dictionary.
    all_Amatrices : dict
        Updated A-columns dictionary.

    """
    storage_solutions = {}
    to_remove = []
    for a in all_QUBOS:
        QUBO = all_QUBOS[a]
        for var_pair in QUBO_storage:
            if var_pair[0] == QUBO:
                to_remove.append(a)
                storage_solutions[a] = var_pair[1]
    to_remove = list(set(to_remove))
    all_QUBOS = delete_keys_from_dict(all_QUBOS, to_remove)
    all_xcols = delete_keys_from_dict(all_xcols, to_remove)
    all_Amatrices = delete_keys_from_dict(all_Amatrices, to_remove)
    return storage_solutions, all_QUBOS, all_xcols, all_Amatrices


def map_embedding_to_QUBO(QUBO, complete_embedding):
    """
    For each variable in QUBO, this function maps that variable to a set of physical qubits
    found in complete_embedding.

    Parameters
    ----------
    QUBO : dict
        keys are linear or quadratic terms, and values are the weights associated
        with each of those terms.
    complete_embedding : dict
        Large all-to-all embedding for the quantum annealing hardware.

    Returns
    -------
    out : dict
        re-mapped embedding .

    """
    vars = []
    for a in QUBO:
        vars.append(a[0])
        vars.append(a[1])
    vars = list(set(vars))
    out = {}
    c = -1
    for a in complete_embedding:
        if c == len(vars)-1:
            break
        c += 1
        out[vars[c]] = complete_embedding[a]
    return out


def remove_duplicate_QUBO(QUBO_storage):
    """
    De-duplicates the QUBO storage list

    Parameters
    ----------
    QUBO_storage : list
        List of QUBOs (each QUBO is a dictionary).

    Returns
    -------
    unique : list
        de-duplicated list.

    """
    unique = []
    for a in QUBO_storage:
        if a not in unique:
            unique.append(a)
    return unique


def remove_values_from_list(input_list, targ):
    """
    Remove the value targ from the list

    Parameters
    ----------
    input_list : list
        list to remove a value from.
    targ : usually int, float or str
        target value to be removed.

    Returns
    -------
    list
        List with the target value removed.

    """
    return [value for value in input_list if value != targ]
