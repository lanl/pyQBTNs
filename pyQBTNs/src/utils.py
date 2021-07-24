import random
import json
import numpy as np
import networkx as nx
import dimod
from sympy import symbols, expand
from dwave.cloud import Client


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
    solver_name : string
        Name of the quantum annealer.

    """
    client = Client.from_config()
    solver_name = client.default_solver
    DWave_solver = client.get_solver(list(solver_name.values())[0])
    A = DWave_solver.undirected_edges
    connectivity_graph = nx.Graph(list(A))
    return connectivity_graph, DWave_solver, list(solver_name.values())[0]


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
    random_state : integer, optional
        random seed. The default is 42.

    Returns
    -------
    rank_3_embeddings : Lisst
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
    ham : integer
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
    vectors : List
        Raw vectors from the D-Wave solver. The length is equal
        to the number of anneals. Each element in vectors is a list of
        length equal to the number of qubits on thee D-Wave device. For the
        case of a D-Wave 2000Q, the number of qubits  is 2048 (including active and inactive).
    problem_embedding : Dict
        Logical embedding for the problem that D-Wave solved. Keys are
        variable names, and values are a list of physical qubits representing
        the logical state of the variable (key).

    Returns
    -------
    all_vectors_unembedded : List
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


def qubo(vars):
    """
    expands out the polynomial so we can extract the coefficients for each variable

    Parameters
    ----------
    vars : List
        List of sympy Symbols().

    Returns
    -------
    result : Sympy expression
        QUBO that has been expanded.

    """
    combined = 1
    for i in vars:
        combined = combined*(1-i)
    result = expand(1-combined)
    return result


def get_T_F_vecs(col):
    """
    Input of a single column vector, returns the variable indices for both
    True and False values in the vector.

    Parameters
    ----------
    col : List or numpy array
        Input column vector.

    Returns
    -------
    T : List
        Variable indicies for True variable state.
    F : List
        Variable indicies for True variable state.

    """
    T = []
    F = []
    for (i, data) in enumerate(col):
        if data == 1:
            T.append(i)
        if data == 0:
            F.append(i)
    return T, F


def get_polynomial(A, V, indicator):
    """
    returns Sympy polynomial. This polynmial is a
    HUBO (Higher order Unconstrained Binary Optimization) problem.

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    V : TYPE
        DESCRIPTION.
    indicator : Boolean
        A boolean variable for storing what coefficient we need in front of the polynomial
        A is the other factor of X (or approximate factor of X) we are using.

    Returns
    -------
    all_polynomials : TYPE
        DESCRIPTION.

    """
    if indicator == 1:
        prefix = -1
    else:
        prefix = 1
    all_polynomials = 0
    for j in V:
        if len(V) == 0:
            break
        vec = A[j]  # Gets the jth row of A
        c = -1
        vars = ""
        for value in vec:
            c += 1
            if value == 1:
                vars += "x"+str(c)+" "
        if vars == "":
            continue
        variables = symbols(vars)
        if type(variables) is tuple:
            tmp_poly = prefix*qubo(list(variables))
        else:
            tmp_poly = prefix*qubo([variables])
        all_polynomials += tmp_poly
    return all_polynomials


def get_fixed_embedding(QUBO, complete_embedding, random_state=42):
    """
    Given an input of a QUBO and an embedding, this function maps the variables from
    the qubo onto the embedding.

    Parameters
    ----------
    QUBO : dictionary
        dictionary where the keys are linear or quadratic terms, and the values are real numbers.
        Represents a Quadratic Unconstrained Binary Optimization problem.
    complete_embedding : dictionary
        all-to-all connectivity embedding for the given QUBO.
    random_state : integer, optional
        random seed parameter. The default is 42.

    Returns
    -------
    QUBO_embedding : dictionary
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


def get_qubo(col, A, bcol_len, random_state=42):
    """
    Given an input of a column factorization problem, i.e. x=Ab where x and b are vectors
    and A is amtrix, we  want to find b given x and A.

    Parameters
    ----------
    col : list or numpy array
        x-column in the problem x=Ab.
    A : numpy array
        matrix A in x=Ab.
    bcol_len : integer
        expected length of the b-column solution vector. Also equal to rank.
    random_state : integer, optional
        random state. The default is 42.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    random.seed(random_state)

    T, F = get_T_F_vecs(col)
    all_polynomials = 0
    all_polynomials += get_polynomial(A, T, 1)
    all_polynomials += get_polynomial(A, F, 0)
    if type(all_polynomials) is int:  # In this condition, the state does not matter do we return random values
        tmp = []
        for i in range(bcol_len):
            tmp.append(random.choice([0, 1]))
        return "NA"
    HUBO = all_polynomials.as_coefficients_dict()  # Error here means we need to just use a random state
    HUBO_dict = {}
    for a in HUBO:  # Convert polynomial to dictionary format we want
        term = ()
        s = str(a).split("*")
        if len(s) == 1:
            tmp = int(s[0].strip("x"))
            term = (tmp,)
        else:
            for i in s:
                tmp = int(i.strip("x"))
                term += (tmp,)
        HUBO_dict[term] = float(HUBO[a])
    coefs = [abs(a) for a in list(HUBO_dict.values())]
    HUBO_TO_QUBO_PENALTY_FACTOR = max(coefs)
    Q = dimod.make_quadratic(HUBO_dict, HUBO_TO_QUBO_PENALTY_FACTOR, dimod.BINARY).to_qubo()[0]
    return Q


def column_solve_postprocess(b_cols, xcol, A):
    """
    Solving x_col = A*b_col for b_col
    This function calls DWave and then selects a column that minimizes error
    A and B are the initial conditions

    Parameters
    ----------
    b_cols : TYPE
        DESCRIPTION.
    xcol : TYPE
        DESCRIPTION.
    A : TYPE
        DESCRIPTION.

    Returns
    -------
    selection : TYPE
        DESCRIPTION.

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


    Parameters
    ----------
    vectors : TYPE
        DESCRIPTION.
    rank : integer
        rank of the column factorization problem.

    Returns
    -------
    bcols : TYPE
        DESCRIPTION.

    """
    bcols = []
    for sample in vectors:
        vec = []
        for i in range(rank):
            try:
                vec.append(sample[i])
            except:
                vec.append(random.choice([0, 1]))
        bcols.append(vec)
    return bcols


def combine_QUBO_storage(QUBO_storage, solved_QUBOs, column_solutions):
    """


    Parameters
    ----------
    QUBO_storage : TYPE
        DESCRIPTION.
    solved_QUBOs : TYPE
        DESCRIPTION.
    column_solutions : TYPE
        DESCRIPTION.

    Returns
    -------
    QUBO_storage : TYPE
        DESCRIPTION.

    """
    assert set(list(column_solutions.keys())) == set(
        list(solved_QUBOs.keys())), "Something has gone wrong"
    indices = list(solved_QUBOs.keys())
    for i in indices:
        QUBO_storage.append([solved_QUBOs[i], column_solutions[i]])
    return QUBO_storage


def filter_out_stored_QUBOs(QUBO_storage, all_QUBOS, all_xcols, all_Amatrices):
    """


    Parameters
    ----------
    QUBO_storage : TYPE
        DESCRIPTION.
    all_QUBOS : dictionary
        DESCRIPTION.
    all_xcols : dictionary
        DESCRIPTION.
    all_Amatrices : dictionary
        DESCRIPTION.

    Returns
    -------
    storage_solutions : TYPE
        DESCRIPTION.
    all_QUBOS : dictionary
        DESCRIPTION.
    all_xcols : dictionary
        DESCRIPTION.
    all_Amatrices : dictionary
        DESCRIPTION.

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
    QUBO : dictionary
        keys are linear or quadratic terms, and values are the weights associated
        with each of those terms.
    complete_embedding : dictionary
        Large all-to-all embedding for the quantum annealing hardware.

    Returns
    -------
    out : dictionary
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
    targ : usually integer, float or string
        target value to be removed.

    Returns
    -------
    list
        List with the target value removed.

    """
    return [value for value in input_list if value != targ]
