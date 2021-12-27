"""Boolean Matrix Factorization."""
import random
import numpy as np
from nimfa.methods.seeding.nndsvd import Nndsvd
from .quantum_solve import batch_parallel_quantum_annealing, batch_quantum_annealing
from .classical_solve import batch_classical_single_QUBO
from .utils import get_hamming_distance


class Matrix_Factorization():
    """
    Matrix Factorization
    """

    def __init__(self, solver_method, NNSVD_INITIAL_STATE_FLAG=0, maximum_iterations=500, maximum_converged_iterations=100, number_of_random_initial_states=100, random_initial_state_lower_bound=0.01, random_initial_state_upper_bound=0.99, parallel_bool=False, random_state=42):
        """


        Parameters
        ----------
        solver_method : string
            Solver method. Set to "d-wave" in order to use a quantum annealing backend.
        NNSVD_INITIAL_STATE_FLAG : integer, optional
            Passeed to the NNSVD initial state generator. The default is 0.
        maximum_iterations : integer, optional
            Absolute upper bound on the number of iterations for each full matrix factorization sub-routine. The default is 500.
        maximum_converged_iterations : integer, optional
            Secondary termination criteria for the iterative matrix factorization sub-routine. Terminates the algorithm if the error rate has converged to the same error rate (hamming distance). The default is 100.
        number_of_random_initial_states : integer, optional
            Numbere of random initial states to try. The default is 100.
        random_initial_state_lower_bound : float, optional
            Lower bound for uniform random proportion for generating random initial states for the matrix factorization. The default is 0.01.
        random_initial_state_upper_bound : float, optional
            Upper bound for uniform random proportion for generating random initial states for the matrix factorization. The default is 0.99.
        parallel_bool : Boolean, optional
            Set to True in order to use parallel quantum annealing. False in order to not use parallel QA. The default is False.
        random_state : integer, optional
            random state. The default is 42.

        Returns
        -------
        None.

        """
        self.solver_method = solver_method
        self.NNSVD_INITIAL_STATE_FLAG = NNSVD_INITIAL_STATE_FLAG
        self.maximum_iterations = maximum_iterations
        self.maximum_converged_iterations = maximum_converged_iterations
        self.number_of_random_initial_states = number_of_random_initial_states
        self.random_initial_state_lower_bound = random_initial_state_lower_bound
        self.random_initial_state_upper_bound = random_initial_state_upper_bound
        self.parallel_bool = parallel_bool
        self.random_state = random_state

    def factor_matrix(self, X, RANK, A, B):
        """


        Parameters
        ----------
        X : 2-dimensional boolean numpy array
            Boolean matrix to be factored.
        RANK : integer
            factorization rank.
        A : 2-dimensional boolean numpy array
            Initial state A.
        B : 2-dimensional boolean numpy array
            Initial state B.

        Returns
        -------
        A : 2-dimensional boolean numpy array
            Best found factor (A) for X=AB.
        B : 2-dimensional boolean numpy array
            Best foudn factor (B) for X=AB.
        bool
            True if exact factorization was found. False otherwise
        error_tracking : List
            List of hamming distances for each pair of factor matrices (A, B) found.
        error_tracking_data : List
            Same as error_tracking, but also includes the pairs of matrices A and B.

        """
        iteration = 0
        N1 = X.shape[0]  # rows
        N2 = X.shape[1]  # columns
        error_tracking = []
        error_tracking_data = []
        prod = np.matmul(A, B)
        err = get_hamming_distance(prod, X)
        error_tracking.append(err)
        error_tracking_data.append([err, A, B])
        if np.array_equal(prod, X):
            return A, B, True, error_tracking, error_tracking_data
        while error_tracking[self.maximum_converged_iterations:].count(min(error_tracking)) <= self.maximum_converged_iterations:
            iteration += 1
            if iteration > self.maximum_iterations:
                break  # Goes to the return at the end of the function
            if self.parallel_bool is True and self.solver_method == "d-wave":
                out = batch_parallel_quantum_annealing(X, N2, A, B, random_state=self.random_state)

            elif self.parallel_bool is False and self.solver_method == "d-wave":
                out = batch_quantum_annealing(X, N2, A, B, random_state=self.random_state)

            elif self.solver_method != "d-wave":
                out = batch_classical_single_QUBO(
                    X, N2, A, B, self.solver_method, random_state=self.random_state)

            B_p = np.array(out)
            B_p = np.transpose(B_p)
            B_p = np.array(B_p, dtype=bool)
            B = B_p
            tmp = np.matmul(A, B)
            assert len(A.shape) == 2  # Sanity checks that we are still dealing with 2-D matrices
            assert len(B.shape) == 2
            assert len(X.shape) == 2
            assert len(tmp.shape) == 2
            err = get_hamming_distance(tmp, X)
            error_tracking.append(err)
            error_tracking_data.append([err, A, B])
            if np.array_equal(tmp, X) is True:
                return A, B, True, error_tracking, error_tracking_data
            iteration += 1
            if iteration > self.maximum_iterations:
                break  # Goes to the return at the end of the function
            A = np.transpose(A)
            B = np.transpose(B)
            X = np.transpose(X)

            #######################################################

            if self.parallel_bool is True and self.solver_method == "d-wave":
                out = batch_parallel_quantum_annealing(X, N1, B, A)

            elif self.parallel_bool is False and self.solver_method == "d-wave":
                out = batch_quantum_annealing(X, N1, B, A)

            elif self.solver_method != "d-wave":
                out = batch_classical_single_QUBO(
                    X, N1, B, A, self.solver_method, random_state=self.random_state)

            A_p = np.array(out)
            A_p = np.transpose(A_p)
            A_p = np.array(A_p, dtype=bool)
            A = A_p
            tmp = np.matmul(B, A)
            assert len(A.shape) == 2, "Something went wrong"
            assert len(B.shape) == 2, "Something went wrong"
            assert len(X.shape) == 2, "Something went wrong"
            assert len(tmp.shape) == 2, "Something went wrong"
            err = get_hamming_distance(tmp, X)
            error_tracking.append(err)
            error_tracking_data.append([err, np.transpose(A), np.transpose(B)])
            if np.array_equal(tmp, X) is True:
                return np.transpose(A), np.transpose(B), True, error_tracking, error_tracking_data
            A = np.transpose(A)
            B = np.transpose(B)
            X = np.transpose(X)
        return A, B, False, error_tracking, error_tracking_data

    def train(self, X, RANK):
        """
        Factor the input matrix into A and B given X in the problem X=AB.

        Parameters
        ----------
        T : numpy array
            Tensor to be factored.
        RANK : integer
            factorization rank.

        Returns
        -------
        A : numpy array
            First matrix factor.
        B : numpy array
            Second matrix factor.

        """
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        N1 = X.shape[0]  # rows
        N2 = X.shape[1]  # columns
        data = []
        hammings = []
        get_init_state = Nndsvd()
        A_init, B_init = get_init_state.initialize(
            X, rank=RANK, options={"flag": self.NNSVD_INITIAL_STATE_FLAG})
        A_init = np.array(A_init.round(), dtype=bool)
        B_init = np.array(B_init.round(), dtype=bool)
        A, B, indicator, error_tracking, error_tracking_data = self.factor_matrix(
            X, RANK, A_init, B_init)
        if indicator:
            return A, B
        hammings += error_tracking
        data += error_tracking_data
        for rep in range(self.number_of_random_initial_states):
            p = random.uniform(self.random_initial_state_lower_bound,
                               self.random_initial_state_upper_bound)

            A_init = np.random.choice(a=[False, True], size=(N1, RANK), p=[p, 1-p])
            A_init = np.array(A_init, dtype=bool)
            B_init = np.random.choice(a=[False, True], size=(RANK, N2), p=[p, 1-p])
            B_init = np.array(B_init, dtype=bool)
            A, B, indicator, error_tracking, error_tracking_data = self.factor_matrix(
                X, RANK, A_init, B_init)

            if indicator:
                return A, B
            hammings += error_tracking
            data += error_tracking_data
        # Now trying the best pair found for more iterations
        minimum_hamming = min(hammings)
        possible_choices = []
        for i in data:
            if i[0] == minimum_hamming:
                possible_choices.append([i[1], i[2]])
        choice = random.choice(possible_choices)
        A_init = choice[0]
        B_init = choice[1]
        A, B, indicator, error_tracking, error_tracking_data = self.factor_matrix(
            X, RANK, A_init, B_init)

        if indicator:
            return A, B
        hammings += error_tracking
        data += error_tracking_data
        # If we reach this point, we did not find an exact factorization
        # Therefore, we now select a random A, B pair that has a minimum hamming distance
        minimum_hamming = min(hammings)
        possible_choices = []
        for i in data:
            if i[0] == minimum_hamming:
                possible_choices.append([i[1], i[2]])
        choice = random.choice(possible_choices)
        A = choice[0]
        B = choice[1]
        return A, B
