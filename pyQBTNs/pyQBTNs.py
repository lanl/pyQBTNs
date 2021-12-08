"""
This software was developed as a tool to factor tensors using quantum annealers.
Right now this software includes 5 different tensor factorization methods, making up three distinct types of tensor networks.

The software allows the user to specify local solvers that do not require a connection to a quantum annealer, but still solve the optimization problems the annealer would solve during the factorization algorithm.

In order to use a D-Wave quantum annealer as the solver for this software, the user must set up a D-Wave configuration file.
"""

from .src.Tensor_Train_Recursive import Tensor_Train_Recursive
from .src.Tensor_Train_Iterative import Tensor_Train_Iterative
from .src.Hierarchical_Tucker import Hierarchical_Tucker
from .src.Tucker_Recursive import Tucker_Recursive
from .src.Tucker_Iterative import Tucker_Iterative
from .src.Matrix_Factorization import Matrix_Factorization
from .src.utils import get_hamming_distance, Start_DWave_connection
from .src.tensor_utils import reconstruct_TT, reconstruct_HT, reconstruct_tucker
from .src.generate_fixed_embeddings import iterative_embedding
import glob
import numpy as np
import json
import networkx as nx
from dwave import embedding


class QBTNs():

    def __init__(self, **parameters):
        """
        Initilize the QBTNs class.

        Parameters
        ----------
        factorization_method : str
            Options are : 'Matrix_Factorization', 'Tensor_Train_Recursive', 'Tensor_Train_Iterative', 'Hierarchical_Tucker', 'Tucker_Recursive', 'Tucker_Iterative'.

        solver_method : str, optional
            Options are : "d-wave", "classical-simulated-annealing", "classical-steepest-descent", "classsical-tabu-sampler". Default is "classical-simulated-annealing"

        NNSVD_INITIAL_STATE_FLAG : integer, optional
            Passeed to the NNSVD initial state generator. The default is 0.

        maximum_iterations : integer, optional
            Absolute upper bound on the number of iterations for each full matrix factorization sub-routine. The default is 500.

        maximum_converged_iterations : integer, optional
            Secondary termination criteria for the iterative matrix factorization sub-routine. Terminates the algorithm if the error rate has converged to the same error rate (hamming distance).
            Default is 100.

        number_of_random_initial_states : integer, optional
            Numbere of random initial states to try. The default is 100.

        random_initial_state_lower_bound : float, optional
            Lower bound for uniform random proportion for generating random initial states for the matrix factorization. The default is 0.01.

        random_initial_state_upper_bound : float, optional
            Upper bound for uniform random proportion for generating random initial states for the matrix factorization. The default is 0.99.

        parallel_bool : Boolean, optional
            Set to True in order to use parallel quantum annealing. False in order to not use parallel QA. The default is False. 
            This effectively means that we solve independent QUBOs using the same call to a solver.

        random_state : int, optional
            Integer to set random seeds in the algorithm.


        """

        allowed_factorization_methods = ['Matrix_Factorization', 'Tensor_Train_Recursive', 'Tensor_Train_Iterative',
                                         'Hierarchical_Tucker', 'Tucker_Recursive', 'Tucker_Iterative']
        # Compute method
        assert 'factorization_method' in parameters, "You must specify a factorization method. Please choose from: %s" % str(
            ','.join(allowed_factorization_methods))

        self.method = parameters['factorization_method']
        del parameters['factorization_method']

        assert self.method in allowed_factorization_methods, "Unknown factorization method. Please choose from: %s" % str(
            ','.join(allowed_factorization_methods))

        allowed_solver_methods = ["d-wave", "classical-simulated-annealing",
                                  "classical-steepest-descent", "classsical-tabu-sampler"]
        if "solver_method" in parameters:
            assert parameters["solver_method"] in allowed_solver_methods, "Unknown solver method. Please choose from: %s" % str(
                ','.join(allowed_solver_methods))
        else:
            parameters["solver_method"] = "classical-simulated-annealing"

        if "random_state" in parameters:
            self.random_state = parameters["random_state"]
        else:
            self.random_state = 42

        # Set the specified model
        if self.method == "Tensor_Train_Recursive":
            self.model = Tensor_Train_Recursive(**parameters)

        elif self.method == "Tensor_Train_Iterative":
            self.model = Tensor_Train_Iterative(**parameters)

        elif self.method == "Hierarchical_Tucker":
            self.model = Hierarchical_Tucker(**parameters)

        elif self.method == "Tucker_Recursive":
            self.model = Tucker_Recursive(**parameters)

        elif self.method == "Tucker_Iterative":
            self.model = Tucker_Iterative(**parameters)

        elif self.method == "Matrix_Factorization":
            self.model = Matrix_Factorization(**parameters)

        if parameters["solver_method"] == "d-wave":
            self.__generate_embeddings()

        # class variables
        self.score = None
        self.latent_factors = None
        self.reconstructed_tensor = None

    def fit(self, Tensor, Rank, **parameters):
        """
        Computes the factors and score for a given factorization method of the input Tensor with the specified rank.

        Parameters
        ----------
        Tensor : numpy.array(dtype=bool)
            Boolean numpy array with at least two dimensions.
        Rank : int
            Rank of the factors. Rank >= 2.
            The size of the quantum annealing hardware limits the size of the rank.
            For the LANL D-Wave 2000Q, the safe limit is rank 8,
            although in some cases much higher rank factorization can be achieved.

        dimensions : list, optional
            Optional argument which supplies the tensor dimensions. Required for Hierarchical tucker because the input is a dict of an HT structure

        ranks : list, optional
            Optional argument which supplies the tensor factorization ranks. Required for Hierarchical tucker


        Returns
        -------
        None.

        """
        assert len(Tensor.shape) > 1, "Tensor must be at least 2 dimensional"
        assert Rank > 1, "Rank must be greater than 1"
        assert isinstance(Rank, int), "Rank must be an integer"

        if self.method == "Matrix_Factorization":
            self.latent_factors = self.model.train(Tensor, Rank)
        elif self.method != "Hierarchical_Tucker":
            dimensions = list(Tensor.shape)
            ranks = [Rank for i in range(len(dimensions))]
            self.latent_factors = self.model.train(Tensor, dimensions, ranks)
        elif self.method == "Hierarchical_Tucker":
            assert 'dimensions' in parameters, "dimensions must be specified for HT"
            assert 'ranks' in parameters, "ranks must be specified for HT"
            assert isinstance(parameters['dimensions'], list), "dimensions must be a list"
            assert isinstance(parameters['ranks'], list), "ranks must be a list"

            self.latent_factors = self.model.train(Tensor, parameters['dimensions'], parameters['ranks'])

        Tensor_prime = self.get_reconstructed_tensor()
        self.score = get_hamming_distance(Tensor, Tensor_prime)

    def get_score(self):
        """
        Returns the hamming distance of the fitted factors to the original tensor

        Returns
        -------
        integer
            Hamming distance between the input tensor and the factors found by the given algorithm.
            The smaller the hamming distance is, the more accurate the factorization process was.

        """

        assert self.latent_factors is not None, "Fit the tensor first using pyQBTNs.QBTNs.fit()."

        return self.score

    def get_factors(self):
        """
        Returns factors computed by the factorization algorithm.

        Returns
        -------
        list or dict or tuple
            returns some sort of data structure containing the computed factors.
            Each factorization algorithm returns slightly different formats for the factors.

        """

        assert self.latent_factors is not None, "Fit the tensor first using pyQBTNs.QBTNs.fit()."

        return self.latent_factors

    def get_reconstructed_tensor(self):
        """
        Gets the boolean numpy array consstructed from the factors found by
        the tensor factorization algorithm.

        Returns
        -------
        numpy.array(dtype=bool)
            Boolean numpy array that has been recontructed from the computed factors.

        """

        assert self.latent_factors is not None, "Fit the tensor first using pyQBTNs.QBTNs.fit()."

        if self.method == "Tensor_Train_Recursive":
            self.reconstructed_tensor = reconstruct_TT(self.latent_factors)

        elif self.method == "Tensor_Train_Iterative":
            self.reconstructed_tensor = reconstruct_TT(self.latent_factors)

        elif self.method == "Hierarchical_Tucker":
            self.reconstructed_tensor = reconstruct_HT(self.latent_factors)

        elif self.method == "Tucker_Recursive":
            self.reconstructed_tensor = reconstruct_tucker(
                self.latent_factors[0], self.latent_factors[1])

        elif self.method == "Tucker_Iterative":
            self.reconstructed_tensor = reconstruct_tucker(
                self.latent_factors[0], self.latent_factors[1])

        elif self.method == "Matrix_Factorization":
            self.reconstructed_tensor = np.matmul(self.latent_factors[0], self.latent_factors[1])

        return self.reconstructed_tensor

    def __generate_embeddings(self):
        """
        Generates all fixed embeddings that might be needed if the solver method is D-Wave.
        The method only generates new embeddings if there is not already jsson embedding files
        located at data/fixed_embeddings/ for the defualt QPU solver in the user's D-Wave
        configuration file.

        Returns
        -------
        None.

        """
        CLIQUE = 65
        small_clique_embeddings = 4
        connectivity_graph, _, solver_name = Start_DWave_connection()
        embedding_files = glob.glob("../data/fixed_embeddings/*.json")
        if "../data/fixed_embeddings/"+solver_name+"_"+str(CLIQUE)+"_node_complete_embedding.json" not in embedding_files:
            print("Generating fixed embedding for a clique of size "+str(CLIQUE) +
                  " for solver "+solver_name+". This will take some time.")
            complete_embedding = embedding.minorminer.find_embedding(
                nx.complete_graph(CLIQUE), connectivity_graph, random_seed=self.random_state)
            file = open("../data/fixed_embeddings/"+solver_name+"_" +
                        str(CLIQUE)+"_node_complete_embedding.json", "w")
            json.dump(complete_embedding, file)
            file.close()

        if "../data/fixed_embeddings/"+solver_name+"_size_"+str(small_clique_embeddings)+"_clique_parallel_QA_embeddings.json" not in embedding_files:
            print("Generating fixed embeddings for the parallel quantum annealing method for solver " +
                  solver_name+". This will take some time.")
            subgraphs = iterative_embedding(connectivity_graph, small_clique_embeddings)
            file = open("../data/fixed_embeddings/"+solver_name+"_size_" +
                        str(small_clique_embeddings)+"_clique_parallel_QA_embeddings.json", "w")
            json.dump(subgraphs, file)
            file.close()
