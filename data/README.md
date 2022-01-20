## Data for reproducibility

`test_tensors/` contains boolean tensors in `.npy` format.

`factorization_results/` contains the results from factoring the test boolean tensors. The results include the computation time used (i.e. QPU time in the case of using the quantum annealer backend), as well as the computed factors of the tensor.

The naming convention used for the tensor files is `orderX_nY_rZ_T_index.npy`, where `X` is the Order of the tensor, `Y` is the size of each dimension of the tensor, `Z` is the rank (used to generate the tensor), `index` is the unique index of the tensor for it's dimension's and rank. For each Order, Size, and rank, there are 5 randomly generated boolean tensors; meaning that `index` ranges from `0-4`
