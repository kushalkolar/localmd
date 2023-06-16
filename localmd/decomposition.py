import jax
import jax.scipy
import jax.numpy as jnp
from jax import jit, vmap
import functools
from functools import partial
import time
import scipy
import scipy.sparse

import jaxopt
import numpy as np

from localmd.evaluation import spatial_roughness_stat_vmap, temporal_roughness_stat_vmap, construct_final_fitness_decision


import sys
import datetime
import os


import os
import pathlib
import sys
import math
import tifffile

from tqdm import tqdm

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
import functools
from functools import partial
import torch
import torch_sparse
import torch.multiprocessing as multiprocessing

import scipy.sparse

from localmd.preprocessing_utils import get_noise_estimate_vmap, center_and_get_noise_estimate
from localmd.tiff_loader import tiff_loader


import time
import datetime
import sys


def display(msg):
    """
    Printing utility that logs time and flushes.
    """
    tag = '[' + datetime.datetime.today().strftime('%y-%m-%d %H:%M:%S') + ']: '
    sys.stdout.write(tag + msg + '\n')
    sys.stdout.flush()


@partial(jit)
def objective_function(X, placeholder, data):
    num_rows = data.shape[0]
    difference = data.shape[1]

    comp1 = jax.lax.dynamic_slice(X, (0, 0), (num_rows, X.shape[1]))
    comp2 = jax.lax.dynamic_slice(X, (num_rows, 0), (difference, X.shape[1]))
    prod = jnp.matmul(comp1, comp2.T)
    return jnp.linalg.norm(data - prod)

#Some observations here: the projected gradient step is significantly slower than unconstrained gradient 
# and it not necessary for us to actually use 
@partial(jit)
def rank_k_fit(data, orig_placeholder):

    shape_1 = data.shape[0]
    shape_2 = data.shape[1]
    init_param = jnp.zeros((shape_1 + shape_2, orig_placeholder.shape[0])) + 1
    solver = ProjectedGradient(fun=objective_function,
                             projection=projection.projection_non_negative,
                             tol=1e-6, maxiter=1000)
    fit_val = solver.run(init_param, placeholder=orig_placeholder, data=data).params

    return fit_val


rank_k_fit_vmap = jit(vmap(rank_k_fit, in_axes=(2, None)))



@partial(jit)
def unconstrained_rank_fit(data, orig_placeholder):

    shape_1 = data.shape[0]
    shape_2 = data.shape[1]
    init_param = jnp.zeros((shape_1 + shape_2, orig_placeholder.shape[0])) + 1

    solver = jaxopt.GradientDescent(fun=objective_function, maxiter=1000)
    params, state = solver.run(init_param, placeholder=orig_placeholder, data=data)

    return params

@partial(jit)
def add_ith_column(mat, vector, i):
    '''
    Jit-able function for adding a vector to a specific column of a matrix (i-th column)
    Inputs: 
        mat: jnp.array, size (d, T)
        vector: jnp.array, size (1, T)
        i: integer between 0 and T-1 inclusive. Denotes which column of mat you'd like to add "vector" to
        
    Returns: 
        mat: jnp.array, size (d, T). 
    '''
    col_range = jnp.arange(mat.shape[1])
    col_filter = col_range == i
    col_filter = jnp.expand_dims(col_filter, axis=0)
    
    dummy_mat = jnp.zeros_like(mat)
    dummy_mat = dummy_mat + vector 
    dummy_mat = dummy_mat * col_filter
    mat = mat + dummy_mat
    
    return mat
    
    
@partial(jit)
def rank_1_deflation_pytree(i, input_pytree):
    '''
    Computes a rank-1 decomposition of residual and appends the results to u and v
    Inputs:
        i: integer indicating which column of our existing data to update with our rank-1 decomposition
            (this is clarified below, look at the parameter definitions for u and v)
        input_pytree. a list python object containing the following jnp.arrays (in order):
            residual: jnp.array. Shape (d, T)
            u: jnp.array. Shape (d, K) for some K that is not relevant to this functionality. We append the column vector of
            our rank-1 decomposition to column k of u.
            v: jnp.array. Shape (K, T). We append the row vector of
                our rank-1 decomposition to row k of v.
            
    Outputs:
        cumulative_results. Pytree (python list) containing (residual, u, v), where the residual has been updated
            (i.e. we subtracted the rank-1 estimate from this procedure) and u and v have been updated as well 
                (i.e. we append the rank-1 estimate from this procedure to the i-th column/row of u/v respectively).

    
    '''
    residual = input_pytree[0]
    u = input_pytree[1]
    v = input_pytree[2]
    
    #Step 1: Get a rank-1 fit for the data
    placeholder = jnp.zeros((1, 1))
    approximation = unconstrained_rank_fit(residual, placeholder) #Output here will be (d + T, 1)-shaped jnp array
    u_k = jax.lax.dynamic_slice(approximation, (0,0), (residual.shape[0], 1))
    v_k = jax.lax.dynamic_slice(approximation, (residual.shape[0], 0), (residual.shape[1], 1))
    
    # v_k = jnp.dot(residual.T, u_k) #This is the debias/rescaling step
    
    new_residual = residual - jnp.dot(u_k, v_k.T)
    
    u_new = add_ith_column(u, u_k, i)
    v_new = add_ith_column(v.T, v_k, i).T
    
    return [new_residual, u_new, v_new]    
    
@partial(jit)
def iterative_rank_1_approx(test_data):
    num_iters = 25
    u_mat = jnp.zeros((test_data.shape[0], num_iters))
    v_mat = jnp.zeros((num_iters, test_data.shape[1]))
    i = 0
    data_pytree = [test_data, u_mat, v_mat]
    final_pytree = jax.lax.fori_loop(0, num_iters, rank_1_deflation_pytree, data_pytree)
    
    return final_pytree


@partial(jit, static_argnums=(2,3))
def truncated_random_svd(input_matrix, key, rank, num_oversamples=10):
    '''
    Key: This function assumes that (1) rank + num_oversamples is less than all dimensions of the input_matrix and (2) num_oversmples >= 1
    
    '''
    d = input_matrix.shape[0]
    T = input_matrix.shape[1]
    random_data = jax.random.normal(key, (T, rank + num_oversamples))
    projected = jnp.matmul(input_matrix, random_data)
    Q, R = jnp.linalg.qr(projected)
    B = jnp.matmul(Q.T, input_matrix)
    U, s, V = jnp.linalg.svd(B, full_matrices=False)
    
    U_final = Q.dot(U)
    V = jnp.multiply(jnp.expand_dims(s, axis=1), V)
    
    #Final step: prune the rank 
    U_truncated = jax.lax.dynamic_slice(U_final, (0, 0), (U_final.shape[0], rank))
    V_truncated = jax.lax.dynamic_slice(V, (0, 0), (rank, V.shape[1]))
    return [U_truncated, V_truncated]




@partial(jit)
def iterative_rank_1_approx_sims(test_data):
    num_iters = 3
    u_mat = jnp.zeros((test_data.shape[0], num_iters))
    v_mat = jnp.zeros((num_iters, test_data.shape[1]))
    i = 0
    data_pytree = [test_data, u_mat, v_mat]
    final_pytree = jax.lax.fori_loop(0, num_iters, rank_1_deflation_pytree, data_pytree)
    
    return final_pytree



@partial(jit, static_argnums=(2,))
def decomposition_no_normalize_approx(block, key, max_rank):
    d1, d2, T = block.shape
    block_2d = jnp.reshape(block, (d1*d2, T), order="F")
    decomposition = truncated_random_svd(block_2d, key, max_rank)
    
    u_mat, v_mat = decomposition[0], decomposition[1]
    u_mat = jnp.reshape(u_mat, (d1, d2, u_mat.shape[1]), order="F")
    
    spatial_statistics = spatial_roughness_stat_vmap(u_mat)
    temporal_statistics = temporal_roughness_stat_vmap(v_mat)

    return spatial_statistics, temporal_statistics

# decomposition_no_normalize_approx_vmap = jit(vmap(decomposition_no_normalize_approx, in_axes = (3, 2)))


@partial(jit)
def decomposition_no_normalize(block):
    d1, d2, T = block.shape
    block_2d = jnp.reshape(block, (d1*d2, T), order="F")
    decomposition = iterative_rank_1_approx_sims(block_2d)
    
    u_mat, v_mat = decomposition[1], decomposition[2]
    u_mat = jnp.reshape(u_mat, (d1, d2, u_mat.shape[1]), order="F")
    
    spatial_statistics = spatial_roughness_stat_vmap(u_mat)
    temporal_statistics = temporal_roughness_stat_vmap(v_mat)

    return spatial_statistics, temporal_statistics

decomposition_no_normalize_vmap = jit(vmap(decomposition_no_normalize, in_axes = (3)))



@partial(jit, static_argnums=(0,1,2,3))
def make_matrix(d1, d2, T, num_comps, key):
    noise_data = jax.random.normal(key, (d1, d2, T))
    random_projection = jax.random.normal(key, (T, num_comps))
    
    return noise_data, random_projection
  
@partial(jit, static_argnums=(0,1,2,3))
def rank_simulation(d1, d2, T, num_comps, key1, key2):
    # noise_data, random_projection = make_matrix(d1, d2, T, num_comps, key)
    noise_data = jax.random.normal(key1, (d1, d2, T))
    spatial, temporal = decomposition_no_normalize_approx(noise_data, key2, num_comps)
    return spatial, temporal

def make_jax_random_key():
    ii32 = np.iinfo(np.int32)
    prng_input = np.random.randint(low=ii32.min, high=ii32.max,size=1, dtype=np.int32)[0]
    key = jax.random.PRNGKey(prng_input)
    
    return key

def threshold_heuristic(dimensions, num_comps=1, iters = 250, percentile_threshold = 5):
    spatial_list = []
    temporal_list = []
    
    d1, d2, T = dimensions
    for k in range(iters):
        key1 = make_jax_random_key()
        key2 = make_jax_random_key()
        x, y = rank_simulation(d1, d2, T, num_comps, key1, key2)
        spatial_list.append(x)
        temporal_list.append(y)

    spatial_thres = np.percentile(np.array(spatial_list).flatten(), percentile_threshold)
    temporal_thres = np.percentile(np.array(temporal_list).flatten(), percentile_threshold)
    return spatial_thres, temporal_thres

@jit
def filter_and_decompose(block,mean_img, std_img,spatial_basis, projection_data,  spatial_thres, temporal_thres, max_consec_failures):
    '''
    Inputs: 
    block: jnp.ndarray. Dimensions (block_1, block_2, T). (block_1, block_2) are the dimensions of this patch of data, T is the number of frames.
    mean_img: jnp.ndarray. Dimensions (block_1, block_2). Mean image of this block (over entire dataset, not just the "T" frames this block contains). 
    std_img: jnp.ndarray. Dimensions (block_1, block_2). Nosie variance image of this block (over the entire dataset, not just the "T" frames this block contains). 
    spatial_basis: jnp.ndarray. Dimensions (block_1, block_2, svd_dim). Here, svd_dim is the dimension of the whole FOV svd we perform before doing the localized SVD on each spatial patch. 
    projection_data: jnp.ndarray. Dimensions (T, max_dimension). Used for the fast approximate SVD method.
    spatial_thres: float. Threshold used to determine whether an estimated spatial component from the SVD is just noise or contains useful signal
    temporal_thres: float. Threshold used to determine whether an estimated temporal component from the SVD is just noise or not.
    max_consec_failures: int, usually 1. After doing the truncated SVD, we iterate over the components, from most to least significant, and 
    '''
    
    ##Step 1: Standardize the data
    block -= mean_img[:, :, None]
    block /= std_img[:, :, None]
    
    #Step 2: Get the temporal basis for the full FOV decomposition: 
    # temporal_basis = jnp.tensordot(jnp.transpose(spatial_basis, axes=(2,0,1)), block, axes=((1,2), (0,1)))
    # block = block - jnp.tensordot(spatial_basis, temporal_basis, axes=((2), (0)))
    
    return single_block_md(block, projection_data, spatial_thres, temporal_thres, max_consec_failures)
    
@partial(jit, static_argnums=(2,))
def single_block_md(block, key, max_rank, spatial_thres, temporal_thres, max_consec_failures):
    '''
    Matrix Decomposition function for all blocks. 
    Inputs: 
        block: jnp.array. Dimensions (block_1, block_2, T). (block_1, block_2) are the dimensions of this patch of data, T is the number of frames. We assume that this data has already been centered and noise-normalized
        projection_data: jnp.array. Dimensions (T, max_dimension). Used for the fast approximate SVD method.
        
    '''
    #TODO: Get rid of max consec failures entirely from function API 
    # block = standardize_block(block) #Center and divide by noise standard deviation before doing matrix decomposition
    d1, d2, T = block.shape
    block_2d = jnp.reshape(block, (d1*d2, T), order="F")
    
    
    decomposition = truncated_random_svd(block_2d, key, max_rank)
    u_mat, v_mat = decomposition[0], decomposition[1]
    u_mat = jnp.reshape(u_mat, (d1, d2, u_mat.shape[1]), order="F")

    
#     ##Now we begin the evaluation phase
    good_comps = construct_final_fitness_decision(u_mat, v_mat.T, spatial_thres,\
                                                  temporal_thres, max_consec_failures)
    
    return u_mat, good_comps, v_mat


def append_decomposition_results(curr_results, new_results):
    '''
    Each results list has 3 ndarrays: 
        results[0]: The set of spatial components. Dimensions (# of blocks, d1*d2, num_comps per block). d1,d2 are block dimensions.
        results[1]: The decisions for whether or not to accept/reject the components. Dimensions (#num blocks, num_comps per block)
        results[2]: The set of temporal components for each block. Dimensions (# of blocks, num_comps, T) where T = number of frames in dataset.
    '''
    curr_results[0] = np.concatenate([curr_results[0], new_results[0][None, :, :]], axis=0)
    curr_results[1] = np.concatenate([curr_results[1], new_results[1][None, :]], axis=0)
    curr_results[2] = np.concatenate([curr_results[2], new_results[2][None, :, :]], axis=0)
    
    return curr_results

def cast_decomposition_results(new_results, cast):
    return (np.array(new_results[0], dtype=cast), np.array(new_results[1], dtype=cast), np.array(new_results[2], dtype=cast))



def get_projector(U):
    #TODO: Use Pytorch_sparse to accelerate the matmul on GPU/TPU... (may be overkill)
    '''
    Input: 
        U: matrix of dimensions (d, R) where d is number of pixels, R is number of frames
    Returns: 
    Tuple (final_matrix_r_sparse, projector)
        final_matrix_r_sparse: a sparse version of U. Type scipy.sparse
        projector: the inverse term used to do a linear subspace projection of a vector onto U. I.e.
        
        Proj_U (X) = U (U^TU)-1 U^T X --> projector is (U^TU)^-1
    '''
    final_matrix_r_sparse = scipy.sparse.coo_matrix(U)
    prod = (final_matrix_r_sparse.T.dot(final_matrix_r_sparse)).toarray()
    theta = np.array(jnp.linalg.inv(prod))
    # projector = (final_matrix_r_sparse.dot(theta.T)).T #Do it like this to take advantage of sparsity using scipy not np
    # return projector
    return (final_matrix_r_sparse.T, theta)



 
def factored_svd(spatial_components, temporal_components, device='cpu', factor=1):
    '''
    Given a matrix factorization M=UQ (with U sparse) factorizes Q = RSVt so
    that [UR]SVt is the SVD of M.
    
    KEY: The product, spatial_components * temporal_components, should be a matrix whose rows have mean 0. 
    This is important because when we produce the factorized SVD, we can then do another round of PCA-like dimensionality reduction (the singular values of this SVD directly gives us the eigenvalues of the correlation matrix)
    
    KEY: Everything needs to be float64 for numerical stability 
    
    Inputs: 
        spatial_components: scipy.sparse matrix
        temporal_components: np.ndarray
        
    Note: float is significantly (i.e. 1 order of magnitude) faster here, that should be the default input
    '''
    spatial_components_sparse = torch_sparse.tensor.from_scipy(spatial_components).to(device).double()
    temporal_components_torch = torch.from_numpy(temporal_components).to(device).double()
    Qt, Lt = torch.linalg.qr(temporal_components_torch.t(), mode='reduced')
    # Step 2: Fast Transformed Spatial Inner Product Sigma = L'U'UL
    Sigma = torch_sparse.matmul(spatial_components_sparse.t(), spatial_components_sparse).to_dense()
    Sigma = torch.matmul(Lt, torch.matmul(Sigma, Lt.t()))
    Sigma = (Sigma + Sigma.t()) / 2 #Trick to enforce symmetry of the matrix (so that next step works as intended, independent of rounding errors from above) 
    
    eig_vals, eig_vecs = torch.linalg.eigh(Sigma)  # Note: eig vals/vecs ascending
    eig_vecs = torch.flip(eig_vecs, dims=(1,))
    eig_vals = torch.flip(eig_vals, dims=(0,))
    
    eig_vec_norms = torch.linalg.norm(eig_vecs, dim=0)
    selected_indices = torch.nonzero((eig_vec_norms > 0) * (eig_vals > 0)).squeeze()
    
    eig_vecs = torch.index_select(eig_vecs, 1, selected_indices)
    eig_vals = torch.index_select(eig_vals, 0, selected_indices) 
    
    singular_values = torch.sqrt(eig_vals)  # Note: now vals descending
    
    # Step 4: Apply Eigen Vectors Such That (UR, V) Are Singular Vectors
    mixing_weights = torch.matmul(Lt.t(), eig_vecs) / singular_values[None, :]
    temporal_basis = torch.matmul(eig_vecs.t(), Qt.t())
    
    #Here we prune the factorized SVD 
    return rank_prune_svd(mixing_weights, singular_values, temporal_basis, factor=factor)


def rank_prune_svd(mixing_weights, singular_values, temporal_basis, factor=0.25):
    '''
    Inputs: 
        mixing_weights: torch.Tensor, shape (R x R)
        singular_values: torch.Tensor, shape (R)
        temporal_basis: torch.Tensor, shape (R, T)
        explained_variance: float between 0 and 1. The fraction of explained variance which we would like to explain. 
    '''
    
    dimension = singular_values.shape[0]
    index = int(math.floor(factor * dimension))
    if index == 0:
        pass
    elif index > singular_values.shape[0]:
        pass
    else:
        mixing_weights = mixing_weights[:, :index]
        singular_values = singular_values[:index]
        temporal_basis = temporal_basis[:index, :]
    display("The rank was originally {} now it is {}".format(dimension, mixing_weights.shape[1]))
    return mixing_weights, singular_values, temporal_basis
    


def rank_prune_svd_variance(mixing_weights, singular_values, temporal_basis, explained_variance_threshold = 0.995):
    '''
    Inputs: 
        mixing_weights: torch.Tensor, shape (R x R)
        singular_values: torch.Tensor, shape (R)
        temporal_basis: torch.Tensor, shape (R, T)
        explained_variance: float between 0 and 1. The fraction of explained variance which we would like to explain. 
    '''
    device = mixing_weights.device
    mixing_weights = mixing_weights
    singular_values = singular_values
    temporal_basis = temporal_basis
    
    singular_values_normalized = singular_values / torch.amax(singular_values) #We assume no divide by zero here
    squared_singular_values = singular_values_normalized * singular_values_normalized
    total_featurewise_variance = torch.sum(squared_singular_values)
    if total_featurewise_variance > 0:
        squared_singular_values /= total_featurewise_variance
    squared_singular_values_cumulative = torch.cumsum(squared_singular_values, dim=0)
    above_threshold = squared_singular_values_cumulative > explained_variance_threshold
    
    nonzero_above_threshold = torch.nonzero(above_threshold)
    if nonzero_above_threshold.numel() == 0:
        display("This threshold was too high")
        return mixing_weights, singular_values, temporal_basis
    ## Add a check here to verify that torch nonzero is actually good
    critical_index = torch.min(nonzero_above_threshold)
    
    
    if torch.index_select(squared_singular_values_cumulative, 0, critical_index) <= explained_variance_threshold:
        display("Warning: for unknown reasons the index did not meet threshold, potential bug") 
        return mixing_weights, singular_values, temporal_basis
    else:
        display("Rank Pruning has been applied. The rank was {}, now it is {}. We have pruned {:.2f} of the components".format(mixing_weights.shape[1], critical_index+1, 1 - (critical_index+1) / mixing_weights.shape[1]))
        keep_indices = torch.arange(critical_index + 1, device=device)
        mixing_weights = torch.index_select(mixing_weights, 1, keep_indices) 
        singular_values = torch.index_select(singular_values, 0, keep_indices)
        temporal_basis = torch.index_select(temporal_basis, 0, keep_indices)
        
    return mixing_weights, singular_values, temporal_basis

def localmd_decomposition(filename, block_sizes, overlap, frame_range, max_components=50, background_rank=15, sim_conf=5, batching=10, tiff_batch_size = 10000, dtype='float32', order="F", num_workers=0, pixel_batch_size=5000, frame_corrector_obj = None, max_consec_failures = 1, rank_prune_factor=1):
    
    if torch.cuda.is_available():
            device='cuda'
    else:
            device='cpu'
            
    load_obj = tiff_loader(filename, dtype=dtype, center=True, normalize=True, background_rank=background_rank, batch_size=tiff_batch_size, order=order, num_workers=num_workers, pixel_batch_size=pixel_batch_size, frame_corrector_obj = frame_corrector_obj)
    start = frame_range[0]
    end = frame_range[1]
    end = min(end, load_obj.shape[2])
    frames = [i for i in range(start, end)]
    block_sizes = block_sizes
    overlap = overlap
    
    ##Step 2a: Get the spatial and temporal thresholds
    display("Running Simulations")
    spatial_thres, temporal_thres = threshold_heuristic([block_sizes[0], block_sizes[1], len(frames)], num_comps = 1, iters=250, percentile_threshold=sim_conf)
    
    ##Step 2b: Load the data you will do blockwise SVD on
    display("Loading Data")
    data = load_obj.temporal_crop_with_filter([i for i in range(start, end)])
    data_std_img = load_obj.std_img #(d1, d2) shape
    data_mean_img = load_obj.mean_img #(d1, d2) shape
    data_spatial_basis = load_obj.spatial_basis.reshape((load_obj.shape[0], load_obj.shape[1], -1), order=load_obj.order)
    
    ##Step 2c: Run PMD and get the U matrix components
    display("Obtaining blocks and running local SVD")
    cumulator = []

    start_t = time.time()

    pairs = []
    
    cumulator_count = 0

    dim_1_iters = list(range(0, data.shape[0] - block_sizes[0] + 1, block_sizes[0] - overlap[0]))
    if dim_1_iters[-1] != data.shape[0] - block_sizes[0] and data.shape[0] - block_sizes[0] != 0:
        dim_1_iters.append(data.shape[0] - block_sizes[0])

    dim_2_iters = list(range(0, data.shape[1] - block_sizes[1] + 1, block_sizes[1] - overlap[1]))
    if dim_2_iters[-1] != data.shape[1] - block_sizes[1] and data.shape[1] - block_sizes[1] != 0:
        dim_2_iters.append(data.shape[1] - block_sizes[1])


    #Define the block weighting matrix
    block_weights = np.ones((block_sizes[0], block_sizes[1]), dtype=dtype)
    hbh = block_sizes[0] // 2
    hbw = block_sizes[1] // 2
    # Increase weights to value block centers more than edges
    block_weights[:hbh, :hbw] += np.minimum(
        np.tile(np.arange(0, hbw), (hbh, 1)),
        np.tile(np.arange(0, hbh), (hbw, 1)).T
    )
    block_weights[:hbh, hbw:] = np.fliplr(block_weights[:hbh, :hbw])
    block_weights[hbh:, :] = np.flipud(block_weights[:hbh, :])
        
    sparse_indices = np.arange(data.shape[0]*data.shape[1]).reshape((data.shape[0], data.shape[1]), order=order)
    row_number = 0
    column_indices = []
    row_indices = []
    spatial_overall_values = []
    cumulative_weights = np.zeros((data.shape[0], data.shape[1]))
    
    for k in dim_1_iters:
        for j in dim_2_iters:
            pairs.append((k, j))
            subset = data[k:k+block_sizes[0], j:j+block_sizes[1], :].astype(dtype)
            key = make_jax_random_key()
            spatial_comps, decisions, _ = single_block_md(subset, key, max_components, spatial_thres, temporal_thres, max_consec_failures)

            spatial_comps = np.array(spatial_comps).astype(dtype)
            dim_1_val = k
            dim_2_val = j
            
            decisions = np.array(decisions).flatten() > 0
            spatial_cropped = spatial_comps[:, :, decisions]
            
            #Weight the spatial components here
            spatial_cropped = spatial_cropped * block_weights[:, :, None]
            current_cumulative_weight = block_weights * spatial_cropped.shape[2]
            cumulative_weights[k:k+block_sizes[0], j:j+block_sizes[1]] += current_cumulative_weight
            
            sparse_col_indices = sparse_indices[k:k+block_sizes[0], j:j+block_sizes[1]][:, :, None]
            sparse_col_indices = sparse_col_indices + np.zeros((1, 1, spatial_cropped.shape[2]))
            sparse_row_indices = np.zeros_like(sparse_col_indices)
            addend = np.arange(row_number, row_number+spatial_cropped.shape[2])[None, None, :]
            
            sparse_row_indices = sparse_row_indices + addend
            sparse_col_indices_f = sparse_col_indices.flatten().tolist()
            sparse_row_indices_f = sparse_row_indices.flatten().tolist()
            spatial_values_f = spatial_cropped.flatten().tolist()
            
            column_indices.extend(sparse_col_indices_f)
            row_indices.extend(sparse_row_indices_f)
            spatial_overall_values.extend(spatial_values_f)
            row_number += spatial_cropped.shape[2]
            
    
    U_r = scipy.sparse.coo_matrix((spatial_overall_values, (column_indices, row_indices)), shape=(data.shape[0]*data.shape[1], row_number))
    
    display("Normalizing by weights")
    weight_normalization_diag = np.zeros((data.shape[0]*data.shape[1],))
    weight_normalization_diag[sparse_indices.flatten()] = cumulative_weights.flatten()
    normalizing_weights = scipy.sparse.diags(
        [(1 / weight_normalization_diag).ravel()], [0])
    U_r = normalizing_weights.dot(U_r)
    
    display("The total number of identified components before pruning is {}".format(U_r.shape[1]))
    
    
    
    display("Computing projector for sparse regression step")
    projector = get_projector(U_r)

    ## Step 2f: Do sparse regression to get the V matrix: 
    display("Running sparse regression")
    V = load_obj.V_projection(projector)

    ## Step 2g: Aggregate the global SVD with the localMD results to create the final decomposition
    display("Aggregating Global SVD with localMD results")
    U_r, V = aggregate_decomposition(U_r, V, load_obj)
    
    U_r = U_r.astype(dtype)
    V = V.astype(dtype)

    ## Step 2h: Do a SVD Reformat given U and V
    display("Running QR decomposition on V")
    R, s, Vt = factored_svd(U_r, V, device='cpu', factor = rank_prune_factor)

    display("Matrix decomposition completed")

    return U_r, R.cpu().numpy(), s.cpu().numpy(), Vt.cpu().numpy(), load_obj


def aggregate_decomposition(U_r, V, load_obj):
    
    if load_obj.background_rank == 0:
        pass
    else:
        spatial_bg = load_obj.spatial_basis
        temporal_bg = load_obj.temporal_basis
        spatial_bg_sparse = scipy.sparse.coo_matrix(spatial_bg)
        U_r = scipy.sparse.hstack([U_r, spatial_bg_sparse])
        V = np.concatenate([V, temporal_bg], axis = 0)
    
    return U_r, V