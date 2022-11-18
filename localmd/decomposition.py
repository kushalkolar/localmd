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


#@partial(jit)
def truncated_random_svd(input_matrix, random_data):
    desired_rank = random_data.shape[1]
    projected = jnp.matmul(input_matrix, random_data)
    Q, R = jnp.linalg.qr(projected)
    B = jnp.matmul(Q.T, input_matrix)
    U, s, V = jnp.linalg.svd(B, full_matrices=False)
    
    U_final = Q.dot(U)
    V = jnp.multiply(jnp.expand_dims(s, axis=1), V)
    return [U_final, V]



@partial(jit)
def iterative_rank_1_approx_sims(test_data):
    num_iters = 3
    u_mat = jnp.zeros((test_data.shape[0], num_iters))
    v_mat = jnp.zeros((num_iters, test_data.shape[1]))
    i = 0
    data_pytree = [test_data, u_mat, v_mat]
    final_pytree = jax.lax.fori_loop(0, num_iters, rank_1_deflation_pytree, data_pytree)
    
    return final_pytree



@partial(jit)
def decomposition_no_normalize_approx(block, random_projection_mat):
    d1, d2, T = block.shape
    block_2d = jnp.reshape(block, (d1*d2, T), order="F")
    decomposition = truncated_random_svd(block_2d, random_projection_mat)
    
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


class random_projection_dataset():
    def __init__(self, iters, dims, num_comps):
        if iters <= 0:
            raise ValueError("need a nonnegative number of iterations")
        self.iters = iters
        self.d1, self.d2, self.T = dims
        self.num_comps = num_comps
        
        
        
    def __len__(self):
        return self.iters
    
    
    def __getitem__(self, index):
        noise_data = np.random.randn(self.d1, self.d2, self.T)
        random_projection = np.random.randn(self.T, self.num_comps)
        return (noise_data, random_projection)
  
def regular_collate(batch):
    return batch[0]

def threshold_heuristic(block_sizes, num_comps = 3, iters=50, percentile_threshold = 5, num_workers=None):
    '''
    We simulate the roughness statistics for components when we fit to standard normal noise
    Inputs: 
        block_sizes: tuple, dimensions of block: d1, d2, T, where d1 and d2 are the dimensions of the block on the FOV and T is the window length
        num_comps: positive integer. Number of components we fit for each (0,1)-random noise dataset
        iters: default parameter, int. Number of times we do this procedure on the GPU. This param is really to avoid memorry blowups on the GPU
        num_sims: default parameter, int. 
    Outputs: 
        spatial_thresh, temporal_thresh. The spatial and temporal statistics
    '''
    d1, d2, T = block_sizes
    spatial_cumulator = np.zeros((0,))
    temporal_cumulator = np.zeros((0, ))
    
    random_dataobj = random_projection_dataset(iters, (d1,d2,T), num_comps)

    if num_workers is None:
        num_cpu = multiprocessing.cpu_count()
        num_workers = min(len(random_dataobj), num_cpu-1)
        num_workers = max(num_workers, 0)
    loader = torch.utils.data.DataLoader(random_dataobj, batch_size=1,
                                             shuffle=False, num_workers=num_workers, collate_fn=regular_collate, timeout=0)
    for i, data in enumerate(tqdm(loader), 0):
        noise_data, random_projection = data

        spatial_temp, temporal_temp = decomposition_no_normalize_approx(noise_data, random_projection)

        spatial_cumulator = np.concatenate([spatial_cumulator, spatial_temp])
        temporal_cumulator = np.concatenate([temporal_cumulator, temporal_temp])

    spatial_thres = np.percentile(spatial_cumulator.flatten(), percentile_threshold)
    temporal_thres = np.percentile(temporal_cumulator.flatten(), percentile_threshold)
    
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
    temporal_basis = jnp.tensordot(jnp.transpose(spatial_basis, axes=(2,0,1)), block, axes=((1,2), (0,1)))
    block = block - jnp.tensordot(spatial_basis, temporal_basis, axes=((2), (0)))
    
    return single_block_md(block, projection_data, spatial_thres, temporal_thres, max_consec_failures)
    
# @partial(jit)
def single_block_md(block, projection_data, spatial_thres, temporal_thres, max_consec_failures):
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
    
    
    
    decomposition = truncated_random_svd(block_2d, projection_data)
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
    final_matrix_r_sparse = scipy.sparse.coo_matrix(U)
    prod = (final_matrix_r_sparse.T.dot(final_matrix_r_sparse)).toarray()
    theta = np.linalg.inv(prod)
    projector = (final_matrix_r_sparse.dot(theta.T)).T #Do it like this to take advantage of sparsity using scipy not np
    return projector



def factored_svd(spatial_components, temporal_components):
    """
    Given a matrix factorization M=UQ (with U sparse) factorizes Q = RSVt so
    that [UR]SVt is the SVD of M.
    TODO: Accelerate using jax or torch for GPU
    """

    # Step 1: Othogonalize Temporal Components LQ = V
    Qt, Lt = np.linalg.qr(temporal_components.T)

    # Step 2: Fast Transformed Spatial Inner Product Sigma = L'U'UL
    Sigma = np.asarray(spatial_components.T.dot(spatial_components).todense())
    Sigma = np.dot(Lt, np.dot(Sigma, Lt.T))

    # Step 3: Eigen Decomposition Of Sigma
    eig_vals, eig_vecs = np.linalg.eigh(Sigma)  # Note: eig vals/vecs ascending
    eig_vecs = eig_vecs[:, ::-1]  # Note: now vecs descending
    singular_values = np.sqrt(eig_vals[::-1])  # Note: now vals descending

    # Step 4: Apply Eigen Vectors Such That (UR, V) Are Singular Vectors
    mixing_weights = Lt.T.dot(eig_vecs) / singular_values[None, :]
    temporal_basis = eig_vecs.T.dot(Qt.T)

    return mixing_weights, singular_values, temporal_basis  # R, s, Vt


def localmd_decomposition(filename, block_sizes, overlap, frame_range, max_components=50, background_rank=15, sim_conf=5, batching=10, tiff_batch_size = 10000, dtype='float32', order="F", num_workers=0, pixel_batch_size=5000, frame_corrector_obj = None):
    load_obj = tiff_loader(filename, dtype=dtype, center=True, normalize=True, background_rank=background_rank, batch_size=tiff_batch_size, order=order, num_workers=num_workers, pixel_batch_size=pixel_batch_size, frame_corrector_obj = frame_corrector_obj)
    start = frame_range[0]
    end = frame_range[1]
    end = min(end, load_obj.shape[2])
    frames = [i for i in range(start, end)]
    block_sizes = block_sizes
    overlap = overlap
    
    ##Step 2a: Get the spatial and temporal thresholds
    display("Running Simulations")
    spatial_thres, temporal_thres = threshold_heuristic([block_sizes[0], block_sizes[1], len(frames)], num_comps = 1, iters=250, percentile_threshold=sim_conf, num_workers=num_workers)
    
    ##Step 2b: Load the data you will do blockwise SVD on
    display("Loading Data")
    data = load_obj.temporal_crop(frames)
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
    for k in dim_1_iters:
        for j in dim_2_iters:
            pairs.append((k, j))
            subset = data[k:k+block_sizes[0], j:j+block_sizes[1], :].astype(dtype)
            projected_data = np.random.randn(subset.shape[2], max_components).astype(dtype)
            crop_mean_img = data_mean_img[k:k+block_sizes[0], j:j+block_sizes[1]]
            crop_std_img = data_std_img[k:k+block_sizes[0], j:j+block_sizes[1]]
            crop_spatial_basis = data_spatial_basis[k:k+block_sizes[0], j:j+block_sizes[1], :]
            spatial_comps, decisions, _ = filter_and_decompose(subset, crop_mean_img, crop_std_img, crop_spatial_basis, projected_data, spatial_thres, temporal_thres, 1)
            # spatial_comps, decisions, _ = single_block_md(subset, projected_data, spatial_thres, temporal_thres, 1)
            spatial_comps = spatial_comps.astype(dtype)

            dim_1_val = k
            dim_2_val = j

            decisions = decisions.flatten()

            spatial_cropped = spatial_comps[:, :, decisions > 0]
            spatial_cropped = spatial_cropped * block_weights[:, :, None]
            
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
    projector = get_projector(U_r)

    ## Step 2f: Do sparse regression to get the V matrix: 
    display("Running sparse regression")
    V = load_obj.V_projection(projector)

    ## Step 2g: Aggregate the global SVD with the localMD results to create the final decomposition
    display("Aggregating Global SVD with localMD results")
    U_r, V = aggregate_decomposition(U_r, V, load_obj)

    ## Step 2h: Do a SVD Reformat given U and V
    display("Running QR decomposition on V")
    R, s, Vt = factored_svd(U_r, V)

    display("Matrix decomposition completed")

    return U_r, R, s, Vt, load_obj


def aggregate_decomposition(U_r, V, load_obj):
    if load_obj.background_rank == 0:
        return U_r, V
    else:
        spatial_bg = load_obj.spatial_basis
        temporal_bg = load_obj.temporal_basis
        spatial_bg_sparse = scipy.sparse.coo_matrix(spatial_bg)
        U_r = scipy.sparse.hstack([U_r, spatial_bg_sparse])
        V = np.concatenate([V, temporal_bg], axis = 0)
    
        return U_r, V