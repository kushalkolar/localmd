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

from localmd.evaluation import spatial_roughness_stat_vmap, temporal_roughness_stat_vmap, construct_final_fitness_decision, filter_by_failures


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

import scipy.sparse

from localmd.preprocessing_utils import get_noise_estimate_vmap, center_and_get_noise_estimate
from localmd.tiff_loader import tiff_loader


import time
import datetime
import sys
import pdb


def display(msg):
    """
    Printing utility that logs time and flushes.
    """
    tag = '[' + datetime.datetime.today().strftime('%y-%m-%d %H:%M:%S') + ']: '
    sys.stdout.write(tag + msg + '\n')
    sys.stdout.flush()

@partial(jit)
def truncated_random_svd(input_matrix, key, rank_placeholder):
    '''
    Input: 
        - input_matrix. jnp.ndarray (d, T), where d is number of pixels, T is number of frames
        - key: jax pseudorandom key for random data gen
        - rank_placeholder: jax.ndarray with shape (rank). We use the shape (rank) to make a matrix with "rank" columns. This is
            a standard workaround for making sure this function can be jitted. 
    Key: This function assumes that (1) rank + num_oversamples is less than all dimensions of the input_matrix and (2) num_oversmples >= 1
    
    '''
    num_oversamples=10
    rank = rank_placeholder.shape[0]
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
def decomposition_no_normalize_approx(block, key, rank_placeholder):
    d1, d2, T = block.shape
    block_2d = jnp.reshape(block, (d1*d2, T), order="F")
    decomposition = truncated_random_svd(block_2d, key, rank_placeholder)
    
    u_mat, v_mat = decomposition[0], decomposition[1]
    u_mat = jnp.reshape(u_mat, (d1, d2, u_mat.shape[1]), order="F")
    
    spatial_statistics = spatial_roughness_stat_vmap(u_mat)
    temporal_statistics = temporal_roughness_stat_vmap(v_mat)

    return spatial_statistics, temporal_statistics

  
@partial(jit, static_argnums=(0,1,2))
def rank_simulation(d1, d2, T, rank_placeholder, key1, key2):
    noise_data = jax.random.normal(key1, (d1, d2, T))
    spatial, temporal = decomposition_no_normalize_approx(noise_data, key2, rank_placeholder)
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
    rank_placeholder = np.zeros((num_comps,))
    for k in range(iters):
        key1 = make_jax_random_key()
        key2 = make_jax_random_key()
        x, y = rank_simulation(d1, d2, T, rank_placeholder, key1, key2)
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
    
    return single_block_md(block, projection_data, spatial_thres, temporal_thres, max_consec_failures)
    
@partial(jit)
def single_block_md(block, key, rank_placeholder, spatial_thres, temporal_thres):
    '''
    Matrix Decomposition function for all blocks. 
    Inputs: 
        - block: jnp.array. Dimensions (block_1, block_2, T). (block_1, block_2) are the dimensions of this patch of data, T is the number of frames. We assume that this data has already been centered and noise-normalized
        - key: jax random number key. 
        - rank_placeholder: jnp.array. Dimensions (max_rank,). Maximum rank of the low-rank decomposition which we permit over this block. We pass this information via shape of an array to enable full JIT of this function 
        - spatial_thres. float. We compute a spatial roughness statistic for each spatial component to determine whether it is noise or smoother signal. This is the threshold for that test. 
        - temporal_thres. float. We compute a temporal roughness statistic for each temporal component to determine whether it is noise or smoother signal. This is the threshold for that test. 
        
    '''
    d1, d2, T = block.shape
    block_2d = jnp.reshape(block, (d1*d2, T), order="F")
    
    
    decomposition = truncated_random_svd(block_2d, key, rank_placeholder)
    u_mat, v_mat = decomposition[0], decomposition[1]
    u_mat = jnp.reshape(u_mat, (d1, d2, u_mat.shape[1]), order="F")

    
    # Now we begin the evaluation phase
    good_comps = construct_final_fitness_decision(u_mat, v_mat.T, spatial_thres,\
                                                  temporal_thres)
    
    return u_mat, good_comps, v_mat

@partial(jit)
def single_residual_block_md(block, existing, key, rank_placeholder, spatial_thres, temporal_thres):
    '''
    Matrix Decomposition function for all blocks. 
    Inputs: 
        - block: jnp.array. Dimensions (block_1, block_2, T). (block_1, block_2) are the dimensions of this patch of data, T is the number of frames. We assume that this data has already been centered and noise-normalized
        - existing: jnp.array. Dimensions (block_1, block_2, T). (block_1, block_2) are the dimensions of this patch of data, T is the number of frames. This is an orthonormal spatial basis set which has already been identified for this spatial block of the FOV. We subtract it from "block" (via linear subspace projection) and THEN run the truncated SVD. The goal here is to find neural signal from "block" which is not already identified by "existing". 
        - key: jax random number key. 
        - rank_placeholder: jnp.array. Dimensions (max_rank,). Maximum rank of the low-rank decomposition which we permit over this block. We pass this information via shape of an array to enable full JIT of this function 
        - spatial_thres. float. We compute a spatial roughness statistic for each spatial component to determine whether it is noise or smoother signal. This is the threshold for that test. 
        - temporal_thres. float. We compute a temporal roughness statistic for each temporal component to determine whether it is noise or smoother signal. This is the threshold for that test. 
    '''
    d1, d2, T = block.shape
    net_comps = existing.shape[2]
    block_2d = jnp.reshape(block, (d1*d2, T), order="F")
    existing_2d = jnp.reshape(existing, (d1*d2, net_comps), order="F")
    
    projection = jnp.matmul(existing_2d, jnp.matmul(existing_2d.transpose(), block_2d))
    block_2d = block_2d - projection
    
    
    decomposition = truncated_random_svd(block_2d, key, rank_placeholder)
    u_mat, v_mat = decomposition[0], decomposition[1]
    u_mat = jnp.reshape(u_mat, (d1, d2, u_mat.shape[1]), order="F")

    
    # Now we begin the evaluation phase
    good_comps = construct_final_fitness_decision(u_mat, v_mat.T, spatial_thres,\
                                                  temporal_thres)
    
    return u_mat, good_comps, v_mat


def windowed_pmd(window_length, block, max_rank, spatial_thres, temporal_thres, max_consec_failures):
    '''
    Implementation of windowed blockwise decomposition. Given a block of the movie (d1, d2, T), we break the movie into smaller chunks. 
    (say (d1, d2, R) where R < T), and run the truncated SVD decomposition iteratively on these blocks. This helps (1) avoid rank blowup and
    (2) make sure our spatial fit 
    
    Inputs: 
        - window_length: int. We break up the block into temporal subsets of this length and do the blockwise SVD decomposition on these blocks
        - block: np.ndarray. Shape (d1, d2, T)
        - max_rank: We break up "block" into temporal segments of length "window_length", and we run truncated SVD on each of these subsets iteratively. max_rank is the max rank of the decomposition we can obtain from any one of these individual blocks
        - spatial_thres: float. See single_block_md for docs
        - temporal_thres. float. See single_block_md for docs
        - max_consec_failures: int. After running the truncated SVD on this data, we look at each pair of rank-1 components (spatial, temporal) in order of significance (singular values). Once the hypothesis test fails a certain number of times on this data, we discard all subsequent components from the decomposition. 
    '''
    d1, d2 = (block.shape[0], block.shape[1])
    window_range = block.shape[2]
    assert window_length <= window_range
    start_points = list(range(0, window_range, window_length))
    if start_points[-1] > window_range - window_length:
        start_points[-1] = window_range - window_length
    
    final_decomposition = np.zeros((d1, d2, max_rank))
    remaining_components = max_rank
    
    component_counter = 0
    rank_placeholder_list = [np.zeros((max_rank,)) for i in range(0, max_rank+1)]
    
    for k in start_points:
        start_value = k
        end_value = start_value + window_length
        
        key = make_jax_random_key()
        if k == 0 or final_decomposition.shape[2] == 0:
            subset = block[:, :, start_value:end_value]
            rank_placeholder = rank_placeholder_list[remaining_components]
            spatial_comps, decisions, _ = single_block_md(subset, key, rank_placeholder, spatial_thres, temporal_thres)
            spatial_comps = np.array(spatial_comps)
            decisions = np.array(decisions).flatten() > 0
            decisions = filter_by_failures(decisions, max_consec_failures)
            spatial_cropped = spatial_comps[:, :, decisions]
            final_filter_index = min(spatial_cropped.shape[2], remaining_components)
            spatial_cropped = spatial_cropped[:, :, :final_filter_index]
        else:
            subset = block[:, :, start_value:end_value]
            rank_placeholder = rank_placeholder_list[remaining_components]
            spatial_comps, decisions, _ = single_residual_block_md(subset, final_decomposition, key, rank_placeholder, spatial_thres, temporal_thres)
            spatial_comps = np.array(spatial_comps)
            decisions = np.array(decisions).flatten() > 0
            decisions = filter_by_failures(decisions, max_consec_failures)
            spatial_cropped = spatial_comps[:, :, decisions]
            final_filter_index = min(spatial_cropped.shape[2], remaining_components)
            spatial_cropped = spatial_cropped[:, :, :final_filter_index]
        
        final_decomposition[:, :, component_counter:component_counter + spatial_cropped.shape[2]] = spatial_cropped
        component_counter += spatial_cropped.shape[2]
        if component_counter == max_rank: 
            break
        else:
            remaining_components = max_rank - component_counter
        
    final_decomposition = final_decomposition[:, :, :component_counter]
    return final_decomposition
    

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


def get_projector(U):
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
    return (final_matrix_r_sparse.T, theta)


#Once Jax has better support for fused sparse-dense matrix operations, can merge the below two functions into one "jitted" function
def eigenvalue_and_eigenvec_routine(Sigma):
    
    eig_vals, eig_vecs = jnp.linalg.eigh(Sigma)  # Note: eig vals/vecs ascending
    eig_vals = np.array(eig_vals)
    eig_vecs = np.array(eig_vecs)
    eig_vecs = np.flip(eig_vecs, axis=(1,))
    eig_vals = np.flip(eig_vals, axis=(0,))
    
    return eig_vecs, eig_vals

def compute_sigma(spatial_components, Lt):
    '''
    Note: Lt here refers to the upper triangular matrix from the QR factorization of V ("temporal_components"). So 
    temporal_components.T = Qt.dot(Lt), which means that 
    UV = U(Lt.T)(Qt.T)
    '''
    Lt = np.array(Lt)
    UtU = spatial_components.T.dot(spatial_components)
    UtUL = UtU.dot(Lt.T)
    Sigma = Lt.dot(UtUL)
    
    return Sigma


def factored_svd(spatial_components, temporal_components, factor = 0.25):
    '''
    This is a fast method to convert a low-rank decomposition (spatial_components * temporal_components) into a 
    Inputs: 
        spatial_components: scipy.sparse.coo_matrix. Shape (d, R)
        temporal_components: jax.numpy. Shape (R, T)
    '''
    Qt, Lt=  jnp.linalg.qr(temporal_components.transpose(), mode='reduced')
    Sigma = compute_sigma(spatial_components, Lt)
    eig_vecs, eig_vals = eigenvalue_and_eigenvec_routine(Sigma)
    Qt = np.array(Qt)
    Lt = np.array(Lt)
    
    eig_vec_norms = np.linalg.norm(eig_vecs, axis = 0)
    selected_indices = eig_vals > 0
    eig_vecs = eig_vecs[:, selected_indices]
    eig_vals = eig_vals[selected_indices]
    singular_values = np.sqrt(eig_vals)
    
    mixing_weights = np.array(jnp.matmul(Lt.T, eig_vecs / singular_values[None, :]))
    temporal_basis = np.array(jnp.matmul(eig_vecs.T, Qt.T))
    
    #Here we prune the factorized SVD 
    return rank_prune_svd(mixing_weights, singular_values, temporal_basis, factor = factor)
    

def rank_prune_svd(mixing_weights, singular_values, temporal_basis, factor = 0.25):
    '''
    Inputs: 
        mixing_weights: numpy.ndarray, shape (R x R)
        singular_values: numpy.ndarray, shape (R)
        temporal_basis: numpy.ndarray, shape (R, T)
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


def localmd_decomposition(filename, block_sizes, overlap, frame_range, window_length, max_components=50, background_rank=15, sim_conf=5, batching=10, tiff_batch_size = 10000, dtype='float32', order="F", num_workers=0, pixel_batch_size=5000, frame_corrector_obj = None, max_consec_failures = 1, rank_prune_factor=1):
            
    load_obj = tiff_loader(filename, dtype=dtype, center=True, normalize=True, background_rank=background_rank, batch_size=tiff_batch_size, order=order, num_workers=num_workers, pixel_batch_size=pixel_batch_size, frame_corrector_obj = frame_corrector_obj)
    start = frame_range[0]
    end = frame_range[1]
    end = min(end, load_obj.shape[2])
    frames = [i for i in range(start, end)]
    assert end - start >= window_length and window_length > 0, "Window length needs to be nonneg and at most the range of frames in spatial fit"
    block_sizes = block_sizes
    overlap = overlap
    
    ##Step 2a: Get the spatial and temporal thresholds
    display("Running Simulations, block dimensions are {} x {} x {} ".format(block_sizes[0], block_sizes[1], window_length))
    spatial_thres, temporal_thres = threshold_heuristic([block_sizes[0], block_sizes[1], window_length], num_comps = 1, iters=250, percentile_threshold=sim_conf)
    
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
            
            spatial_cropped = windowed_pmd(window_length, subset, max_components, spatial_thres, temporal_thres, max_consec_failures)
            
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
    
    #Extract necessary info from the loader object and delete it. This frees up space on GPU for the below linalg.eigh computations
    std_img = load_obj.std_img
    mean_img = load_obj.mean_img
    order = load_obj.order
    shape = load_obj.shape
    del load_obj
    jax.clear_backends()
    
    U_r = U_r.astype(dtype)
    V = V.astype(dtype)

    ## Step 2h: Do a SVD Reformat given U and V
    display("Running QR decomposition on V")
    R, s, Vt = factored_svd(U_r, V, factor = rank_prune_factor)

    display("Matrix decomposition completed")

    return U_r, R, s, Vt, std_img, mean_img, shape, order#, load_obj


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