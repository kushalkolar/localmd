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
# from localmd.preprocessing_utils import standardize_block
from localmd.tiff_loader import tiff_loader

import sys
import datetime
import os

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



#@partial(jit)
def decomposition_no_normalize_approx(block, random_projection_mat):
    d1, d2, T = block.shape
    block_2d = jnp.reshape(block, (d1*d2, T), order="F")
    decomposition = truncated_random_svd(block_2d, random_projection_mat)
    
    u_mat, v_mat = decomposition[0], decomposition[1]
    u_mat = jnp.reshape(u_mat, (d1, d2, u_mat.shape[1]), order="F")
    
    spatial_statistics = spatial_roughness_stat_vmap(u_mat)
    temporal_statistics = temporal_roughness_stat_vmap(v_mat)

    return spatial_statistics, temporal_statistics

decomposition_no_normalize_approx_vmap = jit(vmap(decomposition_no_normalize_approx, in_axes = (3, 2)))


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

def threshold_heuristic(block_sizes, num_comps = 3, iters=10, num_sims=5, percentile_threshold = 5):
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

    for j in range(iters):
        noise_data = np.random.randn(d1, d2, T, num_sims)
        random_projection = np.random.randn(T, num_comps, num_sims)

        results = decomposition_no_normalize_approx_vmap(noise_data, random_projection)

        spatial_temp = results[0].reshape((-1,))
        temporal_temp = results[1].reshape((-1,))

        spatial_cumulator = np.concatenate([spatial_cumulator, spatial_temp])
        temporal_cumulator = np.concatenate([temporal_cumulator, temporal_temp])

    spatial_thres = np.percentile(spatial_cumulator.flatten(), percentile_threshold)
    temporal_thres = np.percentile(temporal_cumulator.flatten(), percentile_threshold)
    
    return spatial_thres, temporal_thres



#@partial(jit)
def single_block_md(block, projection_data, spatial_thres, temporal_thres, max_consec_failures):
    '''
    Matrix Decomposition function for all blocks. 
    Inputs: 
        block: jnp.array. Dimensions (block_1, block_2, T). We assume that this data has already been centered and noise-normalized
    
    '''
    #TODO: Get rid of max consec failures entirely from function API 
    # block = standardize_block(block) #Center and divide by noise standard deviation before doing matrix decomposition
    d1, d2, T = block.shape
    block_2d = jnp.reshape(block, (d1*d2, T), order="F")
    
    
    
    decomposition = truncated_random_svd(block_2d, projection_data)
    u_mat, v_mat = decomposition[0], decomposition[1]
    u_mat = jnp.reshape(u_mat, (d1, d2, u_mat.shape[1]), order="F")
    

    
    ##Now we begin the evaluation phase
    good_comps = construct_final_fitness_decision(u_mat, v_mat.T, spatial_thres,\
                                                  temporal_thres, max_consec_failures)
    
    return u_mat, good_comps, v_mat

single_block_md_vmap = jit(vmap(single_block_md, in_axes=(3,2, None, None, None)))

def append_decomposition_results(curr_results, new_results):
    '''
    Each results list has 3 ndarrays: 
        results[0]: The set of spatial components. Dimensions (# of blocks, d1*d2, num_comps per block). d1,d2 are block dimensions.
        results[1]: The decisions for whether or not to accept/reject the components. Dimensions (#num blocks, num_comps per block)
        results[2]: The set of temporal components for each block. Dimensions (# of blocks, num_comps, T) where T = number of frames in dataset.
    '''
    curr_results[0] = np.concatenate([curr_results[0], new_results[0]], axis=0)
    curr_results[1] = np.concatenate([curr_results[1], new_results[1]], axis=0)
    curr_results[2] = np.concatenate([curr_results[2], new_results[2]], axis=0)
    
    return curr_results

def cast_decomposition_results(new_results, cast):
    return (np.array(new_results[0], dtype=cast), np.array(new_results[1], dtype=cast), np.array(new_results[2], dtype=cast))
#     new_results[0] = np.array(new_results[0], dtype=cast)
#     new_results[1] = np.array(new_results[1], dtype=cast)
#     new_results[2] = np.array(new_results[2], dtype=cast)
    
#     return new_results



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


def localmd_decomposition(filename, block_sizes, overlap, frame_range, max_components=50, background_rank=15, sim_conf=5, batching=10, tiff_batch_size = 10000, dtype='float32', order="F"):
    load_obj = tiff_loader(filename, dtype=dtype, center=True, normalize=True, background_rank=background_rank, batch_size=tiff_batch_size, order=order)
    start = frame_range[0]
    end = frame_range[1]
    end = min(end, load_obj.shape[2])
    frames = [i for i in range(start, end)]
    block_sizes = block_sizes
    overlap = overlap
    
    ##Step 2a: Get the spatial and temporal thresholds
    display("Running Simulations")
    spatial_thres, temporal_thres = threshold_heuristic([block_sizes[0], block_sizes[1], len(frames)], num_comps = 1, iters=25, num_sims = 10, percentile_threshold=sim_conf)
    
    ##Step 2b: Load the data you will do blockwise SVD on
    display("Loading Data")
    data = load_obj.temporal_crop_with_filter(frames)
    
    ##Step 2c: Run PMD and get the U matrix components
    display("Obtaining blocks and running local SVD")
    cumulator = []

    start_t = time.time()

    pairs = []
    
    results = []
    results.append(np.zeros((0, block_sizes[0],block_sizes[1], max_components), dtype=dtype))
    results.append(np.zeros((0, max_components), dtype=dtype))
    results.append(np.zeros((0, max_components, len(frames)), dtype=dtype))
    
    cumulator_count = 0
    max_cumulator = batching

    dim_1_iters = list(range(0, data.shape[0] - block_sizes[0] + 1, block_sizes[0] - overlap[0]))
    if dim_1_iters[-1] != data.shape[0] - block_sizes[0] and data.shape[0] - block_sizes[0] != 0:
        dim_1_iters.append(data.shape[0] - block_sizes[0])

    dim_2_iters = list(range(0, data.shape[1] - block_sizes[1] + 1, block_sizes[1] - overlap[1]))
    if dim_2_iters[-1] != data.shape[1] - block_sizes[1] and data.shape[1] - block_sizes[1] != 0:
        dim_2_iters.append(data.shape[1] - block_sizes[1])

    for k in dim_1_iters:
        for j in dim_2_iters:
            pairs.append((k, j))
            subset = data[k:k+block_sizes[0], j:j+block_sizes[1], :].astype(dtype)
            cumulator.append(subset)
            cumulator_count += 1
            if cumulator_count >= max_cumulator or (k == dim_1_iters[-1] and j == dim_2_iters[-1]): #Ready to calculate the local SVD in batch across the patches
                input_blocks = np.array(cumulator).transpose(1,2,3,0).astype(dtype)
                projected_data = np.random.randn(input_blocks.shape[2], max_components, input_blocks.shape[3]).astype(dtype)
                new_results = single_block_md_vmap(input_blocks, projected_data, spatial_thres, temporal_thres, 1)
                new_results = cast_decomposition_results(new_results, dtype)
                results = append_decomposition_results(results, new_results)
                #Reset counting + cumulating variables
                cumulator = []
                cumulator_count = 0
    



    #Step 2e: Piece it all together into one orthonormal matrix (U-matrix)
    display("Collating results into large spatial basis matrix and projecting data")
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



    final_matrix = np.zeros((data.shape[0], data.shape[1], 0), dtype=dtype)
    for i in range(len(pairs)):
        dim_1_val = pairs[i][0]
        dim_2_val = pairs[i][1]

        spatial_comps = results[0][i, :, :, :]
        decisions = results[1][i, :].flatten()

        spatial_cropped = spatial_comps[:, :, decisions > 0]#.reshape((block_sizes[0], block_sizes[1], -1), order=order)
        spatial_cropped = spatial_cropped * block_weights[:, :, None]
        appendage = np.zeros((data.shape[0], data.shape[1], spatial_cropped.shape[2]), dtype=dtype)
        appendage[dim_1_val:dim_1_val + block_sizes[0], dim_2_val:dim_2_val + block_sizes[1], :] = spatial_cropped
        final_matrix = np.concatenate([final_matrix, appendage], axis = 2)

    

    U_r = final_matrix.reshape((data.shape[0]*data.shape[1], -1), order=order)
    U_r = scipy.sparse.csr_matrix(U_r)
    display("U_r type is {}".format(U_r.dtype))
    projector = get_projector(U_r)

    print(projector.shape)
    print("projector dtype is {}".format(projector.dtype))

    ## Step 2f: Do sparse regression to get the V matrix: 
    display("Running sparse regression")
    V = load_obj.batch_matmul_PMD_fast(projector)
    display("V type is {}".format(V.dtype))
    
    ## Step 2g: Aggregate the global SVD with the localMD results to create the final decomposition
    display("Aggregating Global SVD with localMD results")
    display("the bg rank is {}".format(load_obj.background_rank))
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