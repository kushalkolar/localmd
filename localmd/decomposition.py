import jax
import jax.scipy
import jax.numpy as jnp
import jax.numpy as jnp
from jax import jit, vmap
import functools
from functools import partial

import jaxopt

import numpy as np

@partial(jit)
def objective_function(X, placeholder, data):
    num_rows = data.shape[0]
    difference = data.shape[1]

    comp1 = jax.lax.dynamic_slice(X, (0, 0), (num_rows, X.shape[1]))
    comp2 = jax.lax.dynamic_slice(X, (num_rows, 0), (difference, X.shape[1]))
    prod = jnp.dot(comp1, comp2.T)
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
    
    v_k = jnp.dot(residual.T, u_k) #This is the debias/rescaling step
    
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

    