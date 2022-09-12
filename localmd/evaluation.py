"""
File for all evaluation metrics and functionality related to total variation and trend filtering
"""
import jax
import jax.scipy
import jax.numpy as jnp
import jax.numpy as jnp
from jax import jit, vmap
import functools
from functools import partial

import numpy as np



@partial(jit)
def l1_norm(data):
    '''
    Calculates the overall l1 norm of the data
    '''
    data = jnp.abs(data)
    final_sum = jnp.sum(data)
    return final_sum

@partial(jit)
def trend_filter_stat(trace):
    '''
    Applies a trend filter to a 1D time series dataset
    Key assumption, data has length at least 3
    inputs: 
        trace: np.array (or jnp.array) of shape (T,) 
    Outputs: 
        trend_filter_stat: single value (float)
    '''
    
    length = trace.shape[0]
    left_side = jax.lax.dynamic_slice(trace, (0,), (trace.shape[0] - 2,))
    right_side = jax.lax.dynamic_slice(trace, (2,), (trace.shape[0] - 2,))
    center = jax.lax.dynamic_slice(trace, (1,), (trace.shape[0] - 2,))
    
    combined_mat = center * 2 - left_side - right_side
    combined_mat = jnp.abs(combined_mat)
    return jnp.sum(combined_mat)


@partial(jit)
def total_variation_stat(img):
    '''
    Applies a total variation filter to a 2D image
    Key assumption: image has size at least 3 x 3 pixels
    Input:
        img: np.array (or jnp.array) of shape (x, y)
    '''
    
    center = jax.lax.dynamic_slice(img, (1, 1), \
                                   (img.shape[0] - 2, img.shape[1] - 2))
    c00 = jax.lax.dynamic_slice(img, (0, 0), \
                                 (img.shape[0] - 2, img.shape[1] - 2))
    c10 = jax.lax.dynamic_slice(img, (1, 0), \
                               (img.shape[0] - 2, img.shape[1] - 2))
    c20 =jax.lax.dynamic_slice(img, (2, 0), \
                               (img.shape[0] - 2, img.shape[1] - 2))
    c21 = jax.lax.dynamic_slice(img, (2, 1), \
                                (img.shape[0] - 2, img.shape[1] - 2))
    c22 = jax.lax.dynamic_slice(img, (2, 2), \
                                (img.shape[0] - 2, img.shape[1] - 2))
    c12 = jax.lax.dynamic_slice(img, (1, 2), \
                                (img.shape[0] - 2, img.shape[1] - 2))
    c02 = jax.lax.dynamic_slice(img, (0, 2), \
                                (img.shape[0] - 2, img.shape[1] - 2))
    c01 = jax.lax.dynamic_slice(img, (0, 1), \
                                (img.shape[0] - 2, img.shape[1] - 2))
    
    accumulator = jnp.zeros_like(center)
    
    accumulator = accumulator + jnp.abs(center - c00)
    accumulator = accumulator + jnp.abs(center - c10)
    accumulator = accumulator + jnp.abs(center - c20)
    accumulator = accumulator + jnp.abs(center - c21)
    accumulator = accumulator + jnp.abs(center - c22)
    accumulator = accumulator + jnp.abs(center - c12)
    accumulator = accumulator + jnp.abs(center - c02)
    accumulator = accumulator + jnp.abs(center - c01)
    
    return jnp.sum(accumulator)

@partial(jit)
def spatial_roughness_stat(img):
    '''
    Input: 
        img: jnp.array, dimensions (d1, d2) where d1 and d2 are image FOV dimensions
    Output: 
        stat: float. spatial roughness statistic
    
    '''
    img = img / jnp.linalg.norm(img)
    numerator = total_variation_stat(img)
    denominator = l1_norm(img)
    return numerator/denominator

@partial(jit)
def temporal_roughness_stat(trace):
    '''
    Input: 
        img: jnp.array, shape (T,) where T is the length of the temporal trace
    Output: 
        stat: float. temporal roughness statistic statistic
    
    '''
    trace = trace / jnp.linalg.norm(trace)
    numerator = trend_filter_stat(trace)
    denominator = l1_norm(trace)
    return numerator/denominator

spatial_roughness_stat_vmap = jit(vmap(spatial_roughness_stat, in_axes=(2)))
temporal_roughness_stat_vmap = vmap(temporal_roughness_stat, in_axes=(0))

@partial(jit)
def get_roughness_stats(img, trace):
    spatial_stat = spatial_roughness_stat(img)
    temporal_stat = temporal_roughness_stat(trace)

@partial(jit)
def evaluate_fitness(img, trace, spatial_thres, temporal_thres):
    spatial_stat = spatial_roughness_stat(img)
    temporal_stat = temporal_roughness_stat(trace)
    exp1 = spatial_stat < spatial_thres
    exp2 = temporal_stat < temporal_thres
    bool_exp = exp1 & exp2
    output = jax.lax.cond(bool_exp, lambda x:1, lambda x:0, None)
    
    
    return output

evaluate_fitness_vmap = jit(vmap(evaluate_fitness, in_axes=(2, 1, None, None)))

@partial(jit)
def construct_final_fitness_decision(imgs, traces, spatial_thres, temporal_thres, max_consec_failures):
    output = evaluate_fitness_vmap(imgs, traces, spatial_thres, temporal_thres)
    
    final_output = successive_filter(output, max_consec_failures)
    return final_output

@partial(jit)
def successive_filter(my_list, max_consec_failures):
    start = -1
    consec_failures = 0
    
    init_pytree = (consec_failures, start, my_list, max_consec_failures)
    final_pytree = jax.lax.fori_loop(0, my_list.shape[0], manage_successive_failures_iteration, init_pytree)
    start = final_pytree[1]
    my_list = jax.lax.cond(start == -1, lambda x:x[0], filter_from_starting_pt, (my_list, start))
    
    return my_list
    
@partial(jit)                
def filter_from_starting_pt(input_pytree):
    my_list = input_pytree[0]
    start = input_pytree[1]
    temp_arange = jnp.arange(my_list.shape[0])
    temp_arange = temp_arange < start
    
    return my_list * temp_arange

@partial(jit)
def manage_failure(input_pytree):
    consec_failures = input_pytree[0]
    start = input_pytree[1]
    k = input_pytree[2]
    max_consec_failures = input_pytree[3]
    
    new_pytree = jax.lax.cond(consec_failures == 0, first_failure, multi_failure, (k, start, consec_failures))
    
    return (new_pytree[0], new_pytree[1])

@partial(jit)
def first_failure(input_pytree):
    k = input_pytree[0]
    start = input_pytree[1]
    consec_failures = input_pytree[2]
    
    
    start = k
    consec_failures = consec_failures + 1
    return (consec_failures, start)

@partial(jit)
def multi_failure(input_pytree):
    k = input_pytree[0]
    start = input_pytree[1]
    consec_failures = input_pytree[2]

    consec_failures = consec_failures + 1
    start = start
    
    return (consec_failures, start)

def manage_success(input_pytree):
    consec_failures = input_pytree[0]
    start = input_pytree[1]
    k = input_pytree[2]
    max_consec_failures = input_pytree[3]
    
    consec_failures = jax.lax.cond(consec_failures < max_consec_failures, lambda x:0, lambda x:x, consec_failures)
    
    return (consec_failures, start)
    
    
def manage_successive_failures_iteration(k, input_pytree):
    consec_failures = input_pytree[0]
    start = input_pytree[1]
    my_list = input_pytree[2]
    max_consec_failures = input_pytree[3]
    value = jnp.take(my_list, k)
    
    constructed_pytree = (consec_failures, start, k, max_consec_failures)
    
    output_pytree = jax.lax.cond(value == 0, manage_failure, manage_success, constructed_pytree)
    
    consec_failures = output_pytree[0]
    start = output_pytree[1]
    
    return (consec_failures, start, my_list, max_consec_failures)

