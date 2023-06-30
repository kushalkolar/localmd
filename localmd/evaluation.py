"""
File for all evaluation metrics and functionality related to total variation and trend filtering:
NOTE: Any function with a commented-out decorator "@partial(jit)" is a helper function; it's more effective 
to jit code at the highest level so the compiler can actually optimize the whole thing. But these functions are
jit-able. 
"""
import jax
import jax.scipy
import jax.numpy as jnp
from jax import jit, vmap
import functools
from functools import partial

import numpy as np

#@partial(jit)
def l1_norm(data):
    '''
    Calculates the overall l1 norm of the data
    '''
    data = jnp.abs(data)
    final_sum = jnp.sum(data)
    return final_sum

#@partial(jit)
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


#@partial(jit)
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

def spatial_roughness_stat(u):
    '''
    Input: 
        jax.numpy.array, u. Shape (d1, d2)
    
    Computes ratio of total variatio and l1 norm for u
    '''
    
    lower_vert = jax.lax.dynamic_slice(u, (1, 0), (u.shape[0]-1,u.shape[1]))
    upper_vert = jax.lax.dynamic_slice(u, (0, 0), (u.shape[0]-1, u.shape[1]))
    
    vert_diffs = jnp.abs(lower_vert - upper_vert)
    
    left_horz = jax.lax.dynamic_slice(u, (0,0), (u.shape[0], u.shape[1]-1))
    right_horz = jax.lax.dynamic_slice(u, (0, 1), (u.shape[0], u.shape[1]-1))
    
    horz_diffs = jnp.abs(left_horz - right_horz)
    avg_diff = (jnp.sum(vert_diffs) + jnp.sum(horz_diffs))/(vert_diffs.shape[0]*vert_diffs.shape[1] + horz_diffs.shape[0]*horz_diffs.shape[1])
    
    avg_elem = jnp.mean(jnp.abs(u))
    
    return avg_diff / avg_elem

def temporal_roughness_stat(v):
    '''
    Input: 
        v: jax.numpy.array. Dimenion (T)
    
    '''
    v_left = jax.lax.dynamic_slice(v, (0,), (v.shape[0]-2,))
    v_right = jax.lax.dynamic_slice(v, (2,), (v.shape[0]-2,))
    v_middle = jax.lax.dynamic_slice(v, (1,), (v.shape[0]-2,))
    
    return jnp.mean(jnp.abs(v_left + v_right - 2*v_middle)) / jnp.mean(jnp.abs(v))


spatial_roughness_stat_vmap = vmap(spatial_roughness_stat, in_axes=(2))
temporal_roughness_stat_vmap = vmap(temporal_roughness_stat, in_axes=(0))

#@partial(jit)
def get_roughness_stats(img, trace):
    spatial_stat = spatial_roughness_stat(img)
    temporal_stat = temporal_roughness_stat(trace)

#@partial(jit)
def evaluate_fitness(img, trace, spatial_thres, temporal_thres):
    spatial_stat = spatial_roughness_stat(img)
    temporal_stat = temporal_roughness_stat(trace)
    exp1 = spatial_stat < spatial_thres
    exp2 = temporal_stat < temporal_thres
    bool_exp = exp1 & exp2
    output = jax.lax.cond(bool_exp, lambda x:1, lambda x:0, None)
    
    
    return output

evaluate_fitness_vmap = vmap(evaluate_fitness, in_axes=(2, 1, None, None))

#@partial(jit)
def construct_final_fitness_decision(imgs, traces, spatial_thres, temporal_thres):
    output = evaluate_fitness_vmap(imgs, traces, spatial_thres, temporal_thres)
    return output

def filter_by_failures(decisions, max_consecutive_failures):
    '''
    Input: 
        - decisions: 1-dimensional np.ndarray. Boolean values.
    Ouput: 
        - decisions_filtered: same shape/types a decisionos. 
    '''
    number_of_failures = 0
    all_fails = False
    for k in range(decisions.shape[0]):
        if all_fails:
            decisions[k] = False
        elif not decisions[k]:
            number_of_failures += 1
            if number_of_failures == max_consecutive_failures:
                all_fails = True
        else:
            number_of_failures = 0
    return decisions
            
