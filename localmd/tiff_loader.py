import os
import pathlib
import sys
import math
import tifffile

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
import functools
from functools import partial



import scipy.sparse
import scipy.sparse.linalg
import skimage.io
import skimage.measure

from localmd.preprocessing_utils import get_noise_estimate_vmap, center_and_get_noise_estimate

from sklearn.utils.extmath import randomized_svd

from tqdm import tqdm

from sys import getsizeof
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


class tiff_loader():
    def __init__(self, filename, dtype='float32', center=True, normalize=True, background_rank=15, batch_size=10000, order="F"):
        with tifffile.TiffFile(filename) as tffl: 
            if len(tffl.pages) == 1: 
                raise ValueError("PMD does not accept single-page tiff datasets. Instead, pass your raw through the pipeline starting from the motion correction step.")
        self.order = order
        self.filename = filename
        self.dtype = dtype
        self.shape = self._get_shape()
        self._estimate_batch_size(frame_const=batch_size)
        self.center = center
        self.normalize=normalize
        # if background_rank > 0:
            # print("Currently pre-matrix decomposition SVD is not supported")
        self.background_rank = background_rank
        self.shape = self._get_shape()
        self._initialize_all_normalizers()
        self._initialize_all_background()
    
    
    def _get_size_in_GB(self, obj_to_measure):
        val = getsizeof(obj_to_measure)
        return val/(1024**3)
    
    def _estimate_batch_size_heuristic(self, desired_size_in_GB=50, num_frames_to_sim=10):
        test = np.zeros((self.shape[0], self.shape[1], num_frames_to_sim), dtype=self.dtype)
        gb_size = self._get_size_in_GB(test)
        
        multiplier = math.ceil(desired_size_in_GB/gb_size)
        return multiplier * num_frames_to_sim  #To be safe
        

    
    def _estimate_batch_size(self, frame_const = 20000):
        est_1 = self._estimate_batch_size_heuristic()
        self.batch_size = min(frame_const, est_1)
        display("The batch size used is {}".format(self.batch_size))

        
    
    def _get_shape(self):
        with tifffile.TiffFile(self.filename) as tffl:
            num_frames = len(tffl.pages)
            for page in tffl.pages[0:1]:
                image = page.asarray()
                x, y = page.shape
        return (x,y,num_frames)
                
    def temporal_crop(self, frames):
        '''
        Input: 
            frame_range: tuple (a, b). 'a' is the python index of the first frame and 'b-1' is the python index
            of the last frame
        Returns: 
            Array containing this data
        '''
        x = tifffile.imread(self.filename, key=frames).transpose(1,2,0)
        return x.astype(self.dtype)
        
        
    #DONE
    def _initialize_all_normalizers(self):
        '''
        Constructs mean image and normalization image
        '''
        display("centering...")
        if self.center: 
            self.mean_img = self._calculate_mean()
        else:
            self.mean_img = np.zeros((self.shape[0], self.shape[1])).astype(self.dtype)
        display("normalizing...")
        if self.normalize:
            self.std_img = self._calculate_normalizer()
        else:
            self.std_img = np.ones((self.shape[0], self.shape[1])).astype(self.dtype) 
        return self.mean_img, self.std_img
    
    def _initialize_all_background(self):
        results = self._calculate_background_filter()
        self.spatial_basis = results[0]
        self.temporal_basis = results[1]
     
    #DONE
    def _calculate_mean(self):
        '''
        Calculate the (exact) mean of the dataset, using jax-accelerated numpy functions
        Returns: 
            mean: np.ndarray of shape (self.shape[0], self.shape[1])
        '''
        starttime=time.time()
        num_frames = self.shape[2]
        chunksize= 2000
        num_iters = math.ceil(num_frames / chunksize)
        overall_mean = jnp.zeros((self.shape[0], self.shape[1]))
        num_frames = self.shape[2]
        
        for k in range(num_iters):
            start = chunksize * k
            end = min(num_frames, chunksize*(k+1))
            frames = [i for i in range(start, end)]
            data = tifffile.imread(self.filename, key=frames).astype(self.dtype)
            mean_value = jnp.sum(data, axis=0) / num_frames
            overall_mean = overall_mean + mean_value
        display('Calculate mean took {:.2f} seconds'.format(time.time() - starttime))
        return np.array(overall_mean, dtype=self.dtype)
    
    
    def _calculate_normalizer(self, num_iters = 20):
        if self.shape[2] <= self.batch_size: 
            display("Calculating noise estimate on full data")
            return self._calculate_normalizer_full()
        else:
            sample_size = self.batch_size
            display("Calculating normalizer in subregions")
            cumulator = np.zeros((self.shape[0], self.shape[1]), dtype=self.dtype)
            a = [i for i in range(self.shape[2] - sample_size)]
            num_iters = min(num_iters, self.shape[2] - sample_size)
            start_pts = np.random.choice(a, size = num_iters, replace = False)
            for k in tqdm(range(num_iters)):
                start_pt = start_pts[k]
                frames = [i for i in range(start_pt, min(self.shape[2], start_pt + sample_size))]
                crop_data = tifffile.imread(self.filename, key=frames).transpose(1,2,0).astype(self.dtype)
                noise_est_2d = np.array(center_and_get_noise_estimate(crop_data, self.mean_img), dtype=self.dtype)
                cumulator += noise_est_2d
                
            cumulator /= num_iters
            cumulator[cumulator == 0] = 1.0
            return cumulator
    
    #DONE
    def _calculate_normalizer_full(self):
        frames = [i for i in range(self.shape[2])]
        dataset = tifffile.imread(self.filename).transpose(1,2,0).astype(self.dtype)
        norms = np.array(center_and_get_noise_estimate(dataset, self.mean_img), dtype=self.dtype)
        norms[norms == 0] = 1.0
        return norms
        
    def temporal_crop_standardized(self, frames):
        crop_data = self.temporal_crop(frames)
        crop_data -= self.mean_img[:, :, None]
        crop_data /= self.std_img[:, :, None]
        
        return crop_data.astype(self.dtype)
    

    def _calculate_background_filter(self, n_samples = 10000):
        if self.background_rank <= 0:
            return (np.zeros((self.shape[0]*self.shape[1], 1)).astype(self.dtype), np.zeros((1, self.shape[2])).astype(self.dtype))
        sample_list = [i for i in range(0, self.shape[2])]
        random_data = np.random.choice(sample_list,replace=False, size=min(n_samples, self.shape[2]))
        crop_data = self.temporal_crop_standardized(random_data)
        
        spatial_basis, _, _ = randomized_svd(
        M=crop_data.reshape((-1, crop_data.shape[-1]), order=self.order),
        n_components=self.background_rank
    )
        

        '''
        Given orthonormal spatial basis, we calculuate the temporal basis as: 
        s^t * (Data - mean * 1^t) / stdv
        '''
        
        stdv_reshape = self.std_img.reshape((1, -1), order=self.order)
        spatial_basis_scaled = spatial_basis.T / stdv_reshape 
        
        scaled_spatial_mean = spatial_basis_scaled.dot(self.mean_img.reshape((-1, 1), order=self.order)) #This is s^t/stdv * mean. Shape bg_rank x 1 
        
        #Given orthonormal spatial basis, we now calculate the temporal basis
        num_iters = math.ceil(self.shape[2]/self.batch_size)
        temporal_basis = np.zeros((spatial_basis.shape[1], self.shape[2]))
        display("Calculating temporal background basis")
        for k in tqdm(range(num_iters)):
            start = k*self.batch_size
            end = min((k+1)*self.batch_size, self.shape[2])
            frames = [i for i in range(start, end)]
            crop_data = self.temporal_crop(frames).reshape((-1, len(frames)), order=self.order)
            prod = spatial_basis_scaled.dot(crop_data)
            temporal_basis[:, start:end] = prod
        
        
        temporal_basis -= scaled_spatial_mean
        return (spatial_basis.astype(self.dtype), temporal_basis.astype(self.dtype))
    
    
    def batch_matmul_PMD_fast(self, M):
        '''
        Efficient batch matmul for sparse regression
        Assumption: M has a small number of rows (and many columns)
        TODO: Add use jax.numpy instead of numpy for gpu speedup here
        '''
        result = np.zeros((M.shape[0], self.shape[2]), dtype=self.dtype)
        num_iters = math.ceil(self.shape[2]/self.batch_size)
        MX = M.dot(self.spatial_basis) #This will be r x (background_rank)
        MXY = MX.dot(self.temporal_basis) #This will be r x T
        
        M_std = M / self.std_img.reshape((1, -1), order=self.order)
        M_std_mean = M_std.dot(self.mean_img.reshape((-1, 1), order=self.order))
        
        for k in tqdm(range(num_iters)):
            start = k *self.batch_size
            end = min((k+1)*self.batch_size, self.shape[2])
            frames = [i for i in range(start, end)]
            crop_data = self.temporal_crop(frames).reshape((-1, len(frames)), order=self.order)
            result[:, start:end] = M_std.dot(crop_data)  #This is now r x batch_size 
            
        result -= M_std_mean
        result -= MXY
        
        return result
    
    
    def temporal_crop_with_filter(self, frames):
        crop_data = self.temporal_crop_standardized(frames)
        spatial_r = self.spatial_basis.reshape((self.shape[0], self.shape[1], self.spatial_basis.shape[1]), order=self.order)
        temporal_basis_crop = self.temporal_basis[:, frames]
        return np.array(filter_components(crop_data, spatial_r, temporal_basis_crop))
        # subt = np.tensordot(spatial_r, temporal_basis_crop, axes=(2,0))
        # crop_data -= subt
        # return crop_data
                        
  
@partial(jit)
def filter_components(data, spatial_r, temporal_basis):
    subt = jnp.tensordot(spatial_r, temporal_basis, axes=(2, 0))
    return data - subt
    
