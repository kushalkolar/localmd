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
import torch
import torch.multiprocessing as multiprocessing


import scipy.sparse
import scipy.sparse.linalg
import skimage.io
import skimage.measure

from localmd.preprocessing_utils import get_noise_estimate_vmap, center_and_get_noise_estimate, get_mean_and_noise

from sklearn.utils.extmath import randomized_svd

from tqdm import tqdm

from sys import getsizeof
import time
import datetime
import sys


import jax
import jax.scipy
import jax.numpy as jnp
from jax import jit, vmap
import functools
from functools import partial



def display(msg):
    """
    Printing utility that logs time and flushes.
    """
    tag = '[' + datetime.datetime.today().strftime('%y-%m-%d %H:%M:%S') + ']: '
    sys.stdout.write(tag + msg + '\n')
    sys.stdout.flush()
    

@partial(jit)
def truncated_random_svd(input_matrix, random_data):
    desired_rank = random_data.shape[1]
    projected = jnp.matmul(input_matrix, random_data)
    Q, R = jnp.linalg.qr(projected)
    B = jnp.matmul(Q.T, input_matrix)
    U, s, V = jnp.linalg.svd(B, full_matrices=False)
    
    U_final = Q.dot(U)
    V = jnp.multiply(jnp.expand_dims(s, axis=1), V)
    return [U_final, V]

    

def regular_collate(batch):
    return np.array(batch)

class TiffLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=regular_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn)


class tiff_dataset():
    def __init__(self, filename, batch_size):
        self.filename = filename
        self.shape = self._get_shape()
        self.chunks = math.ceil(self.shape[2]/batch_size)
        self.batch_size = batch_size
        
        
        
    def __len__(self):
        return max(1, self.chunks - 1)
    
    def _get_shape(self):
        with tifffile.TiffFile(self.filename) as tffl:
            num_frames = len(tffl.pages)
            for page in tffl.pages[0:1]:
                image = page.asarray()
                x, y = page.shape
        return (x,y,num_frames)
    
    def __getitem__(self, index):
        start_time = time.time()
        start = index * self.batch_size
        if index == max(0, self.chunks - 2):
            end = self.shape[2]
            #load rest of data here
            keys = [i for i in range(start, end)]
            data = tifffile.imread(self.filename, key=keys)
        elif index < self.chunks - 2:
            end = start + self.batch_size
            keys = [i for i in range(start, end)]
            data= tifffile.imread(self.filename, key=keys)
        else:
            raise ValueError
        return data
    



class tiff_loader():
    def __init__(self, filename, dtype='float32', center=True, normalize=True, background_rank=15, batch_size=2000, order="F", num_workers = None):
        with tifffile.TiffFile(filename) as tffl: 
            if len(tffl.pages) == 1: 
                raise ValueError("PMD does not accept single-page tiff datasets. Instead, pass your raw through the pipeline starting from the motion correction step.")
        self.order = order
        self.filename = filename
        self.dtype = dtype
        self.shape = self._get_shape()
        self._estimate_batch_size(frame_const=batch_size)
        
        #Define the tiff loader
        self.tiff_dataobj = tiff_dataset(self.filename, self.batch_size)
        if num_workers is None:
            num_cpu = multiprocessing.cpu_count()
            num_workers = min(num_cpu - 1, len(self.tiff_dataobj))
            num_workers = max(num_workers, 0)

        
        self.loader = torch.utils.data.DataLoader(self.tiff_dataobj, batch_size=1,
                                             shuffle=False, num_workers=num_workers, collate_fn=regular_collate, timeout=0)
        
        self.center = center
        self.normalize=normalize
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
        

    
    def _estimate_batch_size(self, frame_const = 2000):
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
        
        
    def _initialize_all_normalizers(self):
        '''
        Constructs mean image and normalization image
        '''
        display("Computing Video Statistics")
        if self.center and self.normalize:
            results = self._calculate_mean_and_normalizer()
            self.mean_img = results[0]
            self.std_img = results[1]
        elif self.center: 
            self.mean_img = self._calculate_mean()
            self.std_img = np.ones((self.shape[0], self.shape[1])).astype(self.dtype) 
        elif self.normalize:
            self.std_img = self._calculate_normalizer()
            self.mean_img = np.zeros((self.shape[0], self.shape[1])).astype(self.dtype)
        else:
            self.mean_img = np.zeros((self.shape[0], self.shape[1])).astype(self.dtype)
            self.std_img = np.ones((self.shape[0], self.shape[1])).astype(self.dtype) 
        return self.mean_img, self.std_img
    
    def _initialize_all_background(self):
        self.spatial_basis = self._calculate_background_filter()
     
    def _calculate_mean(self):
        display("Calculating mean")
        overall_mean = jnp.zeros((self.shape[0], self.shape[1]))
        num_frames = self.shape[2]
        for i, data in enumerate(tqdm(self.loader), 0):
            data = jnp.array(data).squeeze()
            mean_value = jnp.sum(data, axis = 0) / num_frames
            overall_mean = overall_mean + mean_value
        display("Finished mean estimate")
        return np.array(overall_mean, dtype=self.dtype)
  
    
    def _calculate_mean_and_normalizer(self):
        '''
        This function takes a full pass through the dataset and calculates the mean and noise variance at the 
        same time, to avoid doing them separately
        '''
        display("Calculating mean and noise variance")
        overall_mean = jnp.zeros((self.shape[0], self.shape[1]))
        overall_normalizer = np.zeros((self.shape[0], self.shape[1]), dtype=self.dtype)
        num_frames = self.shape[2]
        for i, data in enumerate(tqdm(self.loader), 0):
            crop_data = jnp.array(data).squeeze().transpose(1,2,0)
            mean_value, noise_est_2d = get_mean_and_noise(crop_data, num_frames)
            overall_mean = overall_mean + mean_value
            overall_normalizer += np.array(noise_est_2d)
        overall_normalizer /= len(self.tiff_dataobj)
        overall_normalizer[overall_normalizer==0] = 1
        display("Finished mean and noise variance")
        return np.array(overall_mean, dtype=self.dtype), overall_normalizer
    
    
    def _calculate_normalizer(self):
        display("Calculating normalizer")
        overall_normalizer = np.zeros((self.shape[0], self.shape[1]), dtype=self.dtype)
        for i, data in enumerate(tqdm(self.loader), 0):
            crop_data = jnp.array(data).squeeze().transpose(1,2,0)
            noise_est_2d = np.array(center_and_get_noise_estimate(crop_data, self.mean_img), dtype=self.dtype)
            overall_normalizer += noise_est_2d
        overall_normalizer /= len(self.tiff_dataobj)
        overall_normalizer[overall_normalizer == 0] = 1
        display("Finished normalization")
        return np.array(overall_normalizer, dtype=self.dtype)
        
    def temporal_crop_standardized(self, frames):
        crop_data = self.temporal_crop(frames)
        crop_data -= self.mean_img[:, :, None]
        crop_data /= self.std_img[:, :, None]
        
        return crop_data.astype(self.dtype)

    def _calculate_background_filter(self, n_samples = 1000):
        if self.background_rank <= 0:
            return (np.zeros((self.shape[0]*self.shape[1], 1)).astype(self.dtype), np.zeros((1, self.shape[2])).astype(self.dtype))
        sample_list = [i for i in range(0, self.shape[2])]
        random_data = np.random.choice(sample_list,replace=False, size=min(n_samples, self.shape[2]))
        crop_data = self.temporal_crop_standardized(random_data)
        projection_data = np.random.randn(crop_data.shape[-1], int(self.background_rank)).astype(self.dtype)
        spatial_basis, _ = truncated_random_svd(crop_data.reshape((-1, crop_data.shape[-1]), order=self.order), projection_data)        
        return spatial_basis.astype(self.dtype)

    def V_projection(self, M):
        '''
        This function does two things, at a high level: 
        (1) It projects the standardized data onto U (via multiplication by M)
        (2) It calculates the temporal component from the full FOV SVD component as it does this
        
        Efficient batch matmul for sparse regression
        Assumption: M has a small number of rows (and many columns). It is R x d, where R is the
        rank of the decomposition and d is the number of pixels in the movie
        Here we compute M(D - a_1a_2 - mean)/stdv. We break the computation into temporal subsets, so if the dataset is T frames, we compute this product T' frames at a time (where T' usually around 1 or 2K) 
        R = Rank of PMD Decomposition
        d = number of pixels in dataset
        T = number of frames in dataset
        T' = number of frames we load at a time in below for loop
        p_r = rank of whole FOV SVD (self.spatial_basis.shape[1]). Typically very small, around 15
        
        
        M: Projection matrix given as input: it is R x T'
        D: The loaded dataset: dimensions d x T'
        a1 = spatial basis (self.spatial_basis): dimensions d x p_r
        a2 = temporal basis subset (calculation provided below): dimensions p_r x T'
        mean = mean of data (from self.mean_image -- it is reshaped to d x 1 here
        stdv = noise variance of data (from self.std_img -- it is reshaped to 1 x d here
        
        First: need to find a_2. To do so, compute: 
        a1^T(D - mean)/stdv. Most efficient way: first find a1^TD, 
        '''
        
        result = np.zeros((M.shape[0], self.shape[2]), dtype=self.dtype)
        mean_img_r = self.mean_img.reshape((-1, 1), order=self.order)
        std_img_r = self.std_img.reshape((1, -1), order=self.order)
        start = 0
        temporal_basis = np.zeros((self.spatial_basis.shape[1], self.shape[2]), dtype=self.dtype)

        start = 0
        for i, data in enumerate(tqdm(self.loader), 0):
            data = np.array(data).squeeze().transpose(1,2,0)
            num_frames_chunk = data.shape[2]
            D = data.reshape((-1, data.shape[2]), order=self.order)
            
            temporal_basis_chunk, output = V_projection_routine(M, D, self.spatial_basis, mean_img_r, std_img_r)
            
            endpt = min(self.shape[2], start+num_frames_chunk)
            result[:, start:endpt] = np.array(output).astype(self.dtype)
            temporal_basis[:, start:endpt] = np.array(temporal_basis_chunk).astype(self.dtype)
            start = endpt
            

        self.temporal_basis = temporal_basis
        return result  
    

    def temporal_crop_with_filter(self, frames):
        crop_data = self.temporal_crop_standardized(frames)
        crop_data_r = crop_data.reshape((-1, crop_data.shape[2]), order=self.order)
        temporal_basis_crop = self.spatial_basis.T.dot(crop_data_r) #Given standardized data, this is the linear subspace projection (spatial basis has orthonormal cols)
        spatial_r = self.spatial_basis.reshape((self.shape[0], self.shape[1], self.spatial_basis.shape[1]), order=self.order)
        return np.array(filter_components(crop_data, spatial_r, temporal_basis_crop))
  

#@jit
def get_temporal_basis(D, spatial_basis, mean_img_r, std_img_r):
    #Get the relevant temporal component given the spatial basis: 
    spatial_basis = spatial_basis / std_img_r.transpose()
    spatialxdata = jnp.matmul(spatial_basis.transpose(), D) 
    spatialxmean = jnp.matmul(spatial_basis.transpose(), mean_img_r)
    diff = spatialxdata - spatialxmean
    
    return diff

    

@partial(jit)
def V_projection_routine(M, D, spatial_basis, mean_img_r, std_img_r):
    M = M/std_img_r
    temporal_basis_chunk = get_temporal_basis(D, spatial_basis, mean_img_r, std_img_r)
    MD = jnp.matmul(M, D)
    Ma1 = jnp.matmul(M, spatial_basis)
    Ma1a2 = jnp.matmul(Ma1, temporal_basis_chunk)
    M_mean = jnp.matmul(M, mean_img_r)

    output = MD - Ma1a2 - M_mean


    return temporal_basis_chunk, output

  
@partial(jit)
def filter_components(data, spatial_r, temporal_basis):
    subt = jnp.tensordot(spatial_r, temporal_basis, axes=(2, 0))
    return data - subt
    
