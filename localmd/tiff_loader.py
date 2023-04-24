import os
import pathlib
import sys
import math
import tifffile

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
import jax.dlpack
import torch.utils.dlpack
import functools
from functools import partial
import torch
import torch_sparse
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

import math

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
    return batch[0]

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
    def __init__(self, filename, batch_size, frame_corrector=None):
        self.filename = filename
        self.shape = self._get_shape()
        self.chunks = math.ceil(self.shape[2]/batch_size)
        self.batch_size = batch_size
        self.frame_corrector = frame_corrector
        
        
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
        if self.frame_corrector is not None:
            data = self.frame_corrector.register_frames(data.astype("float32"))
        else:
            data = data
        return data
    



class tiff_loader():
    def __init__(self, filename, dtype='float32', center=True, normalize=True, background_rank=15, batch_size=2000, order="F", num_workers = None, pixel_batch_size=5000, frame_corrector_obj = None, num_samples = 8):
        '''
        Inputs: 
            filename: string. describing path to multipage tiff file to be denoised
            dtype: np.dtype. intended format of data
            center: bool. whether or not to center the data before denoising
            normalize: bool. whether or not noise normalize the data
            background_rank: int. we run an approximate truncated svd on the full FOV of the data, of rank 'background_rank'. We subtract this from the data before running the core matrix decomposition compression method
            batch_size: max number of frames to load into memory (CPU and GPU) at a time
            order: the order (either "C" or "F") in which we reshape 2D data into 3D videos and vice versa
            num_workers: int, keep it at 0 for now. Number of workers used in pytorch dataloading. Experimental and best kept at 0. 
            pixel_batch_size: int. maximum number of pixels of data we load onto GPU at any point in time
            frame_corrector_obj: jnormcorre frame corrector object. This is used to register loaded data on the fly. So in a complete pipeline, like the maskNMF pipeline, we can load data, correct it on the fly, and compress it. Avoids the need to explicitly rewrite the data onto disk during registration. 
            num_samples: int. when we estimate mean and noise variance, we take 8 samples of the data, each sample has 'batch_size' number of continuous frames. If there are fewer than num_samples * batch_size frames in the dataset, we just sequentially load the entire dataset to get these estimates. 
        
        
        '''
        with tifffile.TiffFile(filename) as tffl: 
            if len(tffl.pages) == 1: 
                raise ValueError("PMD does not accept single-page tiff datasets. Instead, pass your raw through the pipeline starting from the motion correction step.")
        self.order = order
        self.filename = filename
        self.dtype = dtype
        self.shape = self._get_shape()
        self._estimate_batch_size(frame_const=batch_size)
        self.pixel_batch_size=pixel_batch_size
        self.frame_corrector = frame_corrector_obj
        self.num_samples = num_samples
        
        #Define the tiff loader
        self.tiff_dataobj = tiff_dataset(self.filename, self.batch_size, frame_corrector = self.frame_corrector)
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
            frames: a list of frame values (for e.g. [1,5,2,7,8]) 
        Returns: 
            A (potentially motion-corrected) array containing these frames from the tiff dataset 
        '''
        
        if self.frame_corrector is not None:
            frame_length = len(frames) 
            result = np.zeros((self.shape[0], self.shape[1], frame_length))
            
            value_points = list(range(0, frame_length, self.batch_size))
            if value_points[-1] > frame_length - self.batch_size and frame_length > self.batch_size:
                value_points[-1] = frame_length - self.batch_size
            for k in value_points:
                start_point = k
                end_point = min(k + self.batch_size, frame_length)
                curr_frames = frames[start_point:end_point]
                x = tifffile.imread(self.filename, key=curr_frames).astype("float32")
                result[:, :, start_point:end_point] = np.array(self.frame_corrector.register_frames(x)).transpose(1,2,0)
            return result
        else:
            return tifffile.imread(self.filename, key=frames).transpose(1,2,0).astype(self.dtype)

        
        
        
    def _initialize_all_normalizers(self):
        '''
        Constructs mean image and normalization image
        '''
        display("Computing Video Statistics")
        if self.center and self.normalize:
            
            if self.shape[2] > self.batch_size * self.num_samples:
                results = self._calculate_mean_and_normalizer_sampling()
            else:
                results = self._calculate_mean_and_normalizer()
            self.mean_img = results[0]
            self.std_img = results[1]
        else:
            raise ValueError("Method now requires normalization and centering")
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

    
    def _calculate_mean_and_normalizer_sampling(self):
        '''
        This function takes a full pass through the dataset and calculates the mean and noise variance at the 
        same time, to avoid doing them separately
        '''
        frame_constant = 1024
        display("Calculating mean and noise variance via sampling")
        overall_mean = np.zeros((self.shape[0], self.shape[1]))
        overall_normalizer = np.zeros((self.shape[0], self.shape[1]), dtype=self.dtype)
        num_frames = self.shape[2]
        
        divisor = math.ceil(math.sqrt(self.pixel_batch_size))
        dim1_range_start_pts = np.arange(0, self.shape[0] - divisor, divisor)
        dim1_range_start_pts = np.concatenate([dim1_range_start_pts, [self.shape[0] - divisor]], axis = 0)
        dim2_range_start_pts = np.arange(0, self.shape[1] - divisor, divisor)
        dim2_range_start_pts = np.concatenate([dim2_range_start_pts, [self.shape[1] - divisor]], axis = 0)
        
        elts_to_sample = list(range(0, self.shape[2], frame_constant))
        if elts_to_sample[-1] > self.shape[2] - frame_constant and elts_to_sample[-1] > 0:
            elts_to_sample[-1] = self.shape[2] - frame_constant
        elts_used = np.random.choice(elts_to_sample, min(self.num_samples, len(elts_to_sample)), replace=False)
        elts_used = np.random.choice(elts_to_sample, self.num_samples, replace=False)
        frames_actually_used = 0
        for i in elts_used:
            start_pt_frame = i
            end_pt_frame = min(i + frame_constant, self.shape[2])
            frames_actually_used += end_pt_frame - start_pt_frame
            
            data = np.array(self.temporal_crop([i for i in range(start_pt_frame, end_pt_frame)]))
            mean_value_net = np.zeros((self.shape[0], self.shape[1]))
            normalizer_net = np.zeros((self.shape[0], self.shape[1]))
            for step1 in dim1_range_start_pts:
                for step2 in dim2_range_start_pts:
                    crop_data = data[step1:step1+divisor, step2:step2+divisor, :]
                    mean_value, noise_est_2d = get_mean_and_noise(crop_data, num_frames)
                    mean_value_net[step1:step1+divisor, step2:step2+divisor] = np.array(mean_value)
                    normalizer_net[step1:step1+divisor, step2:step2+divisor] = np.array(noise_est_2d)
                    
            overall_mean += mean_value_net
            overall_normalizer += normalizer_net
        overall_mean = overall_mean * (num_frames / frames_actually_used)
        overall_normalizer /= len(elts_to_sample)
        overall_normalizer[overall_normalizer==0] = 1
        display("Finished mean and noise variance")
        return overall_mean.astype(self.dtype), overall_normalizer

    
    def _calculate_mean_and_normalizer(self):
        '''
        This function takes a full pass through the dataset and calculates the mean and noise variance at the 
        same time, to avoid doing them separately
        '''
        frame_constant = 1024
        display("Calculating mean and noise variance")
        overall_mean = np.zeros((self.shape[0], self.shape[1]))
        overall_normalizer = np.zeros((self.shape[0], self.shape[1]), dtype=self.dtype)
        num_frames = self.shape[2]
        
        divisor = math.ceil(math.sqrt(self.pixel_batch_size))
        dim1_range_start_pts = np.arange(0, self.shape[0] - divisor, divisor)
        dim1_range_start_pts = np.concatenate([dim1_range_start_pts, [self.shape[0] - divisor]], axis = 0)
        dim2_range_start_pts = np.arange(0, self.shape[1] - divisor, divisor)
        dim2_range_start_pts = np.concatenate([dim2_range_start_pts, [self.shape[1] - divisor]], axis = 0)
        
        elts_used = list(range(0, self.shape[2], frame_constant))
        if elts_used[-1] > self.shape[2] - frame_constant and elts_used[-1] > 0:
            elts_used[-1] = self.shape[2] - frame_constant
        

        for i in elts_used:
            start_pt_frame = i
            end_pt_frame = min(i + frame_constant, self.shape[2])
            data = np.array(self.temporal_crop([i for i in range(start_pt_frame, end_pt_frame)]))

            data = np.array(data) 
            mean_value_net = np.zeros((self.shape[0], self.shape[1]))
            normalizer_net = np.zeros((self.shape[0], self.shape[1]))
            for step1 in dim1_range_start_pts:
                for step2 in dim2_range_start_pts:
                    crop_data = data[step1:step1+divisor, step2:step2+divisor, :]
                    mean_value, noise_est_2d = get_mean_and_noise(crop_data, num_frames)
                    mean_value_net[step1:step1+divisor, step2:step2+divisor] = np.array(mean_value)
                    normalizer_net[step1:step1+divisor, step2:step2+divisor] = np.array(noise_est_2d)
                    
            overall_mean += mean_value_net
            overall_normalizer += normalizer_net
        overall_normalizer /= len(elts_used)
        overall_normalizer[overall_normalizer==0] = 1
        display("Finished mean and noise variance")
        return overall_mean.astype(self.dtype), overall_normalizer
    
    
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
   

    def V_projection_torch(self, M):
        '''
        M is a tuple, consisting of two elements
            Element 0: A scipy.sparse.coo_matrix of dimensions (d, R)
            Element 1: A (R, R) np.ndarray. 
        '''
        if torch.cuda.is_available():
            device='cuda'
        else:
            device='cpu'
            
        torch_dtype=torch.float
        U_mat = M[0]
        Inv_mat = M[1]
        
        result = torch.zeros((U_mat.shape[1], self.shape[2]), dtype=torch_dtype).to(device)
        mean_img_r = torch.Tensor(self.mean_img.reshape((-1, 1), order=self.order)).to(device)
        std_img_r = torch.Tensor(self.std_img.reshape((1, -1), order=self.order)).to(device)
        spatial_basis_torch = torch.Tensor(np.array(self.spatial_basis)).to(device)
        
        #Normalize the columns of the sparse projection matrix U_mat ahead of time (this is equivalent to normalizing the rows of each input data point)
        diagonal_elements = self.std_img.reshape((-1,), order=self.order)
        diagonal_elements[diagonal_elements == 0] = 1
        diagonal_elements = np.reciprocal(diagonal_elements)
        diagonal_matrix = scipy.sparse.diags(diagonal_elements, shape=(U_mat.shape[0], U_mat.shape[0])).tocsr()
        U_mat_normalized = diagonal_matrix.dot(U_mat)
        U_mat_torch = torch_sparse.tensor.from_scipy(U_mat_normalized.transpose()).type(torch_dtype).to(device)
        temporal_basis = torch.zeros((self.spatial_basis.shape[1], self.shape[2]), dtype=torch_dtype)
        
        start = 0
        for i, data in enumerate(tqdm(self.loader), 0):
            #Convert data to tensor 
            start_time = time.time()
            
            D = data_reshape_jax(self.order, data) 
            D_torch = jax2torch(D, device)
            D_torch_time = time.time() - start_time
            
            start_time = time.time()
            temporal_basis_chunk,output = V_projection_routine(U_mat_torch, D_torch, spatial_basis_torch, mean_img_r, std_img_r)
            routine_time = time.time() - start_time
            
            start_time = time.time()
            num_frames_chunk = output.shape[1]
            endpt = min(self.shape[2], start+num_frames_chunk)
            temporal_basis[:, start:endpt] = temporal_basis_chunk
            result[:, start:endpt] = output
            start = endpt
            write_time = time.time() - start_time
            
            display("D_time {} routine_time {} writing_time {}".format(D_torch_time, routine_time, write_time))
            
            
        projected_V = Inv_mat.dot(result.detach().cpu().numpy())
        self.temporal_basis = temporal_basis.detach().cpu().numpy()
        return projected_V

            

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
        display("RUNNING JAX")
        result = np.zeros((M.shape[0], self.shape[2]), dtype=self.dtype)
        mean_img_r = self.mean_img.reshape((-1, 1), order=self.order)
        std_img_r = self.std_img.reshape((1, -1), order=self.order)
        start = 0
        temporal_basis = np.zeros((self.spatial_basis.shape[1], self.shape[2]), dtype=self.dtype)

        start = 0
        for i, data in enumerate(tqdm(self.loader), 0):

            temporal_basis_chunk, output = V_projection_routine_jax(self.order, M, data, self.spatial_basis, mean_img_r, std_img_r)
            num_frames_chunk = output.shape[1]
            
            endpt = min(self.shape[2], start+num_frames_chunk)
            result[:, start:endpt] = np.array(output).astype(self.dtype)
            temporal_basis[:, start:endpt] = np.array(temporal_basis_chunk).astype(self.dtype)
            start = endpt


        self.temporal_basis = temporal_basis
        return result  

    
    #No longer used
    def temporal_crop_with_filter(self, frames):
        crop_data = self.temporal_crop_standardized(frames)
        crop_data_r = crop_data.reshape((-1, crop_data.shape[2]), order=self.order)
        temporal_basis_crop = self.spatial_basis.T.dot(crop_data_r) #Given standardized data, this is the linear subspace projection (spatial basis has orthonormal cols)
        spatial_r = self.spatial_basis.reshape((self.shape[0], self.shape[1], self.spatial_basis.shape[1]), order=self.order)
        return np.array(filter_components(crop_data, spatial_r, temporal_basis_crop))
  

#@jit
def get_temporal_basis_jax(D, spatial_basis, mean_img_r, std_img_r):
    #Get the relevant temporal component given the spatial basis: 
    spatial_basis = spatial_basis / std_img_r.transpose()
    spatialxdata = jnp.matmul(spatial_basis.transpose(), D) 
    spatialxmean = jnp.matmul(spatial_basis.transpose(), mean_img_r)
    diff = spatialxdata - spatialxmean
    
    return diff


def get_temporal_basis(D, spatial_basis, mean_img_r, std_img_r):
    #Get the relevant temporal component given the spatial basis: 
    spatial_basis_norm = spatial_basis / std_img_r.t()
    spatialxdata = torch.matmul(spatial_basis_norm.t(), D) 
    spatialxmean = torch.matmul(spatial_basis_norm.t(), mean_img_r)
    diff = spatialxdata - spatialxmean
    
    return diff




def V_projection_routine(M_std, D, spatial_basis, mean_img_r, std_img_r):
    temporal_basis_chunk = get_temporal_basis(D, spatial_basis, mean_img_r, std_img_r)
    MD = torch_sparse.matmul(M_std, D)
    Ma1 = torch_sparse.matmul(M_std, spatial_basis)
    Ma1a2 = torch.matmul(Ma1, temporal_basis_chunk)
    M_mean = torch_sparse.matmul(M_std, mean_img_r)
    
    output = MD
    output.sub_(Ma1a2)
    output.sub_(M_mean)
    
    return temporal_basis_chunk, output

@partial(jit, static_argnums=(0))
def V_projection_routine_jax(order, M, D, spatial_basis, mean_img_r, std_img_r):
    D = jnp.transpose(D, (1,2,0))
    D = jnp.reshape(D, (-1, D.shape[2]), order=order)
    M = M/std_img_r
    temporal_basis_chunk = get_temporal_basis_jax(D, spatial_basis, mean_img_r, std_img_r)
    MD = jnp.matmul(M, D)
    Ma1 = jnp.matmul(M, spatial_basis)
    Ma1a2 = jnp.matmul(Ma1, temporal_basis_chunk)
    M_mean = jnp.matmul(M, mean_img_r)

    output = MD - Ma1a2 - M_mean


    return temporal_basis_chunk, output


@partial(jit, static_argnums=(0))
def data_reshape_jax(order, D):
    D = jnp.transpose(D, (1,2,0))
    D = jnp.reshape(D, (-1, D.shape[2]), order=order)
    return D
    

  
@partial(jit)
def filter_components(data, spatial_r, temporal_basis):
    subt = jnp.tensordot(spatial_r, temporal_basis, axes=(2, 0))
    return data - subt
  
                
def jax2torch(x, device):
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x)).to(device)