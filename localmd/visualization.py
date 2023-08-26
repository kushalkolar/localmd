import scipy
import scipy.sparse
import numpy as np
import functools
from functools import partial
import time
import tifffile



def temporal_crop(self, frames):
        '''
        Input: 
            frames: a list of frame values (for e.g. [1,5,2,7,8]) 
        Returns: 
            A (potentially motion-corrected) array containing these frames from the tiff dataset with shape (d1, d2, T) where d1, d2 are FOV dimensions, T is 
            number of frames selected
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
                x = self.dataset.get_frames(curr_frames).astype(self.dtype).transpose(2,0,1)
                result[:, :, start_point:end_point] = np.array(self.frame_corrector.register_frames(x)).transpose(1,2,0)
            return result
        else:
            return self.dataset.get_frames(frames).astype(self.dtype)

        
def temporal_crop(dataset, frames, frame_corrector = None, batch_size = 100):
    if frame_corrector is not None:
        frame_length = len(frames) 
        result = np.zeros((dataset.shape[0], dataset.shape[1], frame_length))

        value_points = list(range(0, frame_length, batch_size))
        if value_points[-1] > frame_length - batch_size and frame_length > batch_size:
            value_points[-1] = frame_length - batch_size
        for k in value_points:
            start_point = k
            end_point = min(k + batch_size, frame_length)
            curr_frames = frames[start_point:end_point]
            x = dataset.get_frames(curr_frames).astype("float32").transpose(2,0,1)
            result[:, :, start_point:end_point] = np.array(frame_corrector.register_frames(x)).transpose(1,2,0)
            return result
    else:
        return dataset.get_frames(frames).astype("float32")

def generate_PMD_comparison_triptych(dataset, frames, U, R, s, V, mean_img, std_img, data_order, data_shape, dim1_interval, dim2_interval, frame_corrector=None):
    
    #Step 1: Prune the sparse U matrix 
    d1, d2 = data_shape[0], data_shape[1]
    indices_r = np.arange(d1*d2).reshape((d1, d2), order=data_order)
    
    indices_cropped = indices_r[dim1_interval[0]:dim1_interval[1], dim2_interval[0]:dim2_interval[1]].reshape((-1,), order=data_order)
    d1_p, d2_p = dim1_interval[1] - dim1_interval[0], dim2_interval[1] - dim2_interval[0]
    U = U[indices_cropped, :]
    V_crop = V[:, frames]
    sV = s[:, None] * V_crop
    RsV = R.dot(sV)
    PMD_movie = U.tocsr().dot(RsV)
    PMD_movie = PMD_movie.reshape((d1_p, d2_p, -1), order = data_order)
    
    #Rescale the PMD movie to match the raw movie (this is important for doing the comparisons)
    PMD_movie = PMD_movie * std_img[dim1_interval[0]:dim1_interval[1], dim2_interval[0]:dim2_interval[1], None] + mean_img[dim1_interval[0]:dim1_interval[1], dim2_interval[0]:dim2_interval[1], None]
    
    original_movie = temporal_crop(dataset, frames, frame_corrector)
    original_movie = original_movie[dim1_interval[0]:dim1_interval[1], dim2_interval[0]:dim2_interval[1], :]
    
    overall_result = np.zeros((original_movie.shape[0], original_movie.shape[1]*3, original_movie.shape[2]))
    overall_result[:, :PMD_movie.shape[1], :] = original_movie
    overall_result[:, PMD_movie.shape[1]:PMD_movie.shape[1]*2, :] = PMD_movie
    overall_result[:, PMD_movie.shape[1]*2:, :] = original_movie - PMD_movie

    return overall_result
