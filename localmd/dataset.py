import numpy as np
import tifffile
from abc import ABC, abstractmethod

class PMDDataset(ABC):
    
    @property
    @abstractmethod
    def shape(self):
        '''
        This property should return the shape of the dataset, in the form: (d1, d2, T) where d1 and d2 are the field of view dimensions and T is the number of frames
        '''
        pass
    
    @abstractmethod
    def get_frames(self, frames):
        '''
        This function should take as input a list of integer-valued indices which describe the frames which need to be obtained from the dataset. 
        These indices should be given in python-indexing convention (i.e. an index with value of 0 refers to frame 1 of the dataset). 
        Input: 
            frames: list. integer indices
        Output: 
            movie_chunk: array-like object (here, np.ndarray) with dimensions (d1, d2, T) where (d1, d2) are the imaging field of view dimensions and T is the number of frames which were requested (the length of "frames")
        '''
        pass



class MultipageTiffDataset(PMDDataset):
    def __init__(self, filename):
        self.filename = filename
    
    @property
    def shape(self):
        return self._compute_shape(self.filename)
    
    def _compute_shape(self, filename):
        with tifffile.TiffFile(self.filename) as tffl:
            num_frames = len(tffl.pages)
            for page in tffl.pages[0:1]:
                image = page.asarray()
                x, y = page.shape
        return (x,y,num_frames)
    
    def get_frames(self, frames):
        '''
        Input: 
            frames: a list of frames to load
        Output: 
            np.ndarray of dimensions (d1, d2, T) where (d1, d2) are the FOV dimensions and T is the number of frames which have been loaded
        '''
        return tifffile.imread(self.filename, key=frames).transpose(1, 2, 0)
        
