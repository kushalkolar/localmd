# localmd
This is a highly-accelerated implementation of [PMD](https://www.biorxiv.org/content/10.1101/334706v3.full.pdf), a local matrix decomposition method for compressing and denoising functional neuroimaging data. This implementation is optimized for execution on accelerators (GPUs). 

## Installation
Currently, this code is supported for Linux operating systems with python version at least 3.8. We primarily use [JAX](https://github.com/google/jax) for fast CPU/GPU/TPU execution and Just-In-Time compilation; see the Google installation instructions on how to install the appropriate version of JAX and JAXLIB for your specific hardware system. We require: 

```
python>=3.8
jax>=0.3.25
jaxlib>=0.3.25
```

To install this repository into your python environment from the source code, do the following (this approach is recommended for now): 
```
#Step 1: Install the appropriate version of jax for your hardware system 

#Step 2: Run the below lines of code
git clone https://github.com/apasarkar/localmd.git
cd localmd
pip install -e .
```


## Low-Rank Decomposition Documentation
In our demo script (localmd/demos), you will notice that we save the results of our low-rank matrix decomposition in a .npz file. This is our standard representation for saving the compressed results. We use the following:

- `fov_shape` : Tuple storing the shape of the imaging FOV (since the matrix representation of the decomposition uses a single, unraveled spatial dimension).
- `fov_order`: Either "C" or "F". Indicates how to reshape each column of the spatial representation of the data to go from a flattened 1D representation to a 2D image.
- `U_data` :  Data array of the sparse spatial component matrix. 
- `U_indices` : Index array of the sparse spatial component matrix.
- `U_indptr` : Index pointer array of the sparse spatial component matrix.
- `U_shape` : Shape of the sparse spatial component matrix.
- `U_format` : Record indicating the type of sparse representation used to store the spatial component matrix. Currently, this will always be compressed sparse row (CSR).
- `R` : Mixing weights to recover the orthonormal spatial basis (ie. left singular vectors) when left multiplied by the sparse spatial component matrix.
- `s` : Vector of singular values (ie. the diagonal of `S`). Note we only store the diagonal values here. 
- `Vt` : Orthonormal temporal basis (ie. right singular vectors).
- `mean_img` : The mean image of the dataset
- `std_img` : The noise variance image of every pixel of the dataset

Denoting the uncompressed output matrix as `Y_hat`, these component matrices form the factorization `Y_hat = [UR]SVt`. The sparse-dense matrix product `[UR]` is the orthonormal left singular vector matrix `U` from traditional SVD notation and the remaining matrices `s, Vt` are named in accordance with traditional SVD notation. 

To load the spatial matrix from this .npz file in python, do the following:
```
data = np.load(your_compressed_filename, allow_pickle=True)
U = scipy.sparse.csr_matrix(
    (data['U_data'], data['U_indices'], data['U_indptr']),
    shape=data['U_shape']
).tocoo()
V = data['Vt']
R = data['R']
s = data['s']
mean_img = data['mean_img']
std_img = data['noise_var_img']
data_shape = (data['fov_shape'][0], data['fov_shape'][1], V.shape[1])
data_order = data['fov_order'].item()
```

In the most recent commits we now provide a class (PMDArray) which allows you to interact with the PMD decomposition using array-like functionality (things like PMDArray[:, :, 40] to load the 40-th frame of your movie or PMDArray[20:30, 20:40, :] to spatially crop and interact with small subsets of your data). Of course running something like PMDArray[:, :, :] will expand out the full movie into main memory (which will overwhelm your system if your original data, casted to np.float32, is too large for your RAM). See the official_demo.ipynb for more details. 

## Parameter Documentation
For users of this method, there are primarily 3 parameters to modify: 

- ``block_height, block_width`` : We break the FOV of the data into overlapping blocks of dimensions (b1, b2). We find that blocksizes of roughly (20, 20) work well for most data. In general, the blocksize should be large enough to completely fill the largest somatic component in the data (of course, many datasets do not have somatic components; this is a rough reference point).
- ``frames_to_init`` : We begin the method by finding a low-rank (conservative) estimate of the linear subspace in which the signal resides. We use ``frames_to_init`` frames, sampled at time points throughout the movie to compute this spatial basis. 


## Custom Dataformat Support
The localmd package comes with support for multi-page tiff files. It is also extremely easy to add support for your own custom dataformats (these dataformats can consist of single file or multiple files). Simply provide a concrete implementation of the PMDDataset abstract class (see localmd/dataset.py). This implementation only requires two basic functions: (1) A function to return the shape (dimensions) of the data and (2) A function to return arbitrary frames of the data (see ``def get_frames``). 