from setuptools import setup, find_packages
from os import path

with open("README.md", "r") as fh:
    long_description = fh.read()

min_jax_version = "0.3.25"

setup(
    name='localmd',
    description="Method for compressing neuroimaging data using spatially localized low-rank matrix decompositions",
    author='Amol Pasarkar',
    version="0.0.4",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy", "tifffile", "torch", "scipy", "jupyterlab", "tqdm", "jax>={}".format(min_jax_version), "jaxlib>={}".format(min_jax_version)],
    python_requires='>=3.8',
)