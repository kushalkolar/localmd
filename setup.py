import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="localmd",
    version="0.0.1",
    description="Method for compressing calcium imaging data using standard SVD-style compressiono",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
    ),
)