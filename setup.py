from setuptools import setup, find_packages

setup(
    name='blsdata',
    version='0.9.9',
    description='Script for reading and processing .h5 BLS data files',
    url='https://github.com/GIGAluckman/BLS_processing',
    author='Andrey Voronov',
    author_email='andrey.voronov@univie.ac.at',

    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'h5py',
        'logging',
        ],
)