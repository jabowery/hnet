Python translations of the matlab scripts in ../matlab

Install mamba:

https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html

To create the hnet environment:

$ mamba env create -f environment.yml

To update the environment:

$ vi environment.yml
$ mamba env update --file environment.yml --prune

In general, it is a good idea to get used to _not_ providing version numbers in the environment.yml so that mamba is free to do the work of version alignment.

The only exceptions are when the python version must be pinned and when packages that must be installed under pip aren't being aligned by mamba.
