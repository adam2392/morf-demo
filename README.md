Manifold Oblique Random Forests Demonstration on Simulated and Example Datasets
===============================================================================

This project reproduces some of the simulation and example data results in the [MORF paper](https://arxiv.org/abs/1909.11799).

This will produce examples of:

1. simulation examples (see paper and notebooks for full details)
2. neural fragility seizure outcome classification
3. sEEG time series to classify movement from non-motor brain regions

Primarily, you should refer to the ``notebooks/`` to look at experiments
rendered.

System Requirements
===================
Generally to run the figure generation, one
simply needs a standard computer with enough RAM.
Minimally to generate the figures, probably a computer 
with 2GB RAM is sufficient.

We ran tests on computer with the following:

RAM: 16+ GB
CPU: 4+ cores, i7 or equivalent

Software: Mac OSX or Linux Ubuntu 18.04+. One should use Python3.6+.

Installation Guide
==================

Setup environment from pipenv. The `Pipfile` contains the Python 
libraries needed to run the figure generation in [notebook](./notebooks/plot_morf_clf_comparisons.ipynb).


    pipenv install --dev
    
    # use pipenv to install private repo
    pipenv install -e git+git@github.com:adam2392/eztrack
    
    # or
    pipenv install -e /Users/adam2392/Documents/eztrack

    # if dev versions are needed
    pipenv install https://api.github.com/repos/mne-tools/mne-bids/zipball/master --dev
    pipenv install https://api.github.com/repos/mne-tools/mne-python/zipball/master --dev

Instructions for Use
====================
Run the notebook from beginning to end to generate figures, 
by pointing the path to the `data/` folder here. To setup an ipykernel 
to expose your Python virtual environment to the Jupyter kernel:

    make ipykernel

In order to build ReRF, we use a custom version that is at https://github.com/neurodata/SPORF/pull/353.
Build the C++ code from that PR, and then run pip install.

    pip install -e <SPORF_DIR>

Neural fragility dataset
------------------------

See paper for all details: https://www.biorxiv.org/content/10.1101/862797v4

sEEG motor movement in non-motor brain region dataset
----------------------------------------------------- 

See the following papers for more information.

* [Kerr MSD, Sacré P, Kahn K, Park HJ, Johnson M, Lee J, Thompson S, Bulacio J, Jones J, González-Martínez J, Liégeois-Chauvel C, Sarma SV, Gale JT. The Role of Associative Cortices and Hippocampus during Movement Perturbations. Front Neural Circuits. 2017 Apr 19;11:26. doi: 10.3389/fncir.2017.00026. PMID: 28469563; PMCID: PMC5395558.](https://www.frontiersin.org/articles/10.3389/fncir.2017.00026/full#:~:text=These%20regions%20are%20involved%20during,a%20new%20plan%20to%20compensate)
* [Breault MS, Fitzgerald ZB, Sacré P, Gale JT, Sarma SV, González-Martínez JA. Non-motor Brain Regions in Non-dominant Hemisphere Are Influential in Decoding Movement Speed. Front Neurosci. 2019 Jul 16;13:715. doi: 10.3389/fnins.2019.00715. PMID: 31379476; PMCID: PMC6660252.](https://pubmed.ncbi.nlm.nih.gov/31379476/)
