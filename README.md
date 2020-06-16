AnnapuRNA
================


<!-- TOC START min:1 max:6 link:true asterisk:false update:true -->
- [About](#about)
- [Installation](#installation)
  - [Uninstallation](#uninstallation)
- [Usage](#usage)
  - [Quick start](#quick-start)
  - [Full pipeline](#full-pipeline)
- [Software used](#software-used)
<!-- TOC END -->


# About


# Installation

Recommended way of AnnapuRNA installation and running is via conda environment.

1. Install conda.  Please refer to [conda manual](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and install conda version according to your operating system. Please use Python2 version.
2. Clone AnnapuRNA repository: `git clone --depth=1 git@github.com:filipsPL/annapurna.git` or [fetch a zip package](https://github.com/filipsPL/annapurna/archive/master.zip).
3. Go to the AnnapuRNA directory (typically `cd annapurna` under linux) and restore the conda environment from the yml file `conda env create -f conda-environment.yml` (the complete AnnapuRNA conda environment needs ~1.5 GB of free disk space).

## Uninstallation

(if you no longer need the AnnapuRNA :frowning:)

1. Remove the directory with the AnnapuRNA code
2. remove conda environment: `conda remove --name annapurna --all`.
3. To verify that the environment was removed, in your terminal window run `conda info --envs`

# Usage

## Quick start


## Full pipeline




# Software used

During development of the AnnapuRNA we used a number of freely available packages for scietific computations. Here we acknowledge and thanks:

- [Biopython](https://biopython.org/) - a set of freely available tools for biological computation written in Python
- [openbabel](https://github.com/openbabel/openbabel) - a chemical toolbox designed to speak the many languages of chemical data
- [numpy](https://numpy.org/) - a fundamental package for scientific computing with Python
- [pandas](https://pandas.pydata.org/) - a fast, powerful, flexible and easy to use open source data analysis and manipulation tool
- Machine learning:
  - [scikit-learn](https://scikit-learn.org/stable/) - Machine Learning in Python
  - [h2o](https://www.h2o.ai/products/h2o/) from h2o.ai - version [3.9.1.3501](http://h2o-release.s3.amazonaws.com/h2o/master/3501/index.html) - a fully open source, distributed in-memory machine learning platform with linear scalability.
- [rna-tools](https://github.com/mmagnus/rna-tools) (formerly: rna-pdb-tools) by @mmagnus -  a toolbox to analyze sequences, structures and simulations of RNA
- [seaborn](https://seaborn.pydata.org/) - statistical data visualization
