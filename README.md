AnnapuRNA
================


<!-- TOC START min:1 max:6 link:true asterisk:false update:true -->
- [About](#about)
- [Installation](#installation)
  - [Uninstallation](#uninstallation)
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
