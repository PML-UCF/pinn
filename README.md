[![DOI](https://zenodo.org/badge/165746996.svg)](https://zenodo.org/badge/latestdoi/165746996)
[![PyPI version](https://badge.fury.io/py/pml-pinn.svg)](https://badge.fury.io/py/pml-pinn)
# Physics-informed neural networks package
Welcome to the PML repository for physics-informed neural networks. We will use this repository to disseminate our research in this exciting topic. Links for some useful publications:
* [**Fleet Prognosis with Physics-informed Recurrent Neural Networks:**](https://arxiv.org/abs/1901.05512) This paper introduces a novel physics-informed neural network approach to prognosis by extending recurrent neural networks to cumulative damage models. We propose a new recurrent neural network cell designed to merge physics-informed and data-driven layers. With that, engineers and scientists have the chance to use physics-informed layers to model parts that are well understood (e.g., fatigue crack growth) and use data-driven layers to model parts that are poorly characterized (e.g., internal loads).

## Install

To install the stable version just do:
```
pip install pml-pinn
```

### Develop mode

To install in develop mode, clone this repository and do a pip install:
```
git clone https://github.com/PML-UCF/pinn.git
cd pinn
pip install -e .
```

## Citing this repository

Please, cite this repository using: 

    @misc{2019_pinn,
        author    = {F. A. C. Viana, R. G. Nascimento, Y. Yucesan, A. Dourado},
        title     = {Physics-informed neural networks package},
        month     = Aug,
        year      = 2019,
        doi       = {10.5281/zenodo.3356877},
        version   = {0.0.2},
        publisher = {Zenodo},
        url       = {https://github.com/PML-UCF/pinn}
        }
  The corresponding reference entry should look like:

      F. A. C. Viana, R. G. Nascimento, Y. Yucesan, A. Dourado, Physics-informed neural networks package, v0.0.2, Zenodo, https://github.com/PML-UCF/pinn, doi:10.5281/zenodo.3356877.
