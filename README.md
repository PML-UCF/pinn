[![DOI](https://zenodo.org/badge/165746996.svg)](https://zenodo.org/badge/latestdoi/165746996)
[![PyPI version](https://badge.fury.io/py/pml-pinn.svg)](https://badge.fury.io/py/pml-pinn)
# Physics-informed neural networks package
Welcome to the PML repository for physics-informed neural networks. We will use this repository to disseminate our research in this exciting topic.

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
        author    = {Felipe A. C. Viana and Renato G. Nascimento and Yigit Yucesan and Arinan Dourado},
        title     = {Physics-informed neural networks package},
        month     = Aug,
        year      = 2019,
        doi       = {10.5281/zenodo.3356877},
        version   = {0.0.2},
        publisher = {Zenodo},
        url       = {https://github.com/PML-UCF/pinn}
        }
  The corresponding reference entry should look like:

      F. A. C. Viana, R. G. Nascimento, Y. Yucesan, and A. Dourado, Physics-informed neural networks package, v0.0.2, Aug. 2019. doi:10.5281/zenodo.3356877, URL https://github.com/PML-UCF/pinn.

## Publications

* [**A Physics-informed Neural Network for Wind Turbine Main Bearing Fatigue:**](http://www.phmsociety.org/node/2736) Unexpected main bearing failure on a wind turbine causes unwanted maintenance and increased operation costs (mainly due to crane, parts, labor, and production loss). Unfortunately, historical data indicates that failure can happen far earlier than the component design lives. Root cause analysis investigations have pointed to problems inherent from manufacturing as the major contributor, as well as issues related to event loads (e.g., startups, shutdowns, and emergency stops), extreme environmental conditions, and maintenance practices, among others. Altogether, the multiple failure modes and contributors make modeling the remaining useful life of main bearings a very daunting task. In this paper, we present a novel physics-informed neural network modeling approach for main bearing fatigue. The proposed approach is fully hybrid and designed to merge physics-informed and data-driven layers within deep neural networks. The result is a cumulative damage model where the physics-informed layers are used model the relatively well-understood physics (L10 fatigue life) and the data-driven layers account for the hard to model components (e.g., contribution due to poor greasing conditions).

* [**Fleet Prognosis with Physics-informed Recurrent Neural Networks:**](https://arxiv.org/abs/1901.05512) This paper introduces a novel physics-informed neural network approach to prognosis by extending recurrent neural networks to cumulative damage models. We propose a new recurrent neural network cell designed to merge physics-informed and data-driven layers. With that, engineers and scientists have the chance to use physics-informed layers to model parts that are well understood (e.g., fatigue crack growth) and use data-driven layers to model parts that are poorly characterized (e.g., internal loads).

