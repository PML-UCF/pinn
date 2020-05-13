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

* [**Physics-informed neural networks for missing physics estimation in cumulative damage models: a case study in corrosion fatigue:**](https://asmedigitalcollection.asme.org/computingengineering/article-abstract/doi/10.1115/1.4047173/1083614/Physics-informed-neural-networks-for-missing) In this contribution, we present a physics-informed neural network modeling approach for missing physics estimation in cumulative damage models. A hybrid approach is designed to merge physics-informed and data-driven layers within deep neural networks. The result is a cumulative damage model in which physics-informed layers are used to model relatively well-understood phenomenona and data-driven layers account for hard-to-model physics. A numerical experiment is used to present the main features of the proposed framework for damage accumulation. The test problem consists of predicting corrosion-fatigue of an Al 2024-T3 alloy used on panels of aircraft wings. Besides cyclic loading, panels are also subjected to saline corrosion. In this case, physics-informed layers implement the well known Walker's model for crack propagation, while data-driven layers are trained to compensate for the bias in damage accumulation due to the corrosion effects. The physics-informed neural network is trained using full observation of inputs (far-field loads, stress ratio and a corrosivity index – defined per airport) and very limited observation of outputs (crack length at inspection for only a small portion of the fleet). Results show that the physics-informed neural network is able to learn how to compensate for the missing physics of corrosion in the original fatigue model. Predictions from the hybrid model can be used in fleet management, for example, to prioritize inspection across the fleet or forecasting ahead of time the number of planes with damage above a threshold.

* [**A Physics-informed Neural Network for Wind Turbine Main Bearing Fatigue:**](http://www.phmsociety.org/node/2736) Unexpected main bearing failure on a wind turbine causes unwanted maintenance and increased operation costs (mainly due to crane, parts, labor, and production loss). Unfortunately, historical data indicates that failure can happen far earlier than the component design lives. Root cause analysis investigations have pointed to problems inherent from manufacturing as the major contributor, as well as issues related to event loads (e.g., startups, shutdowns, and emergency stops), extreme environmental conditions, and maintenance practices, among others. Altogether, the multiple failure modes and contributors make modeling the remaining useful life of main bearings a very daunting task. In this paper, we present a novel physics-informed neural network modeling approach for main bearing fatigue. The proposed approach is fully hybrid and designed to merge physics-informed and data-driven layers within deep neural networks. The result is a cumulative damage model where the physics-informed layers are used model the relatively well-understood physics (L10 fatigue life) and the data-driven layers account for the hard to model components (e.g., contribution due to poor greasing conditions).

* [**Fleet Prognosis with Physics-informed Recurrent Neural Networks:**](https://arxiv.org/abs/1901.05512) This paper introduces a novel physics-informed neural network approach to prognosis by extending recurrent neural networks to cumulative damage models. We propose a new recurrent neural network cell designed to merge physics-informed and data-driven layers. With that, engineers and scientists have the chance to use physics-informed layers to model parts that are well understood (e.g., fatigue crack growth) and use data-driven layers to model parts that are poorly characterized (e.g., internal loads).

