[![PyPI version](https://badge.fury.io/py/pml-pinn.svg)](https://badge.fury.io/py/pml-pinn)
# Physics-informed neural networks
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
