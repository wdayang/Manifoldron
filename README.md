# Manifoldron: Direct Space Partition via Manifold Discovery
This respository includes implementations on *Manifoldron: Direct Space Partition via Manifold Discovery* in which we first propose a new type of machine learning models referred to as Manifoldron that directly derives decision boundaries from data and partitions the space via manifold structure discovery. Then, we systematically analyze the key characteristics of the Manifoldron including interpretability, manifold characterization capability, and its link to neural networks. The experimental results on 9 small and 11 large datasets demonstrate that the proposed Manifoldron performs competitively compared to the mainstream machine learning models.
<p align="center">
  <img width="480" src="https://user-images.githubusercontent.com/23077770/149367146-620b6d43-6741-4208-984d-2533b1eb24f0.png">
</p>
<p align="center">
  <em>Fig. 1 (a) Pipeline of the Manifoldron. (b) The Manifoldron key steps illustration.</em>
</p>

## Pre-requisites:
- Windows(runned on windows 10)
- Intell CPU(runned on 12 cores i7-8700 CPU @ 3.20GHZ)
- Python=3.7, numpy=1.18.5, pandas=0.25.3, scikit-learn=0.22.1, scipy=1.3.2, matplotlib=3.1.1.

## Files 
**classification**: this directory contains the implementations on classfication tasks;<br/>
**regression**: this directory contains implementations on simple regression tasks;<br/>
**fancy_manifoldron**: this directory includes implementations on 3D complex manifolds.<br/>

## Dataset Preparation
All datasets are publicly available from python scikit-learn package, UCI machine learning repository, Kaggle, and Github. Most of the data can also be directly obtained from our shared google drive. https://drive.google.com/drive/folders/14VHR8H7ucp0Loob1PS9yrgTtE9Jm0wsK?usp=sharing

## Experiment results
<p align="center">
  <img width="1000" src="https://user-images.githubusercontent.com/23077770/149400378-6c14057b-6b6f-47a2-b8c3-fcaa57e640fb.png">
</p>
<p align="center">
  <em>Fig. 2 classification results on the Manifoldron and its counterparts.</em>
</p>
