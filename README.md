# Manifoldron: Direct Space Partition via Manifold Discovery
This respository includes implementations on *Manifoldron: Direct Space Partition via Manifold Discovery* in which we propose a new type of machine learning models referred to as Manifoldron that directly derives decision boundaries from data and partitions the space via manifold structure discovery. Also, we systematically analyze the key characteristics of the Manifoldron including interpretability, manifold characterization capability, and its link to neural networks. The experimental results on 9 small and 11 large datasets demonstrate that the proposed Manifoldron performs competitively compared to the mainstream machine learning models.
<p align="center">
  <img width="480" src="https://user-images.githubusercontent.com/23077770/149367146-620b6d43-6741-4208-984d-2533b1eb24f0.png">
</p>
<p align="center">
  <em>Fig. 1 (a) Pipeline of the Manifoldron. (b) The Manifoldron key steps illustration.</em>
</p>

## Pre-requisites:
- Windows(runned on windows 10, can also run on Ubuntu with the required packages)
- Intell CPU(runned on 12 cores i7-8700 CPU @ 3.20GHZ)
- Python=3.7, numpy=1.18.5, pandas=0.25.3, scikit-learn=0.22.1, scipy=1.3.2, matplotlib=3.1.1.

## Folders 
**classification**: this directory contains the implementations on classfication tasks;<br/>
**regression**: this directory contains implementations on simple regression tasks;<br/>
**fancy_manifoldron**: this directory includes implementations on 3D complex manifolds.<br/>

## Dataset Preparation
All datasets are publicly available from python scikit-learn package, UCI machine learning repository, Kaggle, and Github. Most of the datasets can also directly obtain from our shared google drive. https://drive.google.com/drive/folders/14VHR8H7ucp0Loob1PS9yrgTtE9Jm0wsK?usp=sharing. <br/>
All datasets need to put under the 'data/' folder.

## Running Experiments
As a demo, below shows how different versions of the Manifoldron run on *tic-tac-toe* data.
```ruby
>> python manifoldron_base.py       # the base manifoldron
>> python manifoldron_bagging.py    # the manifoldron with feature bagging
>> python manifoldron_parallel.py   # the manifoldron with parallel computation
```

## Experiment Results
<p align="center">
  <em>Tab. 1 classification results on the Manifoldron and its counterparts.</em>
</p>
<p align="center">
  <img width="1000" src="https://user-images.githubusercontent.com/23077770/149402070-63cb13d4-0026-4e75-bbcd-d3be6c5a5553.png">
</p>


<p align="center">
  <img width="700" src="https://user-images.githubusercontent.com/23077770/149415337-d109d7a3-9961-4c8f-bdec-24d34aa001e3.png">
</p>
<p align="center">
  <em>Fig. 2 Complex simplices.</em>
</p>

<p align="center">
  <em>Tab. 2 Results on complex simplices.</em>
</p>
<p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/23077770/149415839-04f8ac01-a1a0-4aa5-9eda-4fe467281f15.png">
</p>



