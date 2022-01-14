# Manifoldron: Direct Space Partition via Manifold Discovery
[Colab Demo](https://colab.research.google.com/drive/1fK4OD27BYvmVdOorjartuf1CcaQe5PPK?usp=sharing) | [ArXiv]() | [Datasets](https://drive.google.com/drive/folders/14VHR8H7ucp0Loob1PS9yrgTtE9Jm0wsK?usp=sharing)<br/>
This respository includes implementations on *Manifoldron: Direct Space Partition via Manifold Discovery* in which we propose a new type of machine learning models referred to as Manifoldron that directly derives decision boundaries from data and partitions the space via manifold structure discovery. The experimental results on 9 small and 11 large datasets demonstrate that the proposed Manifoldron performs competitively compared to the mainstream machine learning models.
<p align="center">
  <img width="320" src="https://github.com/wdayang/Manifoldron/blob/main/figures/Manifoldron_gif.gif">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img width="420" src="https://user-images.githubusercontent.com/23077770/149440089-e31072ed-1f42-49aa-8590-155236223a0a.png">
</p>
<p align="center">
  <em>Fig. 1 The key steps and the pipeline of the Manifoldron.</em>
</p>

## Pre-requisites:
- Windows (runned on windows 10, can also run on Ubuntu with the required packages)
- Intell CPU (runned on 12 cores i7-8700 CPU @ 3.20GHZ)
- Python=3.7 (Anaconda), numpy=1.18.5, pandas=0.25.3, scikit-learn=0.22.1, scipy=1.3.2, matplotlib=3.1.1.

## Folders 
**classification**: this directory contains the implementations on classfication tasks;<br/>
**regression**: this directory contains implementations on simple regression tasks;<br/>
**fancy_manifoldron**: this directory includes implementations on 3D complex manifolds.<br/>

## Dataset Preparation
All datasets are publicly available from python scikit-learn package, UCI machine learning repository, Kaggle, and Github: [circle](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make\_circles.html), [glass](https://archive.ics.uci.edu/ml/datasets/glass+identification), [ionosphere](https://archive.ics.uci.edu/ml/datasets/ionosphere), [iris](https://scikit-learn.org/stable/auto\_examples/datasets/plot\_iris\_dataset.html),  [moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make\_moons.html), [parkinsons](https://archive.ics.uci.edu/ml/datasets/parkinsons), [seeds](https://archive.ics.uci.edu/ml/datasets/seeds), [spirals](https://github.com/Swarzinium-369/Two\_Spiral\_Problem), [wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load\_wine.html), 
[banknote](https://archive.ics.uci.edu/ml/datasets/banknote+authentication),
[breast](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)),
[chess](https://archive.ics.uci.edu/ml/datasets/Chess+(King-Rook+vs.+King-Pawn)),
[drug](https://archive.ics.uci.edu/ml/datasets/Drug+consumption+\%28quantified\%29),
[letRecog](https://archive.ics.uci.edu/ml/datasets/Letter+Recognition),
[magic04](https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope),
[nursery](https://archive.ics.uci.edu/ml/datasets/nursery),
[satimage](https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite)),
[semeion](https://archive.ics.uci.edu/ml/datasets/semeion+handwritten+digit),
[tic-tac-toe](https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame), and 
[usps5](https://www.kaggle.com/bistaumanga/usps-dataset).
Most datasets can also be directly obtained from our shared google drive. https://drive.google.com/drive/folders/14VHR8H7ucp0Loob1PS9yrgTtE9Jm0wsK?usp=sharing. <br/>
All datasets need to put under the 'classification/data/' folder to run the Manifoldron on specific datasets.

## Running Experiments
### Classification
As a demo, the below scripts show how different versions of the Manifoldron run on *tic-tac-toe* data.
```ruby
>> python manifoldron_base.py       # the base manifoldron
>> python manifoldron_bagging.py    # the manifoldron with feature bagging
>> python manifoldron_parallel.py   # the manifoldron with parallel computation
```
If you would like to run the Manifoldron on other representative classification datasets, please go to 'classification/' folder and run cooresponding .py file. We also provide a classification demonstration on the 2D spiral dataset: https://colab.research.google.com/drive/1fK4OD27BYvmVdOorjartuf1CcaQe5PPK?usp=sharing
#### Experiment Results
<p align="center">
  <em>Tab. 1 Classification results of the Manifoldron and its competitors.</em>
</p>
<p align="center">
  <img width="1000" src="https://user-images.githubusercontent.com/23077770/149402070-63cb13d4-0026-4e75-bbcd-d3be6c5a5553.png">
</p>

### Classification on fancy manifolds
Please first go to 'fancy_manifoldron/manifold generation/' to generate complex manifolds in matlab, copy the generated dataset to 'fancy_manifoldron/comparision/', then run the .py file for classification. 

```ruby
>> matlab PlotDoubleCurlySpiral.m            # generate complex manifolds
>> copy [generated txt data path] [fancy_manifoldron/comparision/]    # move the generated data to disired path
>> python TwoCurlySpirals_manifold.py        # the manifoldron on complex manifolds
```

<p align="center">
  <img width="700" src="https://user-images.githubusercontent.com/23077770/149415337-d109d7a3-9961-4c8f-bdec-24d34aa001e3.png">
</p>
<p align="center">
  <em>Fig. 2 Complex manifolds.</em>
</p>

<p align="center">
  <em>Tab. 2 Results on complex manifolds.</em>
</p>
<p align="center">
  <img width="750" src="https://user-images.githubusercontent.com/23077770/149415839-04f8ac01-a1a0-4aa5-9eda-4fe467281f15.png">
</p>

### Regression
Please go to 'regression/' folder and then run cooresponding .py file to run the manifoldron as regressor.
```ruby
>> python regressor_function1.py       # the manifoldron regressor.
```
