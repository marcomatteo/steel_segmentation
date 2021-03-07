# Steel defect detection use case
> A walk through different solutions for the Severstal Kaggle competition.


![CI](https://github.com/marcomatteo/steel_segmentation/workflows/CI/badge.svg?branch=master)

This repository wants to explore different solutions for the [Severstal](https://www.kaggle.com/c/severstal-steel-defect-detection/overview) competition (ended in November 2019).

I used `pytorch` ([Pytorch website](https://pytorch.org/get-started/locally/)) and `fastai` ([FastAI docs](https://docs.fast.ai/#Installing)) as Deep Learning Framework to this project.

## Install

To install this package you only need to clone the repository and install via pip:

```
git clone https://github.com/marcomatteo/steel_segmentation.git
pip install -e steel_segmentation
```

The library is based on `nbdev`, a powerful tool that builds a python package from Juptyer Notebooks, from the `nbs` folder. 
Check the [here](https://nbdev.fast.ai/) the `nbdev` documentation.

With these commands you can create the library and the relative documentation:
```
nbdev_build_lib
nbdev_build_docs
```

I tried to work in MacOS and Linux enviroment, not sure this is working also in Windows.

## Data requirements

You will need the [Kaggle](https://www.kaggle.com/) competition data. If this is the first time with the API, follow this [link](https://github.com/Kaggle/kaggle-api) and download the credentials.

Move the `kaggle.json` file in the repository directory.

```
!mkdir ~/.kaggle
```

```
!cp ../kaggle.json ~/.kaggle/kaggle.json
```

Now you're authenticated with the Kaggle API. Download and unzip the data with:

```
!kaggle competitions download -c severstal-steel-defect-detection -p {path}
```

```
!unzip -q -n {path}/severstal-steel-defect-detection.zip -d {path}
```

## Results

|Models|Public score|Private score|Percentile Private LB|
|------|------------|-------------|----------|
|Pytorch-UNET-ResNet18|0.87530|0.85364|85°|
|Pytorch-UNET-ResNet34|0.88591|0.88572|46°|
|FastAI-UNET-ResNet34|0.88648|0.88830|23°|
|Pytorch-FPN-ResNet34|0.89054|0.88911|19°|
|Ensemble-UNET-ResNet34-FPN-ResNet34|0.89184|0.89262|16°|
