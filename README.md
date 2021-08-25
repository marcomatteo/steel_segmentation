# Kaggle Severstal Steel Defect Detection
> A walk through different solutions for the Severstal Kaggle competition.


![CI](https://github.com/marcomatteo/steel_segmentation/workflows/CI/badge.svg?branch=master) 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcomatteo/steel_segmentation/blob/master/nbs/index.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/marcomatteo/steel_deployment/HEAD?urlpath=%2Fvoila%2Frender%2Fsteel_deploy.ipynb)

This repository wants to explore different solutions for the [Severstal](https://www.kaggle.com/c/severstal-steel-defect-detection/overview) competition hosted by Kaggle.
Kaggle is a platform that provides various datasets from the real world machine learning problems and engages a large community of people.
Severstal is a Russian company operating in the steel and mining industry. It creates a vast industrial data lake and in the 2019 looked to machine learning to improve automation, increase efficiency, and maintain high quality in their production.

I used `pytorch` ([Pytorch website](https://pytorch.org/get-started/locally/)) and `fastai` ([FastAI docs](https://docs.fast.ai/#Installing)) as Deep Learning Framework to this project.

In the [steel_deployment](https://github.com/marcomatteo/steel_deployment) repository you can find a Binder/Voila web app for the deployment of the models built with this library (still updating).

## Install

To install this package you only need to clone the repository and install via pip:

```
pip install git+https://github.com/marcomatteo/steel_segmentation.git
```

### Editable install

To install and edit this package you only need to clone the repository and install via pip:

```
clone git+https://github.com/marcomatteo/steel_segmentation.git
pip install -e steel_segmentation
```


The library is based on `nbdev`, a powerful tool that builds a python package from Juptyer Notebooks, from the `dev_nbs` folder. 
Check [here](https://nbdev.fast.ai/) the `nbdev` documentation.

```
pip install nbdev
```

To create the library, the documentation and tests execute these commands:
```
nbdev_build_lib
nbdev_test_nbs
nbdev_clean_nbs
nbdev_build_docs
```

This enviroment works on MacOS and Linux. In Windows the WLS with Ubuntu 20.04 is raccomended.

Training only in Windows needs one package more: 
```
conda install pywin32
```

## Download the dataset

To download the [Kaggle](https://www.kaggle.com/) competition data you will need an account (if this is the first time with the API follow this [link](https://github.com/Kaggle/kaggle-api)) to generate the credentials, download and copy the `kaggle.json` into the repository directory.

Now run these cells:

```
!mkdir ~/.kaggle
!cp ../kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
```

Now you're authenticated with the Kaggle API. Download and unzip the data with:

```
!pip install kaggle
!kaggle competitions download -c severstal-steel-defect-detection -p {path}
!mkdir data
!unzip -q -n {path}/severstal-steel-defect-detection.zip -d {path}
```

## Notebooks

All of the experiments are based on Jupyter Notebooks. 
In the `nbs` folder there are all the notebooks used to build the `steel_segmentation` library, to train different Deep Learning models and evaluate them with the testset.

## Results

|Models|Public score|Private score|Percentile Private LB|
|------|------------|-------------|----------|
|Pytorch UNET-ResNet18|0.87530|0.85364|85°|
|Pytorch UNET-ResNet34|0.88591|0.88572|46°|
|FastAI UNET-ResNet34|0.88648|0.88830|23°|
|Pytorch FPN-ResNet34|0.89054|0.88911|19°|
|Ensemble UNET-ResNet34_FPN-ResNet34|0.89184|0.89262|16°|
