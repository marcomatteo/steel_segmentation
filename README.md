# Kaggle Severstal Steel Defect Detection
> A walk through different solutions for the Severstal Kaggle competition.


![CI](https://github.com/marcomatteo/steel_segmentation/workflows/CI/badge.svg?branch=master) 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/marcomatteo/steel_deployment/HEAD?urlpath=%2Fvoila%2Frender%2Fsteel_deploy.ipynb)

This repository wants to explore different solutions for the [Severstal](https://www.kaggle.com/c/severstal-steel-defect-detection/overview) competition hosted by Kaggle.
Kaggle is a platform that provides various datasets from the real world machine learning problems and engages a large community of people.
Severstal is a Russian company operating in the steel and mining industry. It creates a vast industrial data lake and in the 2019 looked to machine learning to improve automation, increase efficiency, and maintain high quality in their production.

The goal is to detect steel defects with segmentation models. The solutions are based on [Pytorch](https://pytorch.org/get-started/locally/) with [FastAI](https://docs.fast.ai/#Installing) as high level deep learning framework.

In this repository you will find some Jupyter Notebooks used to build the `steel_segmentation` library with [nbdev](https://nbdev.fast.ai/) and the training notebooks.

In the [steel_deployment](https://github.com/marcomatteo/steel_deployment) repository you can find a Binder/Voila web app for the deployment of the models built with this library (still updating).

## Install

To install this package, clone and install the repository and install via:

```bash
pip install git+https://github.com/marcomatteo/steel_segmentation.git
```

### Editable install

To install and edit this package:

```bash
clone git+https://github.com/marcomatteo/steel_segmentation.git
cd steel_segmentation
pip install -e steel_segmentation
```

The library is based on [nbdev](https://github.com/AnswerDotAI/nbdev), a powerful tool that builds a python package from Juptyer Notebooks.


To create the library, the documentation and tests use these commands:
```
nbdev_preview
```

This enviroment works on MacOS and Linux. In Windows the WLS with Ubuntu 20.04 is raccomended.

## Download the dataset

To download the [Kaggle](https://www.kaggle.com/) competition data you will need an account (if this is the first time with the API follow this [link](https://github.com/Kaggle/kaggle-api)) to generate the credentials, download and copy the `kaggle.json` into the repository directory.

```
!mkdir ~/.kaggle
!cp ../kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
```

Now you're authenticated with the Kaggle API (you'll need `kaggle` so `pip install kaggle` first), download and unzip the data:

```
!kaggle competitions download -c severstal-steel-defect-detection -p {path}
!mkdir data
!unzip -q -n {path}/severstal-steel-defect-detection.zip -d {path}
```

## Library notebooks

All of the experiments are based on Jupyter Notebooks and in the `nbs` folder there are all the notebooks used to build the `steel_segmentation` library (still updating):

- [Explorating Data Analysis](https://marcomatteo.github.io/steel_segmentation/eda.html): data analysis, plots and utility functions.
- [Transforms](https://marcomatteo.github.io/steel_segmentation/transforms.html): leveraging Middle-level API of `fastai` for custom data loading pipeline.
- [Optimizer utility functions](https://marcomatteo.github.io/steel_segmentation/optimizer.html)
- [Loss functions](https://marcomatteo.github.io/steel_segmentation/loss.html)
- [Metrics](https://marcomatteo.github.io/steel_segmentation/metrics.html)

## Training

Training script in `scripts` folder:

- `segmentation_train.py`: training segmentation models from [qubvel repository](https://github.com/qubvel/segmentation_models.pytorch).
- `create_submission.py` : create a kaggle submission from a segmentation model trained and save the csv in `data/submissions/`.

## Results

|Models|Public score|Private score|Percentile Private LB|
|------|------------|-------------|----------|
|Pytorch UNET-ResNet18|0.87530|0.85364|85°|
|Pytorch UNET-ResNet34|0.88591|0.88572|46°|
|FastAI UNET-ResNet34|0.88648|0.88830|23°|
|Pytorch FPN-ResNet34|0.89054|0.88911|19°|
|Ensemble UNET-ResNet34_FPN-ResNet34|0.89184|0.89262|16°|
