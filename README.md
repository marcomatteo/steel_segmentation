# Steel defect detection use case
> A walk through different solutions for the Severstal Kaggle competition.


This repository wants to explore different solutions for the [Severstal](https://www.kaggle.com/c/severstal-steel-defect-detection/overview) competition (ended in November 2019) by the Living Lab Research team of [IP4FVG](https://www.ip4fvg.it/), the Digital Innovation Hub of Friuli Venezia Giulia.

## Install

To install this package you need to fullfill the following requirements.

1. make sure to install properly the `pytorch` ([pytorch installation](https://pytorch.org/get-started/locally/)) and `fastai` ([fastai installation](https://docs.fast.ai/#Installing)) packages.

2. download the data from Kaggle. If this is the first time with the API, you'll need a [Kaggle](https://www.kaggle.com/) account. Then follow this [link](https://github.com/Kaggle/kaggle-api) to understand how to download the credentials in the repository directory `kaggle.json` file.

3. run the following cell code (if is not already done)

```python
!mkdir ~/.kaggle
```

```python
!cp ../kaggle.json ~/.kaggle/kaggle.json
```

Now you're ready to install the Kaggle API with:

```python
!pip install kaggle
```

And then the download and unzip the data with:

```python
!kaggle competitions download -c severstal-steel-defect-detection -p {path}
```

```python
!unzip -q -n {path}/severstal-steel-defect-detection.zip -d {path}
```

Finally you can install the package with:

```
git clone https://github.com/marcomatteo/steel_segmentation.git
pip install -e steel_segmentation
```

## Developing

To set up the developer enviroment to build the package or try different Deep Learning models using this repository, follow these commands:

```
conda create -n env_nbdev
conda activate env_nbdev
conda install -c fastai -c pytorch fastai
conda install -c conda-forge jupyter_contrib_nbextensions
git clone https://github.com/fastai/nbdev
pip install -e nbdev
git clone https://github.com/marcomatteo/steel_segmentation.git
cd steel_segmentation
nbdev_install_git_hooks
jupyter notebook
```

If there's some issues with `nbdev`, try to uninstall with `pip uninstall nbdev` and install again with `pip install nbdev`.
Now you can edit the Jupyter Notebook files.

To save the new functions or classes from the notebooks to a `.py` module in `steel_segmentation` (only the cells with #exp at the beginning) and create the relative documentation, run:

```
nbdev_build_lib
nbdev_build_docs
```

Hint: for a complete understanding of `nbdev` check this [link](https://nbdev.fast.ai/).

NB: I tried to work in MacOS and Linux enviroment, not sure this is working also in Windows.

## How to use

With the `show_defects` function you can easly view the defected images:

```python
#missing
show_defects(n=5, multi_defects=True)
```


![png](docs/images/output_16_0.png)



![png](docs/images/output_16_1.png)



![png](docs/images/output_16_2.png)



![png](docs/images/output_16_3.png)



![png](docs/images/output_16_4.png)

