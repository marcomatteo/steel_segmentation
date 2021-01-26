# Steel defect detection use case
> A walk through different solutions for the Severstal Kaggle competition.


This repository wants to explore different solutions for the [Severstal](https://www.kaggle.com/c/severstal-steel-defect-detection/overview) competition (ended in November 2019).

I used `pytorch` ([Pytorch website](https://pytorch.org/get-started/locally/)) and `fastai` ([FastAI docs](https://docs.fast.ai/#Installing)) as Deep Learning Framework to this project.

## Install

To install this package you only need to clone the repository and install via pip:

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
pip install -e .
nbdev_install_git_hooks
jupyter notebook
```

If there's some issues with `nbdev`, try to uninstall with `pip uninstall nbdev` and install again with `pip install nbdev`.
Now you can edit the Jupyter Notebook files.

To edit the `.py` modules in `steel_segmentation` you need to modify the Jupyter Notebooks inside the `nbs` folder (in the notebooks, only cells starting with #exp will be exported to the modules) and create the relative documentation, run:

```
nbdev_build_lib
nbdev_build_docs
```

Hint: this repository is build on `nbdev`. For more infos check the [nbdev repository](https://nbdev.fast.ai/).

NB: I tried to work in MacOS and Linux enviroment, not sure this is working also in Windows.

## Requirements

You will need the [Kaggle](https://www.kaggle.com/) competition data. If this is the first time with the API, follow this [link](https://github.com/Kaggle/kaggle-api) and download the credentials.

Copy the `kaggle.json` file in the repository directory.

```
!mkdir ~/.kaggle
```

    mkdir: /Users/marco/.kaggle: File exists


```
!cp ../kaggle.json ~/.kaggle/kaggle.json
```

Now you're ready to install the Kaggle API with:

```
!pip install kaggle
```

And then the download and unzip the data with:

```
!kaggle competitions download -c severstal-steel-defect-detection -p {path}
```

```
!unzip -q -n {path}/severstal-steel-defect-detection.zip -d {path}
```

Finally you can install the package with:

```
git clone https://github.com/marcomatteo/steel_segmentation.git
pip install -e steel_segmentation
```

## How to use

To visualize better the dataset, the `show_defects` function helps you to visualize the defected images.

```
#missing
show_defects(n=5, multi_defects=True)
```


![png](docs/images/output_18_0.png)



![png](docs/images/output_18_1.png)



![png](docs/images/output_18_2.png)



![png](docs/images/output_18_3.png)



![png](docs/images/output_18_4.png)

