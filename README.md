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
!cp kaggle.json ~/.kaggle/kaggle.json
```

Now you're ready to install the Kaggle API with:

```python
!pip install kaggle
```

    Collecting kaggle
      Downloading kaggle-1.5.10.tar.gz (59 kB)
    [K     |████████████████████████████████| 59 kB 1.8 MB/s eta 0:00:01
    [?25hRequirement already satisfied: six>=1.10 in /home/mmb/miniconda3/envs/env_nbdev/lib/python3.8/site-packages (from kaggle) (1.15.0)
    Requirement already satisfied: certifi in /home/mmb/miniconda3/envs/env_nbdev/lib/python3.8/site-packages (from kaggle) (2020.12.5)
    Requirement already satisfied: python-dateutil in /home/mmb/miniconda3/envs/env_nbdev/lib/python3.8/site-packages (from kaggle) (2.8.1)
    Requirement already satisfied: requests in /home/mmb/miniconda3/envs/env_nbdev/lib/python3.8/site-packages (from kaggle) (2.25.1)
    Requirement already satisfied: tqdm in /home/mmb/miniconda3/envs/env_nbdev/lib/python3.8/site-packages (from kaggle) (4.55.1)
    Requirement already satisfied: urllib3 in /home/mmb/miniconda3/envs/env_nbdev/lib/python3.8/site-packages (from kaggle) (1.26.2)
    Collecting python-slugify
      Downloading python-slugify-4.0.1.tar.gz (11 kB)
    Collecting text-unidecode>=1.3
      Downloading text_unidecode-1.3-py2.py3-none-any.whl (78 kB)
    [K     |████████████████████████████████| 78 kB 4.4 MB/s eta 0:00:011
    [?25hRequirement already satisfied: idna<3,>=2.5 in /home/mmb/miniconda3/envs/env_nbdev/lib/python3.8/site-packages (from requests->kaggle) (2.10)
    Requirement already satisfied: chardet<5,>=3.0.2 in /home/mmb/miniconda3/envs/env_nbdev/lib/python3.8/site-packages (from requests->kaggle) (4.0.0)
    Building wheels for collected packages: kaggle, python-slugify
      Building wheel for kaggle (setup.py) ... [?25ldone
    [?25h  Created wheel for kaggle: filename=kaggle-1.5.10-py3-none-any.whl size=73268 sha256=200d1a75e6c148a2f8cd173eec512b8cd901561a1c5d81bf3a55b5f28410ef9b
      Stored in directory: /home/mmb/.cache/pip/wheels/a6/c1/5e/2b235e19b52c15ad35812881f8de4461399907e219c03bf7b5
      Building wheel for python-slugify (setup.py) ... [?25ldone
    [?25h  Created wheel for python-slugify: filename=python_slugify-4.0.1-py2.py3-none-any.whl size=6769 sha256=e7c635448f00b23f2eeace3194ecb21d077d287a407eb6e797363bd5fcb6670a
      Stored in directory: /home/mmb/.cache/pip/wheels/91/4d/4f/e740a68c215791688c46c4d6251770a570e8dfea91af1acb5c
    Successfully built kaggle python-slugify
    Installing collected packages: text-unidecode, python-slugify, kaggle
    Successfully installed kaggle-1.5.10 python-slugify-4.0.1 text-unidecode-1.3


And then the download and unzip the data with:

```python
!kaggle competitions download -c severstal-steel-defect-detection -p {path}
```

    Warning: Your Kaggle API key is readable by other users on this system! To fix this, you can run 'chmod 600 /home/mmb/.kaggle/kaggle.json'
    Downloading severstal-steel-defect-detection.zip to data
    100%|█████████████████████████████████████▉| 1.57G/1.57G [01:16<00:00, 25.7MB/s]
    100%|██████████████████████████████████████| 1.57G/1.57G [01:16<00:00, 22.0MB/s]


```python
!unzip -q -n {path}/severstal-steel-defect-detection.zip -d {path}
```

    /bin/sh: 1: unzip: not found


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
show_defects(n=5, multi_defects=True)
```


![png](docs/images/output_16_0.png)



![png](docs/images/output_16_1.png)



![png](docs/images/output_16_2.png)



![png](docs/images/output_16_3.png)



![png](docs/images/output_16_4.png)

