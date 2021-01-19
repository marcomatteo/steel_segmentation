# Steel defect detection use case
> A walk through different solutions for the Severstal Kaggle competition.


This repository wants to explore different solutions for the [Severstal](https://www.kaggle.com/c/severstal-steel-defect-detection/overview) competition (ended in November 2019) by the Living Lab Research team of [IP4FVG](https://www.ip4fvg.it/), the Digital Innovation Hub of Friuli Venezia Giulia.

## Install

To install this package you need to fullfill the following requirements.

1. make sure to install properly the `pytorch` ([pytorch installation](https://pytorch.org/get-started/locally/)) and `fastai` ([fastai installation](https://docs.fast.ai/#Installing)) packages.

2. download the data from Kaggle. If this is the first time with the API, you'll need a [Kaggle](https://www.kaggle.com/) account. Then follow this [link](https://github.com/Kaggle/kaggle-api) to understand how to download the credentials in the repository directory `kaggle.json` file.

3. run the following cell code (if is not already done)

```python
!cp kaggle.json ~/.kaggle/kaggle.json
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

## How to use

With the `show_defects` function you can easly view the defected images:

```python
show_defects(n=5)
```


![png](docs/images/output_13_0.png)



![png](docs/images/output_13_1.png)



![png](docs/images/output_13_2.png)



![png](docs/images/output_13_3.png)



![png](docs/images/output_13_4.png)

