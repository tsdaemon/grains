Script to estimate grain areas on microscope image using OpenCV.

# HOWTO

First, create a new virtual env and install python requirements:

```shell script
pip install -r requirements.txt
```

Next, install `grains` as a library:

```shell script
pip install -e .
```

Now you can run example scripts:

```shell script
python scipts/extract_grains_area.py --img example_data/Snap-05.jpg --out-img out.jpg --out-csv out.csv
```
# Experiments

Folder `experiments` contains jupyter notebook with my experiments. To access them:
```shell script
pip install -r requirements_dev.txt
cd experiments
jupyter notebook
```