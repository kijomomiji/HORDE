# The instructions for reproducing experiment results will be given (to be updated).


# HORDE: A Hybrid Model for Cardinality Estimation

The codes are developed based on the contributions of Wang et. al. 

<https://github.com/sfu-db/AreCELearnedYet>

# Part 1： Experiment Setup

Here we give an example based on Ubuntu 20.04 OS.

## (1) Create a new conda environment with python 3.8.13

`conda create -n re-execution python=3.8.13`

`conda activate re-execution`

`cd HORDE`

## (2) Install relevant packages

`pip install poetry`

`poetry lock`

remove every locked packages about 'sklearn' in file 'poetry.lock':
![image](https://github.com/kijomomiji/HORDE/blob/main/README_graphs/1.png)
![image](https://github.com/kijomomiji/HORDE/blob/main/README_graphs/2.png)

install remaining packages with the code: `poetry install`

install torch with the code (Our GPU is NVIDIA GeForce RTX 3090 GPU):

`pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116`

`pip install protobuf==3.19.0`

# Part 2：Static experiment

## Download data

Download data from the link: <https://pan.baidu.com/s/1g_wlP-rstUkIKB4WU4_LxQ?pwd=1111>

Unzip the file and place it as: ./HORDE/data

## Accurate module

Construct MRA-T and AC-Forest，and then record their inference time and result:

MRA-T: `just mine-AR_tree_inference census13 original`

AC-Forest: `just mine-tree_inference census13 original`

More commands about accurate module are shown [here](./static-accurate-command.md)

## HORDE performance in static environment

`just mine-CE_plus_sample census13 original base 500 32 100 200`

More commands for HORDE's performance on other datasets are shown [here](./CE_plus_sample.md)








