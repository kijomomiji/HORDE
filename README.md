# HORDE: A Hybrid Model for Cardinality Estimation

The codes are developed based on the contributions of Wang et. al. 

<https://github.com/sfu-db/AreCELearnedYet>

# Part 1： Experiment Setup

Here we give an example based on Ubuntu 20.04 OS.

## (1) Download data

Download data from the link: <https://drive.google.com/file/d/12NNDbKKJyNoYtXBkzMzmMflkQqldOvSH/view?usp=sharing>

Unzip the file and place it as: ./HORDE/data

## (2) Create a new conda environment with python 3.8.13

`conda create -n re-execution python=3.8.13`

`conda activate re-execution`

`cd HORDE`

## (3) Install relevant packages

`curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin`

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

## (1) Encoding data and queries

All the codes for encoding data and queries are shown in  [here](./static_data_label_get.md)


## (2) Accurate module

Construct MRA-T and AC-Forest，and then record their inference time and result:

MRA-T: `just mine-AR_tree_inference census13 original`

AC-Forest: `just mine-tree_inference census13 original`

More commands about accurate module are shown [here](./static-accurate-command.md).

## (3) HORDE performance in static scenarios

`just mine-CE_plus_sample census13 original base 500 32 100 200`

More commands for HORDE's performance on other datasets are shown [here](./CE_plus_sample.md).

## (4) Baseline methods performance in static scenarios

(Note that for baseline models, we use the same hyperparameters reported in the work: <https://github.com/sfu-db/AreCELearnedYet>.

All the commands for 4 datasets (census13, forest10, power7, dmv11) and 3 baselines (Naru, MSCN, DeepDB) are recorded [here](./static_baseline_command.md).

# Part 3: Dynamic experiment

## (1) Get dynamic data/label

All the commands for getting updated data/label with different update ratio (20%, 40%, 60%, 80%, 100%) are shown [here](./dynamic_data_label_get.md).

## (2) HORDE performance in dynamic scenarios

All the commands for training and testing HORDE in dynamic scenarios are shown [here](./CE_plus_sample_update.md).

## (3) Baseline methods performance in dynamic scenarios

All the commands for training and testing Naru/MSCN/DeepDB in dynamic scenarios are shown [here](./dynamic_baseline_command.md).

# Part 4： Multi-table experiment

Getting the performance of HORDE on 3 datasets (Job-light, Scale, Synthetic):

```bash
just train-train_multi_table
```

The performance of MSCN is based on its original code, which could be found in <https://github.com/andreaskipf/learnedcardinalities>.

# Code Refrences:

Naru : <https://github.com/naru-project/naru>

MSCN : <https://github.com/andreaskipf/learnedcardinalities>

DeepDB : <https://github.com/DataManagementLab/deepdb-public>

Are we ready for cardinality estimation? : <https://github.com/sfu-db/AreCELearnedYet>

# Forked repos:

(1) Are we ready for cardinality estimation? <https://github.com/sfu-db/AreCELearnedYet>

Changes: Adding codes for HORDE (including BND-CN and AC-Forest), which are mainly stored in the folder `./HORDE/lecarb/estimator/mine`.

(2) MSCN <https://github.com/andreaskipf/learnedcardinalities>

Note that there is no changes. We only use its codes for making MSCN as a baseline model in multi-table experiments.



