# Individual Acoustic Recognition

## Introduction
This repository contains the code for the paper [Acoustic recognition of individuals in closed and open bird populations](https://www.biorxiv.org/content/10.1101/2024.12.18.629284v2). 

## Installation
The project is written and tested with Python 3.8.10. To install the required packages, create a virtual environment and run the following command:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data
The red-tailed black cockatoo and little penguin data used in the paper were collected by the authors and are available at [Zenodo](https://zenodo.org/records/15054394). 
The remaining chiffchaff, little owl and tree pipit data were collected by the authors of [a different paper](https://royalsocietypublishing.org/doi/10.1098/rsif.2018.0940) and are available at [Zenodo](https://zenodo.org/records/1413495).
Once downloaded, each of the dataset folders should be placed in the `dataset` directory. The individual dataset folders provided here should already contain a `metadata.csv` file, a `birdnet-embeddings` and `google-perch-embeddings` directory, merely the data needs to be added.
The ESC-50 dataset can be downloaded from [here](https://github.com/karolpiczak/ESC-50) and should also be added to the `dataset` directory.
The folder structure format is purposely left inconsistent to allow for direct downloading and copying into the `dataset` directory from various sources. The complete folder structure should look like this:
```
dataset/
├── chiffchaff-fg/
│   ├── birdnet-embeddings/
│   │   └── within-year-embeddings.json
│   ├── google-perch-embeddings/
│   ├── cutted_day1_PC1101_0000.wav
│   ├── ...
│   └── metadata.csv
├── ESC-50-master/
│   ├── birdnet-embeddings/
│   │   └── embeddings.json
│   ├── google-perch-embeddings/
│   ├── ESC-10/
│   └── ...
├── littleowl-fg/
│   ├── birdnet-embeddings/
│   │   └── embeddings.json
│   ├── google-perch-embeddings/
│   ├── littleowl2017fg_test_7_0000.wav
│   ├── ...
│   └── metadata.csv
├── pipit-fg/
│   ├── birdnet-embeddings/
│   │   └── embeddings.json
│   ├── google-perch-embeddings/
│   ├── pipit2017fg_more_0212_0000.wav
│   ├── ...
│   └── metadata.csv
├── littlepenguin/
│   ├── birdnet-embeddings/
│   │   └── embeddings.json
│   ├── google-perch-embeddings/
│   ├── 13B3-1_clipped/
│   │   ├── 13B3-1_exhale-i1_1.wav
│   │   └── ...
│   └── metadata.csv
└── rtbc-call-types/
    ├── begging/
    │   └── 32PC1/
    │       └── 2021-22/
    │           ├── 0.wav
    │           └── ...
    ├── metadata.csv
    ├── birdnet-embeddings/
    │   └── embeddings.json
    └── begging-birdnet-embeddings/

```

## Training and evaluating a network in a closed-set scenario
Training and evaluation happens in one step here. 
Select one of the `<dataset>_test_<classifier>_classification` config files in `configs` and run the following command:
```
python main.py --cfg <path_to_config_file>
python main.py --cfg configs/chiffchaff/chiffchaff_test_birdnet_classification.yaml
```

## Training and evaluating a network in an out-of-distribution scenario
Training and evaluation happens in two steps here. 
To train, select one of the `<dataset>_outlier_<classifier>_classification*` config files in `configs` and run the following command:
```
python main.py --cfg <path_to_config_file>
python main.py --cfg configs/chiffchaff/chiffchaff_outlier_birdnet_classification.yaml
```
Once the model is trained (it is automatically saved), select the same config file and run the following command to evaluate its out-of-distribution performance:
```
python experiments/outlier_detection.py --cfg <path_to_config_file>
python experiments/outlier_detection.py --cfg configs/chiffchaff/chiffchaff_outlier_birdnet_classification.yaml
```

