### [Paper](https://link.springer.com/article/10.1007/s11042-023-14662-0)  <br>

Pytorch implementation of our paper [On exploring weakly supervised domain adaptation strategies for semantic segmentation using synthetic data](https://link.springer.com/article/10.1007/s11042-023-14662-0).<br>

## Preparation

### Pre-requisites
* Python 3.7
* Pytorch >= 0.4.1
* CUDA 9.0 or higher

### Installation
0. Clone the repo:
```bash
$ git clone https://github.com/RobertoAlcoverCouso/OEWSDA
$ cd OEWSDA
```
1. Create a conda environment:
```bash
$ conda create -n OEWSDA python=3.7
$ conda activate OEWSDA
``` 
2. Install OpenCV if you don't already have it:

```bash
$ conda install -c menpo opencv
```

3. Install this repository and the dependencies using pip ```<root_dir>``` stands for ```./``` if you follow the instructions:
```bash
$ pip install -e <root_dir> 
```
### Datasets
By default, the datasets are put in ```<root_dir>/../data```. An alternative option is to explicitlly specify the parameters in the cfg file.


* **GTA5**: Please follow the instructions [here](https://download.visinf.tu-darmstadt.de/data/from_games/) to download images and semantic segmentation annotations. The GTA5 dataset directory should have this basic structure:
```bash
<root_dir>/../data/GTA5/                               % GTA dataset root
<root_dir>/../data/GTA5/images/                        % GTA images
<root_dir>/../data/GTA5/labels/                        % Semantic segmentation labels
...
```

* **Cityscapes**: Please follow the instructions in [Cityscapes](https://www.cityscapes-dataset.com/) to download the images and validation ground-truths. The Cityscapes dataset directory should have this basic structure:
```bash
<root_dir>/../data/Cityscapes/                         % Cityscapes dataset root
<root_dir>/../data/Cityscapes/leftImg8bit              % Cityscapes images
<root_dir>/../data/Cityscapes/leftImg8bit/val
<root_dir>/../data/Cityscapes/gtFine                   % Semantic segmentation labels
<root_dir>/../data/Cityscapes/gtFine/val
...
```
### Prepare the datasets

For each dataset analized, there is a <dataset_name>_utils.py file in the <root_dir>/dataset folder. This file will transform the semantic labels to the Cityscapes train labels  to be employed. For example for the Cityscapes dataset:
```bash
$ python  dataset/cityscapes_utils.py
```
This should have created 3 csvs for each of the subsets:  "trainCS.csv", "valCS.csv" and "testCS.csv"


### Train 

Modify the yalm file corresponding to the experiment you want to run "<experiment_name>.yalm" to include the datasets you want to train with, the proportion to use in the range of 0-1 as follows and the model you want to use as follows:
```yalm
architecture: <architecture_name>
train_set:
    <dataset_1>: <proportion_of_dataset_1>
    <dataset_2>: <proportion_of_dataset_2>
    ...
```
<architecture_name> is expected to be one of the following: "deeplabv3","FCN" or "psp"
To train run the command line:
```bash
python main.py --config config/<experiment_name>.yalm
```
Note that for fine_tuning a restore file is expected in the "restore_file" argument.

### Validate

run the command line: 
python main.py -r <model_to_validate> --config config/validate.yalm
