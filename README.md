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

1. Install OpenCV if you don't already have it:

```bash
$ conda install -c menpo opencv
```

2. Install this repository and the dependencies using pip:
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

* **Cityscapes**: Please follow the instructions in [Cityscape](https://www.cityscapes-dataset.com/) to download the images and validation ground-truths. The Cityscapes dataset directory should have this basic structure:
```bash
<root_dir>/../data/Cityscapes/                         % Cityscapes dataset root
<root_dir>/../data/Cityscapes/leftImg8bit              % Cityscapes images
<root_dir>/../data/Cityscapes/leftImg8bit/val
<root_dir>/../data/Cityscapes/gtFine                   % Semantic segmentation labels
<root_dir>/../data/Cityscapes/gtFine/val
...
```
### Prepare the datasets

For each dataset analized, there is a <dataset_name>_utils.py file in the <root_dir>/dataset folder. This file will transform the semantic labels to the Cityscapes train labels  to be employed.

### Train 

run the command line:
python main.py --config config/<experiment>.yalm

### Validate

run the command line: 
python main.py -r <model_to_validate> --config config/validate.yalm