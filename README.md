# DeltaNet

## 简洁版
训练测试代码主入口：

`main_basic.py`: basic model

`main_conditional.py`: 使用一个condition的model

`main_multi_conditional.py`: 使用多个condition的model

所有代码运行所需参数已经在`config`文件中设置好

运行命令只需指定配置文件和所需使用的gpu id

示例：
`python main<model>.py --cfg config/iu_con_1.yml --gpu 0`


## 详细版

## Requirements
- `Python >= 3.6`
- `Pytorch >= 1.7`
- `torchvison`
- [Microsoft COCO Caption Evaluation Tools](https://github.com/tylin/coco-caption)
- [CheXpert](https://github.com/stanfordmlgroup/chexpert-labeler)

## Data

Download IU and MIMIC-CXR datasets, and place them in `data` folder.

- IU dataset from [here](https://iuhealth.org/find-medical-services/x-rays)
- MIMIC-CXR dataset from [here](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

## Folder Structure
- config : setup training arguments and data path
- data : store IU and MIMIC dataset
- Models: basic model and all our models
    - the layer define of our model 
    - loss function
    - metrics
    - some utils
- data_loader.py: dataloader
- build_vocab.py: tokenizer
- preprocessing: data preprocess
- pycocoevalcap: Microsoft COCO Caption Evaluation Tools

## Data preparation

1. `data`
    - Download and unzip IU dataset to here
    
2. `preprocess_data.py` 
    - Create dataset split for IU dataset. 
    - Preprocess report for normalizion, such as lower-case, replacing nonsense tokens(e.g., 'xxxx-a-xxxx').
    - Build vocabulary file.

3. `extract_feature.py`
    - Extract visual feature for retrieve conditional image.
    
4. `retieve_conditional_pair.py`
    - Retrieve conditional image using cosine similarity.
    
## Training
- dataset: iu / mimic

1. `main_<model>.py`
    - The model will be trained using command `python main_<model>.py --cfg config/<dataset_N> --expe_name <experiment name> --gpu <GPU_ID>`
    - More options can be found in `config/opts.py` file.

2. `Models`
    - `Basic.py`: Basic Model
    - `Conditional.py`: DeltaNet condition generation model
    - `MultiConditional.py`: DeltaNet multiple conditional generation model
    - `Generator.py` and `Beam.py`: Generate report using beam search
    - `Modules.py`: Implement of scaled dot-product attention
    - `SubLayers.py`: Implement of multi-head attention
    - `misc.py` and `utils.py`: Utils function

## Testing

`test.py`: Generate report from trained model using command `python test.py --pretrained <path to the checkpoint file>`

## Evaluate
   
1. `metrics.py`
    - Evaluate the generated report.
    
2. `pycocoevalcap`
    - Microsoft COCO Caption Evaluation Tools
    - Need clone from GitHub and modified the code to work with Python 3
    
