# APPLeNet: Visual Attention Parameterized Prompt Learning for Few-Shot Remote Sensing Image Generalization using CLIP

Official repository of APPLeNet, which is one of the first works in Remote Sensing to perform unknown class and domain generalization using *prompt learning* by adapting pre-trained vision-language models (VLM) like [CLIP](https://arxiv.org/abs/2103.00020).

## **CVPRw 2023**

[![paper](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/papers/Jha_APPLeNet_Visual_Attention_Parameterized_Prompt_Learning_for_Few-Shot_Remote_Sensing_CVPRW_2023_paper.pdf)
[![supplement](https://img.shields.io/badge/Supplementary-Material-F9D371)](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/supplemental/Jha_APPLeNet_Visual_Attention_CVPRW_2023_supplemental.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/abs/2304.05995)

## Abstract

![teaser](https://github.com/mainaksingha01/APPLeNet/blob/master/images/teaser.png)

In recent years, the success of large-scale visionlanguage models (VLMs) such as CLIP has led to their increased usage in various computer vision tasks. These models enable zero-shot inference through carefully crafted instructional text prompts without task-specific supervision.
However, the potential of VLMs for generalization tasks in remote sensing (RS) has not been fully realized. To address this research gap, we propose a novel image-conditioned prompt learning strategy called the Visual Attention Parameterized Prompts Learning Network (APPLeNet). APPLeNet emphasizes the importance of multi-scale feature learning in RS scene classification and disentangles visual style and content primitives for domain generalization tasks. To achieve this, APPLeNet combines visual content features obtained from different layers of the vision encoder and style properties obtained from feature statistics of domain-specific batches. An attention-driven injection module is further introduced to generate visual tokens from this information. We also introduce an anticorrelation regularizer to ensure discrimination among the token embeddings, as this visual information is combined with the textual tokens. To validate APPLeNet, we curated
four available RS benchmarks and introduced experimental protocols and datasets for three domain generalization tasks.

## Architecture

![architecture](https://github.com/mainaksingha01/APPLeNet/blob/master/images/applenet.png)

APPLeNet is composed of a text encoder, an image encoder, and an injection block designed for multi-scale visual feature refinement. The image encoder produces multi-level visual content features, and the batch statistics for a domain as the style features, that are passed through a residual attention-based injection block.

## Datasets
- For Base-to-New Class and Cross-Dataset Generalization:
  - [PatternNet](https://sites.google.com/view/zhouwx/dataset)
  - [RSICD](https://github.com/201528014227051/RSICD_optimal)
  - [RESISC45](https://www.tensorflow.org/datasets/catalog/resisc45)
  - [MLRSNet](https://data.mendeley.com/datasets/7j9bv9vwsx/3)

## Released Datasets (Version-2):
- For Domain Generalization:
  - [PatternNetv2]([https://sites.google.com/view/zhouwx/dataset](https://drive.google.com/file/d/1bukts4nv_ZV7opYL3dFETD66EueiFaNo/view?usp=sharing))
  - [RSICDv2](https://drive.google.com/file/d/1cHQPA7X_jX4tw8vFgEON1SDn_MGf3uv2/view?usp=sharing)
  - [RESISC45v2]([https://www.tensorflow.org/datasets/catalog/resisc45](https://drive.google.com/file/d/1B7GTf-Pz6ggaIhm9ShQXfXTe0GilfiQA/view?usp=sharing)https://drive.google.com/file/d/1B7GTf-Pz6ggaIhm9ShQXfXTe0GilfiQA/view?usp=sharing)
  - [MLRSNetv2]([https://data.mendeley.com/datasets/7j9bv9vwsx/3](https://drive.google.com/file/d/18uaXD5CxnAyYPOswctf6_izcIO8VF4oI/view?usp=sharing)https://drive.google.com/file/d/18uaXD5CxnAyYPOswctf6_izcIO8VF4oI/view?usp=sharing)
 
## Code

 - `files` folder contains the dataloader files of each datasets.
 - `models` folder contains the code of our model.
 - Clone this repository [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch) inside this repo for the metrices.
 - `scripts` folder holds the scripts of each of the generalization tasks both for training and testing.

```shell
$ cd scripts
$ bash base2new_train.sh patternnet 1
```

## Results

### Base-to-New Class Generalization

![base2new](https://github.com/mainaksingha01/APPLeNet/blob/master/images/base2new.png)

### Cross Dataset Generalization

![crossdataset](https://github.com/mainaksingha01/APPLeNet/blob/master/images/crossdataset.png)

### Domain Generalization

![domaingen](https://github.com/mainaksingha01/APPLeNet/blob/master/images/domaingen.png)

## Bibtex

Please cite our work if you use our work. Thanks.

```
@inproceedings{singha2023applenet,
  title={Applenet: Visual attention parameterized prompt learning for few-shot remote sensing image generalization using clip},
  author={Singha, Mainak and Jha, Ankit and Solanki, Bhupendra and Bose, Shirsha and Banerjee, Biplab},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```

## Acknowledgements

Thanks to the authors of [CoOp](https://github.com/KaiyangZhou/CoOp) as our code is mainly based on this repository.
