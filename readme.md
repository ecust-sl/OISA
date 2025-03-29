# OISA：Online Iterative Self-Alignment for Radiology Report Generation

This is the implementation of Online Iterative Self-Alignment for Radiology Report Generation.

## Requirements

- pip install -r requirement.txt

## Datasets

- We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.


For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) .

For `MIMIC-CXR`, you can download the dataset from [here](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing) .You can apply the dataset [here](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing) with your license of [PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).

- Preference DataSets：

For SFT1: data\mimic_cxr\FINAL2\SFT1

For SFT2: data\mimic_cxr\FINAL2\SFT2

For SFT3: data\mimic_cxr\FINAL2\SFT3

## Train

Run `bash \scripts\mimic_cxr\SFT1\mimic_cxr_sft1.sh` to train a model on the SFT-1 data.

Run `bash \scripts\mimic_cxr\SFT2\mimic_cxr_sft2.sh` to train a model on the SFT-2 data.

Run `bash \scripts\mimic_cxr\SFT3\mimic_cxr_sft3.sh` to train a model on the SFT-3 data.

You can download the pre-trained model for CheXbert from here: [Chexbert](https://stanfordmedicine.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9). 

For using RadGraph, you can refer to the following link: [RadGraph](https://github.com/hlk-1135/RadGraph). The specific model checkpoint can be downloaded from here: [model_checkpoint](https://physionet.org/content/radgraph/1.0.0/models/model_checkpoint/#files-panel). Place the related files in my `MPO_IU\MPO_TRAIN\RadGraph` directory

For using GREEN metric, you can refer to the following link: [GREEN]([Stanford-AIMI/GREEN: [EMNLP, Findings 2024\] a radiology report generation metric that leverages the natural language understanding of language models to identify and explain clinically significant errors in candidate reports](https://github.com/Stanford-AIMI/GREEN))

For using RadCliQ metric, you can refer to the following link: [RadCliQ]([pleyad/RadCLIQ-CXR](https://github.com/pleyad/RadCLIQ-CXR))

## Test

Run `bash test_iu_xray.sh` to test a model on the IU X-Ray data.

Run `bash test_mimic_cxr.sh` to test a model on the MIMIC-CXR data.

