# Expanding Vision in Tree Counting: Novel Ground Truth Generation and Deep Learning Model

## Members

Minh Nhat Ton-That - 10422050@student.vgu.edu.vn

Tin Viet Le - 10422078@student.vgu.edu.vn

Nhien Hao Truong - 10422062@student.vgu.edu.vn

An Dinh Le - d0le@ucsd.edu

Anh-Duy Pham - apham@uni-osnabrueck.de

Hien Bich Vo - hien.vb@vgu.edu.vn

## Introduction
This is the code repository for the 10th IEEE INTERNATIONAL CONFERENCE ON COMMUNICATIONS AND ELECTRONICS 2024 (IEEE ICCE 2024) paper ["Expanding Vision in
Tree Counting: Novel Ground Truth Generation and Deep Learning Model"](https://ieeexplore.ieee.org/document/10634677) 

In our paper, we propose two contributions:
- A new ground truth generation practice to potentially boost the training effectiveness.
- A new tree-counting model called TreeVision with competitive performance compared with state-of-the-arts.

## Project structure
```
TreeVision Project
├───dataloaders                 # Contains .h5 files for the KCL-London and Yosemite dataset      
|   ├───london                     
|   |   │   ├───train.h5
|   │   |   └───val.h5
│   └───yosemite                           
|   |   │   ├───train.h5
|   │   |   └───val.h5
├───datasets                    # Contains images and labels of the KCL-London and Yosemite dataset.
|   ├───london                     
|   |   │   ├───train
|   |   │   |   │   ├───images
|   |   │   |   │   └───labels
|   |   │   ├───val
|   |   │   |   │   ├───images
|   |   │   |   │   └───labels
|   |   │   └───test
|   |   │   |   │   ├───images
|   |   │   |   │   └───labels
│   ├───yosemite                 # The images are cropped to size 512x512 according to IMAGE_PREPROCESS.md     
|   |   │   ├───zone A           # Zone A and C are reserved for testing
|   |   │   |   │   ├───images
|   |   │   |   │   └───labels
|   |   │   ├───zone B           # Zone B and D are reserved for training
|   |   │   |   │   ├───images
|   |   │   |   │   └───labels
|   |   │   ├───zone C
|   |   │   |   │   ├───images
|   |   │   |   │   └───labels
|   |   │   └───zone D
|   |   │   |   │   ├───images
|   |   │   |   │   └───labels
│   └───README.md  
├───weights                      # Default folder to save checkpoints and the best weight after training
│   ├───CSRNet_Shanghai_B_weights.h5 
├───dataloader.py
├───IMAGE_PREPROCESS.md
├───Github_Image_Processing.ipynb
├───london.yaml
├───model_utils.py
├───model.py
├───requirements.txt
├───README.md
├───test.py
├───train.py
├───main.ipynb
├───utils.py  
└───yosemite.yaml                                    
```
## Dataset


## Environment setup
We develop our project on Google Colab platform with Python 3.10.

To set up the environment, please:
 
 1. Clone this repository
 2. Run `pip install -r requirements.txt`
 3. Download the KCL-London and Yosemite datasets, along with their corresponding .h5 files, using the provided links.
 4. Arrange the downloaded files and folders to match the specified project structure.

## Evaluation
Download our pretrained model on the Yosemite and London datasets

[//]: # (Nhiên gắn cái link drive của từng file weight vào lần lượt chữ Yosemite và chữ london)

## Acknowledgement

