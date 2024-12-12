# Image Preprocessing Documentation

## Overview
This document describes the image preprocessing pipeline for preparing tree counting datasets. The pipeline includes cropping large satellite images, generating labels, and creating HDF5 datasets.

## Requirements
- PIL (Python Imaging Library)
- OpenCV
- NumPy 
- H5py
- SciPy
- PyTorch
- Matplotlib

## Note for Yosemite dataset
The original dataset includes a large image and a label file, which contains the coordinates of the trees within the image. It is essential to crop the image into smaller sections while preserving these coordinates. The author has cropped the image into 512x512 sections from four zones of the original image, with each zone containing 675 images arranged in a grid of 9 images in width and 75 images in height.

To enhance the performance of the Gaussian filter, as discussed in our paper, the image should be cropped to a larger size. However, to maintain 675 images per zone, the larger image size must be 512 multiplied by a divisor of either 75 or 9, depending on the dimension. Since the power capability of our machine is limited, we have decided to crop the image into 1536x1536 sections. After cropping, apply the Gaussian filter to the image to create the density map, and then split the cropped image back into 512x512 sections.

## Step by step to process the dataset
Download the dataset from the provided link in readme.md
### For Yosemite dataset
Before running the crop, it is crucial to generate the label annotation in txt format since the original label is in png format
``` Python
file_path = "datasets/yosemite/labels.txt"  
im = Image.open('datasets/yosemite/z20_label.png') 
pix = im.load()
x, y = im.size
with open(file_path, "w") as f:
  for i in range(x):
    for j in range(y):
      value = pix[i, j]
      if value > 0:
        rel_x = i
        rel_y = j
        f.write(f"{rel_x} {rel_y}\n")
```
#### Crop the image into 512x512 or 1536x1536 segments
Use the following code to crop the image into 512x512 or 1536x1536 segments
``` Python
#Row by Row cropping
input_image_path = 'datasets/yosemite/z20_data.png'  # Specify the path to your input image
input_label_path = 'datasets/yosemite/labels.txt'  # Specify the path to your input labels file
crop_size =  512 # Specify the size of the square crop (both width and height)
output_folder = f'datasets/yosemite_{crop_size}'  # Specify the output folder where cropped images and labels will be saved

main_row(input_image_path, input_label_path, output_folder, crop_size, crop_size)
```
#### Apply the Gaussian filter to both dataset
Then we apply the Gaussian filter to the cropped images to generate the density map
``` Python
generate_our_own_data("yosemite_1536")
```
#### For image size 1536x1536, we split the image and density map into 512x512 segments
Function to crop the image and labels into 512x512 segments
``` Python
image_source = 'datasets/yosemite_1536/zone_A/images'
label_source = 'datasets/yosemite_1536/zone_A/labels'
output_folder = 'datasets/yosemite_1536'
zone = 'zone_A'
split_9_images_labels(image_source, label_source, output_folder, zone, 3, 3)
```
Function to crop the density map into 512x512 segments
``` Python
original_file_path = 'Density_Map/train_yosemite_1536.h5' # Specify the path to the original HDF5 file
splitDensityMap(original_file_path)
```
### For KCL-London dataset
Since the KCL-London dataset is already in the correct format, we only need to add all the density maps into h5 format
``` Python
generate_our_own_data("london")
```
### Notes
- Images are cropped into 1536x1536 pixel segments to further split into image of size 512x512 after apply Gaussian filter
- Or directly cropped into size 512x125
- Labels are automatically adjusted for cropped segments
- Density maps are generated using gaussian filtering
- Datasets are split into train/test sets
- Output is saved in HDF5 format for efficient loading
```