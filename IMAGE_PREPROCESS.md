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

## Dataset download and directory structure
Link to London dataset: "https://drive.google.com/drive/folders/18qfBuIGlYOr3hYGulug1QUcUHmonboY0?usp=sharing"
Link to Yosemite dataset version crop directly to size 512x512: "https://drive.google.com/drive/folders/1utzrXXKkO5DdWa6wfPOJklxY3m3KrIRu?usp=sharing"
Link to Yosemite dataset version crop from size 1536x1536 to size 512x512: "https://drive.google.com/drive/folders/16M3RUIcpvIMloO0zoeUwtTC-tOsPDBE8?usp=sharing"
Link to original Yosemite dataset: "https://github.com/nightonion/yosemite-tree-dataset"

After downloading the datasets, you should organize the files as follows:
```
Dataset/
├── London/
│   ├── test/
│   ├── train/
│   └── val/
└── Yosemite/
    ├── Dataset_Row_1536x1536/
    │   ├── zone_A/
    │   ├── zone_B/
    │   ├── zone_C/
    │   └── zone_D/
    └── labels.txt
    |__z20_data.png
    |__z20_label.png
```

## Note for Yosemite dataset

The original dataset includes a large image and a label file, which contains the coordinates of the trees within the image. It is essential to crop the image into smaller sections while preserving these coordinates. The author has cropped the image into 512x512 sections from four zones of the original image, with each zone containing 675 images arranged in a grid of 9 images in width and 75 images in height.

To enhance the performance of the Gaussian filter, as discussed in our paper, the image should be cropped to a larger size. However, to maintain 675 images per zone, the larger image size must be 512 multiplied by a divisor of either 75 or 9, depending on the dimension. Since the power capability of our machine is limited, we have decided to crop the image into 1536x1536 sections. After cropping, apply the Gaussian filter to the image to create the density map, and then split the cropped image back into 512x512 sections.

## Preprocessing Steps

### 1. Image Size Configuration
```python
PIL.Image.MAX_IMAGE_PIXELS = 1262080000
```

### 2. Cropping Functions

#### Small Crop Function
```python
def smallCrop(image, cw, ch, labels, left, top):
    """
    Crops image to specified dimensions while preserving label coordinates
    
    Args:
        image: Source image
        cw: Crop width
        ch: Crop height
        labels: List of (x,y) coordinates
        left: Left starting position
        top: Top starting position
    """
    width, height = image.size
    cropped_image = image.crop((left, top, left + cw, top + ch))
    updated_labels = [(x - left, y - top) for x, y in labels if left <= x <= left + cw and top <= y <= top + ch]
    return cropped_image, updated_labels
```

#### Main Row Cropping
```python
def main_row(input_image_path, input_label_path, output_folder, cw, ch):
    """
    Main function for row-wise image cropping
    
    Args:
        input_image_path: Path to source image
        input_label_path: Path to label file
        output_folder: Output directory
        cw: Crop width (1536)
        ch: Crop height (1536)
    """
    os.makedirs(output_folder, exist_ok=True)
    folders = ['zone_A', 'zone_B', 'zone_C', 'zone_D']
    for folder in folders:
        zone = os.path.join(output_folder, folder)
        os.makedirs(os.path.join(zone, 'images'), exist_ok=True)
        os.makedirs(os.path.join(zone, 'labels'), exist_ok=True)

    image = Image.open(input_image_path)
    with open(input_label_path, 'r') as label_file:
        labels = [tuple(map(float, line.strip().split())) for line in label_file]

    x_min = 4000
    x_max = x_min + int(19200/4) - cw
    y_min = 4000
    y_max = y_min + 38400
    count = 0

    for i in range(len(folders)):
        print(f"Width range {x_min} - {x_max}")
        print(f"Height range {y_min} - {y_max}")
        j = 1
        for top in range(y_min, y_max, ch):
            for left in range(x_min, x_max, cw):
                print(count, end = " ")
                print("Left:", left, "Top:", top)
                cropped_image, updated_labels = smallCrop(image, cw, ch, labels, left, top)
                output_image_path = os.path.join(output_folder, folders[i], 'images', f'IMG_{count}.jpg')
                cropped_image.save(output_image_path)
                output_label_path = os.path.join(output_folder, folders[i], 'labels', f'IMG_{count}.txt')
                with open(output_label_path, 'w') as updated_label_file:
                    for x, y in updated_labels:
                        updated_label_file.write(f"{x} {y}\n")
                j += 1
                count += 1
        print(f"The number of images and labels in {folders[i]}: {j-1}\n")
        x_min += int(19200/4)
        x_max += int(19200/4)
```

### 3. Dataset Generation

#### Create HDF5 Files
```python
def create_hdf5(dataset_path: str, dataset: str):
    """
    Create empty training and validation HDF5 files with placeholders
    for images and labels (density maps).

    Note:
    Datasets are saved in [dataset_name]/train.h5 and [dataset_name]/valid.h5.
    Existing files will be overwritten.

    Args:
        dataset_name: used to create a folder for train.h5 and valid.h5

    Returns:
        A tuple of pointers to training and validation HDF5 files.
    """
    os.makedirs(dataset_path, exist_ok=True)
    train_h5 = h5py.File(os.path.join(dataset_path, f'train_{dataset}.h5'), 'w')
    valid_h5 = h5py.File(os.path.join(dataset_path, f'valid_{dataset}.h5'), 'w')
    return train_h5, valid_h5
```

#### Generate Labels
```python
def generate_label(label_info: np.array, image_shape: List[int]):
    """
    Generate a density map based on objects positions.

    Args:
        label_info: (x, y) objects positions
        image_shape: (width, height) of a density map to be generated

    Returns:
        A density map.
    """
    label = np.zeros(image_shape, dtype=np.float32)
    for x, y in label_info:
        if y < image_shape[0] and x < image_shape[1]:
            label[int(y)][int(x)] = 1
    label = gaussian_filter_density(label)
    return label
```

### Usage Examples

#### Configure paths
```python
input_image_path = 'Dataset/Yosemite/z20_data.png'
input_label_path = 'Dataset/Yosemite/labels.txt'
output_folder = 'Dataset/Yosemite/Dataset_Row_1536x1536'
```

#### Set crop dimensions
```python
crop_width = 1536
crop_height = 1536
```

#### Process images
```python
main_row(input_image_path, input_label_path, output_folder, crop_width, crop_height)
```

#### Generate HDF5 Datasets
```python
# For Yosemite dataset
generate_our_own_data("yosemite_1536x1536")

# For London dataset
generate_our_own_data("london")
```

### Notes
- Images are cropped into 1536x1536 pixel segments to further split into image of size 512x512 after apply Gaussian filter
- Or directly cropped into size 512x125
- Labels are automatically adjusted for cropped segments
- Density maps are generated using gaussian filtering
- Datasets are split into train/validation sets
- Output is saved in HDF5 format for efficient loading
```