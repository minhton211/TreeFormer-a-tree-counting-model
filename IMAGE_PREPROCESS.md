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

## Directory Structure
```
Dataset/
├── London/
│   ├── test/
│   ├── train/
│   └── val/
└── Yosemite/
    ├── Dataset_Row_1536x2560/
    │   ├── zone_A/
    │   ├── zone_B/
    │   ├── zone_C/
    │   └── zone_D/
    └── labels.txt
```

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
        ch: Crop height (2560)
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
output_folder = 'Dataset/Yosemite/Dataset_Row_1536x2560'
```

#### Set crop dimensions
```python
crop_width = 1536
crop_height = 2560
```

#### Process images
```python
main_row(input_image_path, input_label_path, output_folder, crop_width, crop_height)
```

#### Generate HDF5 Datasets
```python
# For Yosemite dataset
generate_our_own_data("yosemite_1536x2560")

# For London dataset
generate_our_own_data("london")
```

### Notes
- Images are cropped into 1536x2560 pixel segments to further split into image of size 512x512 after apply Gaussian filter
- Or directly cropped into size 512x125
- Labels are automatically adjusted for cropped segments
- Density maps are generated using gaussian filtering
- Datasets are split into train/validation sets
- Output is saved in HDF5 format for efficient loading
```

This Markdown file provides a comprehensive guide to the image preprocessing steps used in your project.

Similar code found with 2 license types

tinvietle: give me the full markdown code so that i can copy into my file

Workspace: Here is the full Markdown documentation for the image preprocessing steps:

```md
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

## Directory Structure
```
Dataset/
├── London/
│   ├── test/
│   ├── train/
│   └── val/
└── Yosemite/
    ├── Dataset_Row_1536x2560/
    │   ├── zone_A/
    │   ├── zone_B/
    │   ├── zone_C/
    │   └── zone_D/
    └── labels.txt
```

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
        ch: Crop height (2560)
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
output_folder = 'Dataset/Yosemite/Dataset_Row_1536x2560'
```

#### Set crop dimensions
```python
crop_width = 1536
crop_height = 2560
```

#### Process images
```python
main_row(input_image_path, input_label_path, output_folder, crop_width, crop_height)
```

#### Generate HDF5 Datasets
```python
# For Yosemite dataset
generate_our_own_data("yosemite_1536x2560")

# For London dataset
generate_our_own_data("london")
```

### Notes
- Images are cropped into 1536x2560 pixel segments
- Labels are automatically adjusted for cropped segments
- Density maps are generated using gaussian filtering
- Datasets are split into train/validation sets
- Output is saved in HDF5 format for efficient loading
```