# Mapping Enhanced Lifeforms with High Resolution Remote Sensing Data (IGARSS 2023)
Authors: [Minho Kim](https://minho.me), Iryna Dronova, John Radke

High resolution fuel maps are useful for high resolution wildfire simulations and detection of hazards on the landscape. In general, high resolution Enhanced Lifeform Maps (ELMs) are used in conjunction with other data layers to create these fuel maps. However, ELMs are costly to make with substantial manual editing involved. In response, this study uses deep learning-based semantic segmentation models to generate 5-m resolution ELMs (14 classes) in Marin and San Mateo, California using high resolution remote sensing datasets. ELM classes were found to be severely imbalanced, leading to model overfitting. Sample weighted loss functions helped alleviate this issue to an extent. High resolution ELMs are bound to be more valuable with the growing fire risk and landscape heterogeneity, particularly near the wildland urban interface.

All experiments were trained from scratch in PyTorch (v2.0.1) and Python 3.10.4 and were performed using NVIDIA 2080ti GPUs with 12 GB of memory. The code enables parallel processing of GPUs. 

Key Requirements
---------------------
- matplotlib
- pandas
- rasterio
- torch
- torchvision
- sklearn
* See environment.yml for details

Usage
---------------------
1. Install a new conda environment
```
$ conda env create --name elm --file environment.yml
```
2. Activate the new environment and navigate to the "src" folder
```
$ conda activate elm
$ cd src
```
3. Download train and label image files.
4. Modify the file paths to the train and label images in "main.py" lines 197 and 199
5. Run the main.py code. Refer to the test.sh file to view an example to run in terminal.
