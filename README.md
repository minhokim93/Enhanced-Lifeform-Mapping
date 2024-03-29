# SEMANTIC SEGMENTATION OF ENHANCED LANDFORM MAPS USING HIGH RESOLUTION SATELLITE IMAGES (Oral Presentation @ IGARSS 2023)
Authors: [Minho Kim](https://minho.me), Iryna Dronova, John Radke

High resolution fuel maps are useful for high resolution wildfire simulations and detection of hazards on the landscape. In general, high resolution Enhanced Lifeform Maps (ELMs) are used in conjunction with other data layers to create these fuel maps. However, ELMs are costly to make with substantial manual editing involved. In response, this study uses deep learning-based semantic segmentation models to generate 5-m resolution ELMs (14 classes) in Marin and San Mateo, California using high resolution remote sensing datasets. ELM classes were found to be severely imbalanced, leading to model overfitting. Sample weighted loss functions helped alleviate this issue to an extent. High resolution ELMs are bound to be more valuable with the growing fire risk and landscape heterogeneity, particularly near the wildland urban interface. [Paper Link](https://cmsfiles.s3.amazonaws.com/ig23/proceedings/papers/0005491.pdf?X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAZW5HH2C3GPEL7I72%2F20230716%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230716T082102Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3600&X-Amz-Signature=e37289eb5ce1b103748215f59df5329e0b1d1749dddfb543360e73b6ac7c00cc)

All experiments were trained from scratch in PyTorch (v2.0.1) and Python 3.10.4 and were performed using NVIDIA 2080ti GPUs with 12 GB of memory. The code enables parallel processing of GPUs. ELMs were acquired from http://pacificvegmap.org, maintained by Tukman Geospatial with funding from CALFIRE (Thank You!).

The figure below shows an overview of the study:

<p align="center">
  <img src="./figures/figure4.jpg" alt="Image" />
</p>

Datasets
---------------------
Input data include Planetscope, Sentinel-2, and Sentinel-1 imagery. High resolution DSM was acquired from Pacific Veg Map.
- Marin Data [(Link)](https://drive.google.com/file/d/1gAf7L-5UXDd7g0zLy3_jZarysBM2-ate/view?usp=share_link)
- San Mateo Data [(Link)](https://drive.google.com/file/d/1G2Z0OT_i3o5Lx8ap4dYH5exMWQP0153v/view?usp=share_link)
- Ground Truth Labels [(Link)](https://drive.google.com/file/d/15O7bEEh3B2UiUE-nVCE5GJ07iFMrwJLN/view?usp=share_link)

Planetscope data can be acquired from Planet Labs.

Key Requirements
---------------------
- matplotlib
- pandas
- rasterio
- torch
- torchvision
- sklearn
  
See environment.yml for details

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

Citation
---------------------
**Please cite the following paper if this code is useful and helpful for your research.**

