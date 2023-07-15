# Mapping Enhanced Lifeforms with High Resolution Remote Sensing Data (IGARSS 2023)
Authors: [Minho Kim](https://minho.me), Iryna Dronova, John Radke


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
