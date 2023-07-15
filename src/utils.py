'''
Utility Functions for preprocessing and path directories
'''

import numpy as np
import matplotlib.pyplot as plt
import os, glob
from patchify import patchify, unpatchify
import torch
import rasterio

def clip(img, percentile):
    out = np.zeros_like(img.shape[2])
    for i in range(img.shape[2]):
        a = 0 
        b = 255 
        c = np.percentile(img[:,:,i], percentile)
        d = np.percentile(img[:,:,i], 100 - percentile)        
        t = a + (img[:,:,i] - c) * (b - a) / (d - c)    
        t[t<a] = a
        t[t>b] = b
        img[:,:,i] =t
    rgb_out = img.astype(np.uint8)   
    return rgb_out
    
def get_paths(county, label):
    # Image patches
    img_patch_dir = os.path.join(base_path,'images_test')
    label_patch_dir = os.path.join(base_path,'labels_test')

    # List directories
    dsm_dir = os.path.join(img_patch_dir, 'dsm')
    ps_img_dir = os.path.join(img_patch_dir, 'ps_summer')
    s1_img_dir = os.path.join(img_patch_dir, 's1_summer')
    s2_img_dir = os.path.join(img_patch_dir, 's2_summer')

    dsm_patches = sorted(glob.glob(os.path.join(img_patch_dir, 'dsm') + '/*.tif'))
    ps_img_patches = sorted(glob.glob(os.path.join(img_patch_dir, 'ps_summer') + '/*.tif'))
    s1_img_patches = sorted(glob.glob(os.path.join(img_patch_dir, 's1_summer') + '/*.tif'))
    s2_img_patches = sorted(glob.glob(os.path.join(img_patch_dir, 's2_summer') + '/*.tif'))
    label_patches = sorted(glob.glob(label_patch_dir + '/' + label + '/*.tif'))

    return img_patch_dir,label_patch_dir,dsm_dir,ps_img_dir,s1_img_dir,s2_img_dir,ps_img_patches
    
def create_stack(*arrays):
    stack = np.asarray(arrays)
    return np.rollaxis(stack, 0, 4)

def prep_data(image, label, patch_size, mode, img_name, threshold):
    
    instances,instances_labels, indexes = [],[],[]

    size_x = (image.shape[1] // patch_size) * patch_size  # width to the nearest size divisible by patch size
    size_y = (image.shape[0] // patch_size) * patch_size  # height to the nearest size divisible by patch size
    # Extract patches from each image, step=patch_size means no overlap
    step=patch_size
    
    if mode == "train":
        if "dsm" in img_name:
            image = image[0:size_x, 0:size_y]
            patch_img = patchify(image, (patch_size, patch_size), step=patch_size)
        else: 
            n_bands = image.shape[2]
            image = image[0:size_x, 0:size_y,:]
            patch_img = patchify(image, (patch_size, patch_size, n_bands), step=patch_size)

    lbl = label[0:size_x, 0:size_y]
    patch_lbl = patchify(lbl, (patch_size, patch_size), step=patch_size)
    patch_lbl[patch_lbl==-9999] = 0
    labels = patch_lbl

    
    # iterate over patch axis
    i=0
    for j in range(patch_img.shape[0]):
        for k in range(patch_img.shape[1]):
            # print("j :", j, "k :", k)

            single_img = patch_img[j, k] # patches are located like a grid. use (j, k) indices to extract single patched image
            single_lbl = labels[j, k] # patches are located like a grid. use (j, k) indices to extract single patched image
            
            # lbl = single_lbl.argmax(axis=-1)+1
            lbl = single_lbl
            count, num = np.unique(lbl, return_counts=True)
            
            if len(count) > 1 and 0 not in count:
            # if 1 in count and num[0]/patch_size**2 < threshold:
            # if len(count) > 1 and 0 in count and num[0]/patch_size**2 < threshold: 
                ### If the total number of classes > 1 AND "class 1" is included
                ### Fraction (num[0] / patch_size**2) is the proportion of 

                # print("Filtered Image #: ",i)
                # print("Filtered freq proportion: ", num[0]/patch_size**2)
                # print("Number of unique classes: ", count, "and pixel frequency of classes: ", num)
                # pass

                # Drop extra dimension from patchify
            # else:
                # print("Image #: ",i)

                instances.append(np.squeeze(single_img))
                instances_labels.append(np.squeeze(single_lbl))
            
            # print("{}% remaining".format(np.round(((patch_img.shape[0]*patch_img.shape[1]) - i)/(patch_img.shape[0]*patch_img.shape[1])*100),2))
            i += 1

    indexes.append(len(instances))
    # instances[instances==-9999] = 0

    return instances, instances_labels, indexes

def patches(mode=None, img_path=None, img_name=None, label_path=None, patch_size=256,
            s1=None, s2=None, ps=None, dsm=None, threshold=None):

    img_stack=None

    # Training datasets
    if mode == "train": 
        # Reads images
        if "ps" in img_name and ps is not None:
            with rasterio.open(img_path, 'r') as src:
                b1,b2,b3,b4 = src.read([3,2,1,4])
            img_stack = np.dstack((b1,b2,b3,b4))
        elif  's1' in img_name and s1 is not None:
            with rasterio.open(img_path, 'r') as src:
                vv,vh = src.read([1,2])
            img_stack = np.dstack((vv,vh))
        elif "s2" in img_name and s2 is not None:
            # Sentinel-2 : B5, B6, B7, B9, B10, B11, B12 --> Total 8 bands
            with rasterio.open(img_path, 'r') as src:
                b5,b6,b7,b9,b10,b11,b12 = src.read([2,3,4,5,6,7,8])
            img_stack = np.dstack((b5,b6,b7,b9,b10,b11,b12))
        elif "dsm" in img_name and dsm is not None:
            with rasterio.open(img_path, 'r') as src:
                dsm = src.read(1)
            img_stack = dsm

    # # Labels            
    # elif mode == "labels":
    with rasterio.open(label_path, 'r') as src:
        label = src.read(1)
    #     img_stack=None
        
    if img_stack is not None:
        imgs, labels, idxs = prep_data(img_stack,label,patch_size, mode, img_name, threshold)

        return imgs, labels, idxs

def omit(dir_list, term):
    filtered_dirs = []
    
    for dir_path in dir_list: 
        dir_name = os.path.basename(dir_path)

        # Check if the term is present in the directory name
        if term in dir_name:
            continue

        filtered_dirs.append(dir_path)
    
    return filtered_dirs
    

def train_patches(data_path=None, county=None, label_path=None, patch_size=256,
                s1=True, s2=True, ps=True, dsm=True, threshold=0.8):

    data_list = sorted(os.listdir(os.path.join(data_path, county)))

    # Select input image features (S1, S2, PS, DSM)
    if s1 == None: 
        print("Omitting S1")
        data_list = omit(data_list, 's1')
    if s2 == None: 
        print("Omitting S2")
        data_list = omit(data_list, 's2')
    if ps == None:
        print("Omitting PS")
        data_list = omit(data_list, 'ps')
    if dsm == None:
        print("Omitting DSM")
        data_list = omit(data_list, 'dsm')

    print("Data List: ", data_list)

    stack, indexes, labels_list = [],[],[]

    # Preprocess and patchify for all images in data_list
    for img in data_list:
        mode='train'

        img_name = img.split('.tif')[0]
        img_path = os.path.join(data_path, county, img)
        print(img_name)
        lbl_path = os.path.join(label_path, county+'_elm.tif')

        # Patchify
        imgs, labels, idxs = patches(mode=mode, img_path=img_path, img_name=img_name, label_path=lbl_path, patch_size=patch_size,
                                    s1=s1, s2=s2, ps=ps, dsm=dsm, threshold=threshold)
        imgs = np.array(imgs)

        # Expand 1-band images to 4D tensors
        if len(np.array(imgs).shape) < 4: 
            imgs = np.expand_dims(imgs, -1)
        
        stack.append(imgs)
        indexes.append(idxs)
        labels_list.append(labels)

    imgs = np.concatenate((stack),axis=-1)

    return imgs, indexes, labels_list


def minmax(img):
    a = ( img - np.nanmin(img) ) / ( np.nanmin(img) + np.nanmax(img) )
    return a       

def minmax_bands(image):
    rescaled = [minmax(image[:,:,:,i]) for i in range(image.shape[-1])]
    stack = np.stack(rescaled) # [bands, x, y, batches]
    stack = np.rollaxis(stack, 0, 4) # [batches, x, y, bands]

    return stack


# Set class weights for weighted focal loss
def class_weights(data):
    import torch

    unique_values, value_counts = np.unique(data, return_counts=True)

    total_samples = sum(value_counts)
    class_frequencies = [count / total_samples for count in value_counts]
    class_weights = [1 / freq for freq in class_frequencies]
    class_weights = class_weights / np.sum(class_weights)
    class_weights = torch.tensor(class_weights)
    # class_weights[torch.isnan(class_weights)] = 0  # Set weights for null values to 0

    return class_weights


def to_onehot(label_tensor=None, num_classes=None, device=None):
    if num_classes is None:
        num_classes = int(label_tensor.max().detach().item() + 1)

    tensor_onehot = torch.zeros(
        label_tensor.shape[0],
        num_classes,
        *label_tensor.shape[1:],
        dtype=label_tensor.dtype,
        device=label_tensor.device,
    )
    index = label_tensor.long().unsqueeze(1).expand_as(tensor_onehot)
    
    return tensor_onehot.scatter_(1, index, 1.0).squeeze(0)
