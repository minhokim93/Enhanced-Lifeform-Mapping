'''
ELM Semantic Segmentation
Author: Minho Kim
'''

# Libraries
import os, glob, math, datetime, time, random
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

# DL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau # LR Scheduler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Custom
import utils, loss_functions, scores, train, models, hyperparameters

import warnings

# Main Training Loop
class tensorDataset(Dataset):

    def __init__(self, images, masks, augmentations=None):
        self.images = images
        self.masks  = masks
        self.augmentations = augmentations

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]

        if self.augmentations is not None:
            # Apply augmentations to both image and mask
            random.seed(_seed)
            image = self.augmentations(image)
            mask = self.augmentations(mask)

        # Turn on gradient for image
        img = image.detach().clone().requires_grad_(True)
        mask = mask.long()
            
        return img, mask
    
    def __len__(self):
        return len(self.images)
    
def trainer(train_dl=None, test_dl=None, model=None, model_name=None, epochs=None, aux=None, early_stopping=None, device=None):

    # Prepare model and hyperparameters
    train_loss,val_loss, train_acc, val_acc = [], [], [], []
    train_f1score, val_f1score = [],[]
    
    # Train and validation loop
    min_valid_loss = np.inf # Set
    min_valid_acc = 0

    history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[], 'train_f1_score':[], 'test_f1_score':[], 
                'train_recall':[], 'train_precision':[], 'test_recall':[], 'test_precision':[]}

    start = time.time()
    for epoch in range(epochs):
        # print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc, train_precision, train_recall, train_f1 = train.fit(model, train_dl, optimizer, criterion, aux, device)
        val_epoch_loss, val_epoch_acc, val_precision, val_recall, val_f1 = train.validate(model, test_dl, criterion, aux, device)
        
        train_loss = train_epoch_loss 
        test_loss = val_epoch_loss 
        train_acc = train_epoch_acc
        test_acc = val_epoch_acc

        print("Epoch:{}/{} || AVG Training Loss:{:.3f} || AVG Val Loss:{:.3f} || Train Precision :{:.3f} || Val Precision:{:.3f} || Train Recall :{:.3f} || Val Recall:{:.3f} || AVG Train F1:{:.3f} || AVG Val F1:{:.3f} || AVG Training Acc {:.2f} % || AVG Val Acc {:.2f} %"
                        .format(
                            epoch + 1,epochs,
                            train_loss,test_loss,
                            train_precision, val_precision,
                            train_recall, val_recall, 
                            train_f1, val_f1,
                            train_acc*100,test_acc*100))

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_f1_score'].append(train_f1)
        history['test_f1_score'].append(val_f1)
        history['train_recall'].append(train_recall)
        history['train_precision'].append(train_precision)
        history['test_recall'].append(val_recall)
        history['test_precision'].append(val_precision)
        
        lr = optimizer.param_groups[0]['lr']
        early_stopping(test_loss)

        # # Monitor validation loss and save model when val_loss < 
        # if min_valid_loss > test_loss:
        #     print('Validation loss decreased from ', min_valid_loss, ' to ', test_loss, ' --> Saving Model')
        #     # print(f'Validation Loss Decreased({min_valid_loss:.6f} to {test_loss:.6f}) \t\t #####Saving The Model')
        #     min_valid_loss = test_loss
        #     torch.save(model.state_dict(), model_name + ".pt")

        # Monitor validation accuracy and save model when val_f1 >
        if min_valid_acc < val_f1:
            print('Validation F1 increased from ', min_valid_acc, ' to ', val_f1)
            min_valid_acc = val_f1
            
            if min_valid_loss > test_loss:
                print('Validation loss decreased from ', min_valid_loss, ' to ', test_loss, ' --> Saving Model')
                min_valid_loss = test_loss
                torch.save(model.state_dict(), 'models/' + model_name + '_' + modelname + '_seed'+str(_seed)+'.pt')

        if early_stopping.early_stop:
            break
    
    end = time.time()

    print("Training time [minutes] : ", (end-start)/60, ' for model ', model_name)

    return history

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--patch_size', default=256, type=int, help='Patch size')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--epochs', default=100, type=int, help='Epochs')
    parser.add_argument('--seeds', default=42, type=int, help='Seeds for reproducibility')
    parser.add_argument('--threshold', default=0.1, type=int, help='Threshold for class proportion')
    parser.add_argument('--patience', default=15, type=int, help='Early stopping')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--lossf', default=1, type=int, help='Loss functions (1: Cross-entropy | 2: Focal Loss | 3: Weighted focal loss')

    parser.add_argument('--s1', default=None, type=bool, help='Sentinel-1')
    parser.add_argument('--s2', default=None, type=bool, help='Sentinel-2')
    parser.add_argument('--ps', default=True, type=bool, help='Planetscope')
    parser.add_argument('--dsm', default=None, type=bool, help='DSM')

    parser.add_argument('--enc', default=None, type=str, help='resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x4d, resnext101_32x8d, se_resnet50, se_resnet101, se_resnet152, densenet121, densenet169, densenet201, inceptionresnetv2, mobilenet_v2, efficientnet-b-b1')
    parser.add_argument('--dec', default=None, type=str, help='UNET, UNETPLUSPLUS, DEEPLABV3, DEEPLABV3PLUS, MANET, FPN, PSPNET, LINKNET')
    parser.add_argument('--aux', default=None, type=bool, help='AUX parameters for model')
    parser.add_argument('--resnet', default=None, type=int, help='Resnet call')

    parser.add_argument('--plot', default=True, type=bool, help='Plot loss, f1, and confusion matrix')
    parser.add_argument('--save', default=True, type=bool, help='Save results')
    parser.add_argument('--gpu', default='cuda', type=str, help='Single GPU processing: cuda:0 | cuda:1 | cuda:2 | cuda:3')
    parser.add_argument('--modelname', default=None, type=str, help='Add more text to model name')

    args = parser.parse_args()    

    # Hyperparameter
    patch_size = args.patch_size
    _batch_size = args.batch_size
    _epochs = args.epochs
    _seed = args.seeds
    threshold = args.threshold
    patience = args.patience
    lr = args.lr
    lossf = args.lossf
    gpu = args.gpu

    # Input images
    s1 = args.s1
    s2 = args.s2
    ps = args.ps
    dsm = args.dsm
    
    # Model parameters
    enc = args.enc
    dec = args.dec
    if enc == "None": enc = None
    if dec == "None": dec = None
    aux = args.aux
    resnet = args.resnet

    # Other modes
    plot = args.plot
    save = args.save
    modelname = args.modelname

    device = gpu if torch.cuda.is_available() else "cpu"
    print(device)
    # torch.cuda.empty_cache()  # Release GPU memory
    
    warnings.filterwarnings('ignore')
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Input paths
    data_path = '/home/minho/fires/LDARCH289/data_train'
    data_list = sorted(os.listdir(data_path))
    label_path = '/home/minho/fires/LDARCH289/data_label'
    label_list = sorted(os.listdir(label_path))

    # counties = ['marin'] # Determine county folders for dataset stack
    counties = ['marin', 'sanmateo'] # Determine county folders for dataset stack

    # Training patches for each county
    stack1, indexes1, labels1 = utils.train_patches(data_path, 'marin', label_path, patch_size, s1=s1, s2=s2, ps=ps, dsm=dsm, threshold=threshold)
    stack2, indexes2, labels2 = utils.train_patches(data_path, 'sanmateo', label_path, patch_size, s1=s1, s2=s2, ps=ps, dsm=dsm, threshold=threshold)
    imgs = np.vstack((stack1, stack2))
    labels=labels1[0]+labels2[0]

    # Stack images and labels
    imgs[imgs==-9999]= np.nan; nan_mask = np.isnan(imgs); imgs[nan_mask] = 0
    labels = np.array(labels); labels[labels==-9999] = np.nan; nan_mask = np.isnan(labels); labels[nan_mask] = 0
    imgs = utils.minmax_bands(imgs) # Minmax scaling

    # Dataset split
    X_test = None
    X_train, X_val, Y_train, Y_val = train_test_split(imgs, labels, test_size=0.2, random_state=_seed)
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.25, random_state=_seed)

    # Find weights for model loss function (Also n_classes)
    num,count = np.unique(labels, return_counts = True)
    n_classes = len(count)
    # max_class_count = count[np.argmax(count)]
    weights = sum(count) / count
    weights /= max(weights)
    weights = np.append(0, weights) # normalized
    # weights[0] = 0

    X_train_res = np.rollaxis(X_train, 3,1)
    X_val_res = np.rollaxis(X_val, 3,1)

    # Reshape and prepare tensors for Train and Validation sets
    train_x = X_train_res.reshape(X_train_res.shape[0], X_train_res.shape[1], X_train_res.shape[2], X_train_res.shape[3])
    train_x  = torch.from_numpy(X_train_res).float()
    val_x = X_val_res.reshape(X_val_res.shape[0], X_val_res.shape[1], X_val_res.shape[2], X_val_res.shape[3])
    val_x  = torch.from_numpy(X_val_res).float()

    # Create Test set
    if X_test is not None:
        # X_test = minmax_bands(X_test)
        X_test_res = np.rollaxis(X_test, 3,1)
        test_x = X_test_res.reshape(X_test_res.shape[0], X_test_res.shape[1], X_test_res.shape[2], X_test_res.shape[3])
        test_x  = torch.from_numpy(X_test_res).float()
        test_y = Y_test.astype(int)
        test_y = torch.from_numpy(Y_test)
        # print("TESTING :", test_x.shape, test_y.shape)
        del X_test_res
    # del X_train, X_test

    # Converting the target (Labels) into torch format
    train_y = Y_train.astype(int)
    train_y = torch.from_numpy(Y_train)
    val_y = Y_val.astype(int)
    val_y = torch.from_numpy(Y_val)

    # Shape of training data
    # print("TRAINING :", train_x.shape, train_y.shape)
    # print("VALIDATION :", val_x.shape, val_y.shape)

    del X_train_res, X_val_res
    
    # Augmentations
    augmentations = T.Compose([
        T.RandomHorizontalFlip(p=0.5),  # Randomly flip horizontally
        T.RandomVerticalFlip(p=0.5),    # Randomly flip vertically
    ])

    # Set Dataset and DataLoaders
    train_dataset = tensorDataset(train_x, train_y,augmentations=augmentations)
    train_dataloader = DataLoader(train_dataset, batch_size=_batch_size, drop_last=True, shuffle=True)
    val_dataset = tensorDataset(val_x, val_y)
    val_dataloader = DataLoader(val_dataset, batch_size=_batch_size, drop_last=False, shuffle=False)
    if X_test is not None:
        test_dataset = tensorDataset(test_x, test_y)
        test_dataloader = DataLoader(test_dataset, batch_size=_batch_size, drop_last=False, shuffle=False)

    ### Model parameters
    aux = args.aux
    aux_params=dict(
        pooling='avg',             # one of 'avg', 'max'
        dropout=0.5,               # dropout ratio, default is None
        # activation='sigmoid',      # activation function, default is None
        # classes=n_classes+1,                 # define number of output labels
    )

    n_classes = len(count)
    
    # Select model
    # 1. Only decoder
    if enc == None and dec is not None: 
        enc = "only"
        import segmentation_models_pytorch as smp

        # model = models.get_model(decoder=dec, in_channels=imgs.shape[-1], n_classes=n_classes, aux=aux, device=device)
        model = smp.Unet(in_channels=imgs.shape[-1],classes=n_classes+1)
    
    
    # 2. Only encoder --> Custom model (3-layer CNN, resnet)
    elif enc == None and dec == None and resnet:
        enc = ""; dec = "ResNet" + str(resnet)
        # model = models.CNN(in_channels=imgs.shape[-1], num_classes=n_classes+1, dims=32)
        model = models.get_resnet(resnet=resnet, num_input_channels=imgs.shape[-1], num_classes = n_classes)

    # 3. Encoder-decoder
    else:
        model = models.get_model(encoder=enc, decoder=dec, in_channels=imgs.shape[-1], n_classes=n_classes, aux=aux, device=device)



    # if torch.cuda.is_available():
    #     model = nn.DataParallel(model) # Parallel Processing (GPUs)
    model.to(device) # Set to GPU
    model_name = enc + '_' + dec + '_patch' + str(patch_size)
    print("Model : ", model_name + " " + modelname)

    ### Hyperparameters 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    early_stopping = hyperparameters.EarlyStopping(patience)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.5, min_lr=1e-9, verbose=True)

    ### Loss Function
    print("WEIGHTS : ", len(weights))

    gamma_test=2
    if lossf == 0:
        criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    elif lossf == 1:
        criterion = nn.CrossEntropyLoss(torch.Tensor(weights), ignore_index=0).to(device)
    elif lossf == 2:
        criterion = loss_functions.FocalLoss(gamma=gamma_test, alpha=torch.Tensor(weights), ignore_index=0).to(device)
    elif lossf == 3:
        criterion = loss_functions.WeightedFocalLoss(gamma=gamma_test, class_weights=torch.Tensor(weights)).to(device)
    elif lossf == 4:
        criterion = loss_functions.TverskyLoss(n_classes=n_classes, alpha=0.7, beta=0.3).to(device)
    elif lossf == 5:
        criterion = loss_functions.FocalTverskyLoss(n_classes=n_classes, alpha=0.7, beta=0.3).to(device)
    
    # Main run
    history = trainer(train_dl = train_dataloader, test_dl = val_dataloader, 
                      model=model, model_name=model_name, epochs=_epochs, aux=aux,
                      early_stopping=early_stopping, device=device)
    if save:
        pd.DataFrame(history).to_csv('results/'+model_name+'_'+modelname+'_seed'+str(_seed)+'.csv')

    ### Visualization
    train_acc = history['train_acc']
    val_acc = history['test_acc']
    train_f1 = history['train_f1_score']
    val_f1 = history['test_f1_score']
    train_precision = history['train_precision']
    val_precision = history['test_precision']
    train_recall = history['train_recall']
    val_recall = history['test_recall']

    loss = history['train_loss']
    val_loss = history['test_loss']

    if plot: 
        ### Plot
        epochs_range = range(len(train_acc))

        plt.figure(figsize=(10,4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_f1, label='Training F1')
        plt.plot(epochs_range, val_f1, label='Validation F1')
        plt.legend(loc='lower right')
        plt.title(model_name + '+ F1 Score')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title(model_name + '+ Loss')
        plt.show()
        if save:
            plt.savefig('results/resultsplot_'+model_name + '_' + modelname + '_lossf_' + str(lossf) + '_seed'+str(_seed)+'.png')


        # Save results history to CSV file
        # pd.DataFrame(history).to_csv('results/'+model_name + '_grid' + str(patch_size) + '.csv')
        print("Print and save results")

        # ### Predict Phase
        torch.cuda.empty_cache()  # Release GPU memory

        model_path = 'models/' + model_name + '_' + modelname + '_seed'+str(_seed)+'.pt'
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # 1. Predict
        predicted_labels = []
        true_labels = []

        with torch.no_grad():
            for inputs, labels in test_dataloader:
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                if aux:
                    outputs = outputs.squeeze(1)
                
                # Get predicted labels
                _, predicted = torch.max(outputs, dim=1)
                
                # Append predicted and true labels to the lists
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # Convert lists to numpy arrays
        predicted_labels = np.array(predicted_labels)
        true_labels = np.array(true_labels)

        # 2. Confusion Matrix
        from sklearn.metrics import confusion_matrix

        unique_labels = np.unique(np.concatenate((true_labels.flatten(), predicted_labels.flatten())))

        # Compute confusion matrix
        confusion_mat = confusion_matrix(true_labels.flatten(), predicted_labels.flatten(), labels=unique_labels)
        cell_accuracies = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]

        # Plot confusion matrix with grid lines and accuracy percentages
        plt.figure(figsize=(15, 15))
        im=plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.RdBu_r)
        plt.title('Confusion Matrix')
        plt.colorbar(im,fraction=0.046, pad=0.04)
        
        # Add grid lines
        tick_marks = np.arange(len(unique_labels))
        plt.xticks(tick_marks, unique_labels, rotation=45)
        plt.yticks(tick_marks, unique_labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.gca().set_xticks(np.arange(-.5, len(unique_labels), 1), minor=True)
        plt.gca().set_yticks(np.arange(-.5, len(unique_labels), 1), minor=True)
        plt.grid(color='gray', which='minor', linestyle='-', linewidth=1)

        # Add accuracy percentages to the cells
        thresh = confusion_mat.max() / 2.0
        for i in range(len(unique_labels)):
            for j in range(len(unique_labels)):
                plt.text(j, i, str(np.round(cell_accuracies[i,j]*100, 2)) + str('%'),
                        horizontalalignment='center',
                        color='white' if confusion_mat[i, j] > thresh else 'black')

        plt.tight_layout()
        plt.show()

        if save:
            plt.savefig('results/confmatrix_'+model_name + 'lossf_' + str(lossf) + '.png')
        # plt.show()

        # del model
        torch.cuda.empty_cache()  # Release GPU memory        