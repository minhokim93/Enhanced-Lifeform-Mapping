'''
Model training and validation (PyTorch)
- Fit
- Validate
'''

import numpy as np
import torch

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, recall_score, precision_score,accuracy_score, balanced_accuracy_score

import torch.nn.functional as F
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")


def fit(model, train_dl, optimizer, criterion, aux, device):

    y_true = []
    y_pred = []

    oa_list, f1_list, precision_list, recall_list = [],[],[],[]
    oa_list2 = []
    total_train_losses,total_train_accuracy,total_train_f1 = [],[],[]

    model.train()
    train_loss = 0.0
    total_samples= 0
    train_corrects = 0

    for i, (inputs, labels) in enumerate(train_dl):

        
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # with torch.set_grad_enabled(True):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        train_corrects += torch.sum(preds == labels.data)

        # # Convert y_true and y_pred to numpy arrays
        y_true = np.array(labels.cpu().numpy())
        y_pred = np.array(preds.cpu().numpy())
        y_true = y_true.reshape(y_true.shape[0], -1)
        y_pred = y_pred.reshape(y_pred.shape[0], -1)     

        # # Convert to multilabel-indicator format
        mlb = MultiLabelBinarizer()
        y_true_multilabel = mlb.fit_transform(y_true)
        y_pred_multilabel = mlb.transform(y_pred)

        # Compute precision, recall, and F1 score
        overall_acc = accuracy_score(y_true_multilabel.flatten(), y_pred_multilabel.flatten())

        precision = precision_score(y_true_multilabel, y_pred_multilabel, average='macro', zero_division=0)
        recall = recall_score(y_true_multilabel, y_pred_multilabel, average='macro', zero_division=0)
        f1 = f1_score(y_true_multilabel, y_pred_multilabel, average='macro', zero_division=0)
        # print(f'TRAIN SINGLE BATCH | loss: {train_loss:.4f}, f1: {f1:.3f}, accuracy: {overall_acc:.3f}')

    oa_list.append(overall_acc)
    f1_list.append(f1)
    precision_list.append(precision)
    recall_list.append(recall)

    print(f'TRAIN BATCH | loss: {np.mean(train_loss):.4f}, f1: {np.mean(f1_list):.3f}, accuracy: {np.mean(oa_list):.3f}')
      
    return train_loss/total_samples, np.mean(oa_list), np.mean(precision_list), np.mean(recall_list), np.mean(f1_list)

def validate(model, test_dl, criterion, aux, device):

    y_true,y_pred = [],[]
    oa_list, f1_list, precision_list, recall_list = [],[],[],[]
    oa_list2 = []
    total_val_losses,total_val_accuracy,total_val_f1 = [],[],[]

    model.eval()
    val_loss = 0.0
    total_samples = 0
    val_corrects = 0

    for inputs, labels in test_dl:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            # preds = preds + 1
            loss = criterion(outputs, labels)

        if not torch.isnan(loss):
            val_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

        # Convert y_true and y_pred to numpy arrays
        y_true = np.array(labels.cpu().numpy())
        y_pred = np.array(preds.cpu().numpy())
        y_true = y_true.reshape(y_true.shape[0], -1)
        y_pred = y_pred.reshape(y_pred.shape[0], -1)     

        # Convert to multilabel-indicator format
        mlb = MultiLabelBinarizer()
        y_true_multilabel = mlb.fit_transform(y_true)
        y_pred_multilabel = mlb.transform(y_pred)
        
        # Compute precision, recall, and F1 score
        overall_acc = accuracy_score(y_true_multilabel.flatten(), y_pred_multilabel.flatten())

        precision = precision_score(y_true_multilabel, y_pred_multilabel, average='macro', zero_division=0)
        recall = recall_score(y_true_multilabel, y_pred_multilabel, average='macro', zero_division=0)
        f1 = f1_score(y_true_multilabel, y_pred_multilabel, average='macro', zero_division=0)
        # print(f'VALIDATION SINGLE BATCH | loss: {val_loss:.4f}, f1: {f1:.3f}, accuracy: {overall_acc:.3f}')

    oa_list.append(overall_acc)
    f1_list.append(f1)
    precision_list.append(precision)
    recall_list.append(recall)

    print(f'VALIDATION BATCH | loss: {np.mean(val_loss):.4f}, f1: {np.mean(f1_list):.3f}, accuracy: {np.mean(oa_list):.3f}')
    
    return val_loss/total_samples, np.mean(oa_list), np.mean(precision_list), np.mean(recall_list), np.mean(f1_list)