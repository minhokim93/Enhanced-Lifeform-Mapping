'''
Accuracy Scores
'''

import numpy as np
import torch

# Define a function to calculate the Dice score
def f1_dice_score(preds, true_mask):
    '''
    https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
    preds should be (B, 25, H, W)
    true_mask should be (B, H, W)
    '''

    f1_batch = []

    for i in range(len(preds)):
        f1_image = []
        img  = torch.Tensor(preds[i])
        mask = torch.Tensor(true_mask[i])

        # Change shape of img from [25, H, W] to [H, W]
        img = torch.argmax(img, dim=0)
    
        for label in range(15):
            if torch.sum(mask == label) != 0:
                area_of_intersect = torch.sum((img == label) * (mask == label))
                area_of_img       = torch.sum(img == label)
                area_of_label     = torch.sum(mask == label)
                f1 = 2*area_of_intersect / (area_of_img + area_of_label)
                f1_image.append(f1)
        
        f1_batch.append(np.mean([tensor.cpu() for tensor in f1_image]))
    return np.mean(f1_batch)