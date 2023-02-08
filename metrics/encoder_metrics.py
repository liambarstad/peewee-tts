import torch
from torch import nn

#   pca of each centroid
#   euclidian distance between centroid and pred
#   avg euclidian distance between other centroids and pred
#   number closest to own centroid
#       precision / recall?

def contrast_metric(sji, sjk, **kwargs):
    with torch.no_grad():
        contrast = 1 - torch.sigmoid(sji) + torch.max(torch.sigmoid(sjk), dim=2).values
        return (contrast.sum() / (len(contrast) * contrast.shape[1])).item()

def loss_metric(loss, **kwargs):
    with torch.no_grad():
        return (loss.sum() / (loss.shape[0] * loss.shape[1])).item()
